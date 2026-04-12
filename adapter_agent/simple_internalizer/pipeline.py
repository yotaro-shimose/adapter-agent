import asyncio
import itertools
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Self

import tinker
from more_itertools import chunked
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from prisma import Prisma
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.rl import Trajectory
from tinker_cookbook.rl.types import TokensWithLogprobs, Transition
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import compute_grpo_loss
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import OptimizerParams
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient

from .evaluate_worker import EvaluateWorker
from .executor import InternalizeExecutor
from .types import PipelineConfig, PreparedTasks

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


class SimplePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        service_client: tinker.ServiceClient,
        training_client: tinker.TrainingClient,
        solver_model: TinkerModel,
        generator: GeneratorAgent,
        executor: InternalizeExecutor,
        ml_logger: MLLogger,
        prisma_client: Prisma,
        reference_client: tinker.SamplingClient,
        log_dir: Path,
    ):
        self.config = config
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.service_client = service_client
        self.training_client = training_client
        self.solver_model = solver_model
        self.generator = generator
        self.executor = executor
        self.ml_logger = ml_logger
        self.prisma_client = prisma_client
        self.reference_client = reference_client

        # New components for decoupled evaluation
        self.shared_sampling_client = SharedSamplingClient(
            self.solver_model.sampling_client
        )
        self.log_dir = log_dir
        self.eval_trigger = asyncio.Event()
        self.evaluate_worker: Optional[EvaluateWorker] = None

        self.rl_tasks_queue: asyncio.Queue[tuple[Knowledge, Any]] = asyncio.Queue()
        self.rl_results_queue: asyncio.Queue[RLGroup] = asyncio.Queue()

    @classmethod
    async def create(cls, config: PipelineConfig, rust_doc_analyzer: Any) -> Self:
        gemini = get_gemini()
        generator = GeneratorAgent(model=gemini, rust_doc_analyzer=rust_doc_analyzer)
        verifier = Verifier(model=gemini, rust_doc_analyzer=rust_doc_analyzer)

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_loading_settings.model_name,
            rank=config.model_loading_settings.lora_rank,
        )

        if config.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Loading trainer state from {config.model_loading_settings.resume_trainer_path}..."
            )
            await training_client.load_state_async(
                config.model_loading_settings.resume_trainer_path
            )

        tinker_model, _, _ = setup_tinkermodel(
            model_name=config.model_loading_settings.model_name,
            service_client=service_client,
            path=config.model_loading_settings.resume_sampler_path,
        )

        log_dir = Path("logs") / "simple_internalizer" / config.model_loading_settings.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": config.model_loading_settings.model_name,
                "library": config.library_name,
                "init_sft_steps": config.init_sft_steps,
                "iter_sft_steps": config.iter_sft_steps,
                "sft_batch_size": config.sft_batch_size,
                "lora_rank": config.model_loading_settings.lora_rank,
            },
        )

        db_manager = PostgresDB()
        await db_manager.connect()
        client = await db_manager.get_client()
        await client.simpletrainrun.upsert(
            where={"id": config.simple_train_id},
            data={"create": {"id": config.simple_train_id}, "update": {}},
        )

        runtime_pool = RuntimePool(
            config.runtime_settings, max_size=config.runtime_pool_size
        )
        executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)

        # Initialize reference_client from the training_client state (base or loaded checkpoint)
        reference_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )
        logger.info("Initialized reference client from training state.")

        return cls(
            config=config,
            service_client=service_client,
            training_client=training_client,
            solver_model=tinker_model,
            generator=generator,
            executor=executor,
            ml_logger=ml_logger,
            prisma_client=client,
            reference_client=reference_client,
            log_dir=log_dir,
        )

    async def _generate_and_cache(
        self, knowledge: Knowledge, count: int, prefix: str
    ) -> list[Any]:
        cache_file = self.cache_dir / f"{knowledge.id}_{prefix}_{count}.pkl"
        if cache_file.exists():
            logger.info(
                f"Loading {count} {prefix} QRAs for '{knowledge.title}' from cache."
            )
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info(f"Generating {count} {prefix} QRAs for '{knowledge.title}'...")
        sem = asyncio.Semaphore(self.config.generation_concurrency)

        async def _gen():
            async with sem:
                while True:
                    qra = await self.generator.generate_sft(knowledge)
                    if qra:
                        return qra
                    await asyncio.sleep(0.1)

        results = await asyncio.gather(*[_gen() for _ in range(count)])
        with open(cache_file, "wb") as f:
            pickle.dump(results, f)
        return results

    async def run(self) -> None:
        prepared = await self._prepare_knowledge_tasks()
        sft_batch_iter = self._create_sft_batch_iterator(prepared.sft_qras)

        # Initialize EvaluateWorker
        self.evaluate_worker = EvaluateWorker(
            config=self.config,
            executor=self.executor,
            ml_logger=self.ml_logger,
            prisma_client=self.prisma_client,
            shared_sampling_client=self.shared_sampling_client,
            renderer=self.solver_model.renderer,
            eval_tasks=prepared.eval_tasks,
            extra_eval_suites=self.config.extra_eval_suites,
            trigger=self.eval_trigger,
        )
        eval_worker_task = asyncio.create_task(self.evaluate_worker.run_loop())

        if self.config.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Skipping initial SFT because resume_trainer_path is provided: {self.config.model_loading_settings.resume_trainer_path}"
            )
        else:
            logger.info(f"Running initial {self.config.init_sft_steps} steps of SFT...")
            await self._run_sft_steps(sft_batch_iter, self.config.init_sft_steps)

            if self.config.init_sft_steps > 0:
                logger.info("Saving initial SFT checkpoint...")
                await checkpoint_utils.save_checkpoint_async(
                    training_client=self.training_client,
                    name="init_sft",
                    log_path=str(self.log_dir),
                    loop_state={"step": self.config.init_sft_steps},
                    kind="both",
                    ttl_seconds=self.config.ttl_seconds,
                )

                # Update reference client to the post-SFT state
                self.reference_client = (
                    await self.training_client.save_weights_and_get_sampling_client_async()
                )
                logger.info("Updated reference client to SFT state.")

        spawner_task = None
        worker_tasks = []

        if not self.config.stop_grpo:
            spawner_task, worker_tasks = await self._start_rl_workers()

        try:
            for iteration in range(self.config.max_iterations):
                logger.info(
                    f"--- Iteration {iteration + 1} (RL{' STOPPED' if self.config.stop_grpo else ''}) ---"
                )

                if self.config.stop_grpo:
                    await asyncio.sleep(10)  # Wait if purely evaluating
                    continue

                batch_groups = await self._collect_rl_batch(self.config.rl_batch_size)

                # Consume all currently available samples in the queue to minimize off-policy-ness
                additional_count = 0
                while not self.rl_results_queue.empty():
                    batch_groups.append(self.rl_results_queue.get_nowait())
                    additional_count += 1

                if additional_count > 0:
                    logger.info(
                        f"Drained additional {additional_count} samples from queue."
                    )

                logger.info(
                    f"Running RL (GRPO) update on collected batch of size {len(batch_groups)}..."
                )
                await self._run_grpo_update(batch_groups)

        finally:
            eval_worker_task.cancel()
            if spawner_task:
                spawner_task.cancel()
            for t in worker_tasks:
                t.cancel()

            tasks_to_wait = [eval_worker_task]
            if spawner_task:
                tasks_to_wait.append(spawner_task)
            tasks_to_wait.extend(worker_tasks)
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _start_rl_workers(self) -> tuple[asyncio.Task, list[asyncio.Task]]:
        rl_tasks = await self._prepare_rl_tasks()
        for k, qra in rl_tasks:
            await self.rl_tasks_queue.put((k, qra))

        num_workers = self.config.rl_worker_count
        worker_tasks: list[asyncio.Task] = []

        async def _staggered_spawner():
            for i in range(num_workers):
                worker_tasks.append(asyncio.create_task(self._rl_worker()))
                if i < num_workers - 1:
                    await asyncio.sleep(self.config.rl_worker_stagger_s)

        spawner_task = asyncio.create_task(_staggered_spawner())
        return spawner_task, worker_tasks

    async def _rl_worker(self) -> None:
        if self.evaluate_worker is None:
            raise RuntimeError("EvaluateWorker not initialized")
        system_prompt = self.evaluate_worker.system_prompt
        while True:
            try:
                k, qra = await self.rl_tasks_queue.get()
                # Get current shared client snapshot
                indexed_client = self.shared_sampling_client.get_client()

                group = await self._collect_rl_group(
                    system_prompt, qra, k, indexed_client
                )
                if group:
                    await self.rl_results_queue.put(group)

                self.rl_tasks_queue.task_done()
                await self.rl_tasks_queue.put((k, qra))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"RL worker error: {e}")
                await asyncio.sleep(1)

    async def _collect_rl_group(
        self, system_prompt: str, qra: Any, knowledge: Knowledge, indexed_client: Any
    ) -> RLGroup | None:
        model_input = self.solver_model.renderer.build_generation_prompt(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=qra.question),
            ]
        )

        sample_results = await indexed_client.client.sample_async(
            prompt=model_input,
            num_samples=self.config.rl_rollout,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )

        async def _verify_single_rollout(seq) -> tuple[Trajectory, float]:
            tokens, logprobs = seq.tokens, seq.logprobs
            ac = TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)
            trajectory = Trajectory(
                transitions=[
                    Transition(ob=model_input, ac=ac, reward=0.0, episode_done=True)
                ],
                final_ob=model_input,
            )

            msg, ok = self.solver_model.renderer.parse_response(tokens)
            content = msg.get("content") if ok else None
            is_success, reasoning, answer_text, exec_out, verif_out = (
                False,
                "",
                "",
                "",
                "",
            )

            if ok and content:
                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "thinking":
                            reasoning += part["thinking"]
                        elif part["type"] == "text":
                            answer_text += part["text"]
                else:
                    answer_text = str(content)

                outcome = await self.executor.run_execution_and_verification(
                    qra.question, reasoning, answer_text
                )
                is_success, exec_out, verif_out = (
                    outcome.success,
                    outcome.execution_output,
                    outcome.verification_output,
                )
            else:
                verif_out = "Parse failed."

            try:
                await self.prisma_client.simpletrajectory.create(
                    data={
                        "simple_train_id": self.config.simple_train_id,
                        "knowledge_id": knowledge.id,
                        "knowledge_title": knowledge.title,
                        "step": indexed_client.version,
                        "question": qra.question,
                        "reasoning": reasoning,
                        "answer": answer_text,
                        "success": is_success,
                        "execution_output": exec_out,
                        "verification_output": verif_out,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record RL trajectory: {e}")

            return trajectory, (1.0 if is_success else 0.0)

        rollout_results = await asyncio.gather(
            *[_verify_single_rollout(seq) for seq in sample_results.sequences]
        )
        trajectories, rewards = (
            [r[0] for r in rollout_results],
            [r[1] for r in rollout_results],
        )

        return RLGroup(trajectories=trajectories, rewards=rewards)

    async def _collect_rl_batch(self, batch_size: int) -> list[RLGroup]:
        logger.info(f"Waiting for a batch of {batch_size} RL groups...")
        batch_groups = []
        while len(batch_groups) < batch_size:
            try:
                group = await asyncio.wait_for(
                    self.rl_results_queue.get(), timeout=10.0
                )
                batch_groups.append(group)
            except asyncio.TimeoutError:
                self.ml_logger.log_metrics(
                    {"status/collected_rl_groups": len(batch_groups)}
                )
        return batch_groups

    async def _run_grpo_update(self, batch_groups: list[RLGroup]) -> None:
        if not batch_groups:
            return

        # Calculate mean reward across all trajectories in all groups
        all_rewards = [r for g in batch_groups for r in g.rewards]
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        # Filter for valid groups (those with reward variance)
        valid_groups = [g for g in batch_groups if max(g.rewards) > min(g.rewards)]

        optimizer_params = OptimizerParams(
            adam_params=self.config.rl_adam_params,
            loss_fn=self.config.rl_loss_fn,
            num_steps=1,
            kl_penalty_coef=self.config.kl_penalty_coef,
            kl_discount_factor=self.config.kl_discount_factor,
        )

        res = None
        if valid_groups:
            res = await compute_grpo_loss(
                valid_groups,
                self.training_client,
                optimizer_params,
                kl_reference_client=self.reference_client,
            )

        if res:
            opt_future = await self.training_client.optim_step_async(
                self.config.rl_adam_params
            )
            await opt_future.result_async()

            metrics = {f"rl/{k}": v for k, v in res.metrics.items()}
        else:
            metrics = {}

        metrics.update(
            {
                "rl/mean_reward": mean_reward,
                "rl/num_valid_groups": len(valid_groups),
                "rl/num_total_groups": len(batch_groups),
            }
        )
        self.ml_logger.log_metrics(metrics)

        if res:
            logger.info("Synchronizing sampling weights and triggering evaluation...")
            new_client = (
                await self.training_client.save_weights_and_get_sampling_client_async()
            )
            self.shared_sampling_client.update_client(new_client)
            self.eval_trigger.set()

    async def _run_sft_steps(self, batch_iter, num_steps: int) -> None:
        if num_steps == 0:
            return
        adam_params = self.config.adam_params
        for i in range(num_steps):
            fwd_future = await self.training_client.forward_backward_async(
                data=next(batch_iter), loss_fn="cross_entropy"
            )
            opt_future = await self.training_client.optim_step_async(adam_params)
            fwd_res = await fwd_future.result_async()
            await opt_future.result_async()

            self.ml_logger.log_metrics(
                {f"sft/{k}": v for k, v in fwd_res.metrics.items()}
            )

        logger.info(
            "Synchronizing sampling weights after SFT and triggering evaluation..."
        )
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.shared_sampling_client.update_client(new_client)
        self.eval_trigger.set()

    async def _prepare_knowledge_tasks(self) -> PreparedTasks:
        async def _prep_single(k: Knowledge):
            s_qras = await self._generate_and_cache(k, self.config.k_sft, "sft")
            e_qras = await self._generate_and_cache(k, self.config.k_eval, "eval")
            return s_qras, (k, e_qras[0])

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.config.knowledge_list]
        )
        sft_qras = [q for r in results for q in r[0]]
        eval_tasks = [r[1] for r in results]
        return PreparedTasks(sft_qras=sft_qras, eval_tasks=eval_tasks)

    async def _prepare_rl_tasks(self) -> list[tuple[Knowledge, Any]]:
        async def _prep_single(k: Knowledge):
            tasks = await self._generate_and_cache(k, self.config.k_rl, "rl")
            return [(k, t) for t in tasks]

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.config.knowledge_list]
        )
        return list(itertools.chain.from_iterable(results))

    def _create_sft_batch_iterator(self, sft_qras: list[Any]):
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_loading_settings.model_name
        )
        datums = [
            conversation_to_datum(
                conversation=[
                    Message(role="user", content=q.question),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingPart(type="thinking", thinking=q.reasoning),
                            TextPart(type="text", text=f"\n\n{q.answer}"),
                        ],
                    ),
                ],
                renderer=renderer,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_length=None,
            )
            for q in sft_qras
        ]
        batches = list(chunked(datums, self.config.sft_batch_size))
        if not batches:
            raise ValueError("No batches created.")
        return itertools.cycle(batches)
