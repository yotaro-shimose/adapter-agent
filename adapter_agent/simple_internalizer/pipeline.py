import asyncio
import itertools
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Self

import tinker
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from prisma import Prisma
from tinker import Datum
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.rl import Trajectory
from tinker_cookbook.rl.types import TokensWithLogprobs, Transition
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import compute_grpo_loss
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge, Task
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import OptimizerParams
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient

from .evaluate_worker import EvaluateWorker
from .executor import InternalizeExecutor
from .types import PipelineConfig, PreparedTasks, RLSource, SeedSuite

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class RLBatchState:
    """Accumulates RL groups with decoupled cadences:
    - `valid_buffer` feeds training (pops exactly `target_valid` when ready).
    - `window_groups` feeds rollout metrics (flushes every `metrics_window`
      groups over ALL produced groups, valid or not).
    Surplus valid groups are retained across calls — none are discarded.
    """

    target_valid: int
    metrics_window: int
    valid_buffer: list[RLGroup] = field(default_factory=list)
    window_groups: list[RLGroup] = field(default_factory=list)

    def add(self, group: RLGroup) -> dict[str, float] | None:
        self.window_groups.append(group)
        if max(group.rewards) > min(group.rewards):
            self.valid_buffer.append(group)
        if len(self.window_groups) >= self.metrics_window:
            return self._flush_window()
        return None

    def _flush_window(self) -> dict[str, float]:
        all_rewards = [r for g in self.window_groups for r in g.rewards]
        num_valid = sum(
            1 for g in self.window_groups if max(g.rewards) > min(g.rewards)
        )
        metrics = {
            "rollout/mean_reward": sum(all_rewards) / len(all_rewards),
            "rollout/valid_group_ratio": num_valid / len(self.window_groups),
            "rollout/window_size": float(len(self.window_groups)),
        }
        self.window_groups = []
        return metrics

    def ready(self) -> bool:
        return len(self.valid_buffer) >= self.target_valid

    def pop_batch(self) -> list[RLGroup]:
        batch = self.valid_buffer[: self.target_valid]
        self.valid_buffer = self.valid_buffer[self.target_valid :]
        return batch


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
        knowledge_list: list[Knowledge],
        seed_suites: list[SeedSuite],
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
        self.knowledge_list = knowledge_list
        self.seed_suites = seed_suites

        # New components for decoupled evaluation
        self.shared_sampling_client = SharedSamplingClient(
            self.solver_model.sampling_client
        )
        self.log_dir = log_dir
        self.eval_trigger = asyncio.Event()
        self.evaluate_worker: Optional[EvaluateWorker] = None

        self.rl_tasks_queue: asyncio.Queue[tuple[Task, RLSource]] = asyncio.Queue()
        self.rl_results_queue: asyncio.Queue[RLGroup] = asyncio.Queue()
        self._rl_batch_state = RLBatchState(
            target_valid=config.rl_batch_size,
            metrics_window=config.rl_metrics_window,
        )

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

        log_dir = (
            Path("logs")
            / "simple_internalizer"
            / config.model_loading_settings.model_name
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": config.model_loading_settings.model_name,
                "library": config.library_name,
                "sft_epochs": config.sft_epochs,
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

        logger.info(f"Loading granular knowledge from run ID: {config.granular_id}")
        granulars = await client.granularknowledge.find_many(
            where={"simple_train_id": config.granular_id}
        )
        if not granulars:
            raise ValueError(
                f"No granular knowledge found for granular-id={config.granular_id}."
            )
        knowledge_list = [
            Knowledge(id=g.id, title=g.title, content=g.content) for g in granulars
        ]
        logger.info(
            f"Loaded {len(knowledge_list)} granular knowledge items "
            f"(granular-id={config.granular_id})."
        )

        study_solved_tasks = await cls._load_solved_seed_tasks(
            client, config.study_experiment_id
        )
        logger.info(
            f"Loaded {len(study_solved_tasks)} solved seed tasks from experiment "
            f"'{config.study_experiment_id}'."
        )
        study_solved_suite = SeedSuite(
            name="study_solved",
            tasks=study_solved_tasks,
            for_rl=True,
            for_eval=True,
        )
        seed_suites = [study_solved_suite, *config.extra_seed_suites]
        for s in config.extra_seed_suites:
            logger.info(
                f"Registered extra seed suite '{s.name}' with {len(s.tasks)} tasks "
                f"(for_rl={s.for_rl}, for_eval={s.for_eval})."
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
            knowledge_list=knowledge_list,
            seed_suites=seed_suites,
        )

    async def _generate_and_cache(
        self,
        knowledge: Knowledge,
        count: int,
        prefix: str,
        verify: bool = True,
        is_coding: bool = True,
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
                    if is_coding:
                        qra = await self.generator.generate_sft(knowledge)
                    else:
                        qra = await self.generator.generate_sft_noncode(knowledge)

                    if not qra:
                        await asyncio.sleep(0.1)
                        continue

                    if not verify:
                        return qra

                    # Verification loop
                    for attempt in range(self.config.max_fix_attempts + 1):
                        outcome = await self.executor.run_execution_and_verification(
                            qra.question, qra.reasoning, qra.answer
                        )
                        if outcome.success:
                            return qra

                        if attempt < self.config.max_fix_attempts:
                            logger.info(
                                f"Refining {prefix} QRA for '{knowledge.title}' (attempt {attempt + 1}/{self.config.max_fix_attempts})..."
                            )
                            fixed_qra = await self.generator.fix_qra(
                                knowledge, qra, outcome.verification_output
                            )
                            if fixed_qra:
                                qra = fixed_qra
                            else:
                                break  # Model failure during fix
                        else:
                            logger.warning(
                                f"Failed to verify {prefix} QRA for '{knowledge.title}' after {self.config.max_fix_attempts} fix attempts."
                            )

                    # If we reach here, it failed. Abandon this slot and return None.
                    return None

        results = await asyncio.gather(*[_gen() for _ in range(count)])
        valid_results = [r for r in results if r is not None]

        if valid_results:
            with open(cache_file, "wb") as f:
                pickle.dump(valid_results, f)

        return valid_results

    def _print_generation_summary(
        self,
        knowledge_list: list[Knowledge],
        prepared_results: list[tuple[list[Any], Any]],
    ):
        print("\n" + "=" * 80)
        print(f"{'Knowledge Title':<50} | {'Target':<6} | {'Success':<7} | {'Status'}")
        print("-" * 80)
        for k, (s_qras, e_item) in zip(knowledge_list, prepared_results):
            # For simplicity, we only track SFT count here as it's the main bulk
            target = self.config.k_sft
            success = len(s_qras)
            if success == target:
                status = "✅ OK"
            elif success > 0:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAILED"

            # Check eval status
            if e_item is None:
                status += " (Eval Failed)"

            title = (k.title[:47] + "...") if len(k.title) > 50 else k.title
            print(f"{title:<50} | {target:<6} | {success:<7} | {status}")
        print("=" * 80 + "\n")

    async def run(self) -> None:
        prepared = await self._prepare_knowledge_tasks()
        sft_batch_iter = self._create_sft_batch_iterator(
            prepared.sft_qras, self.config.sft_epochs
        )

        # Initialize EvaluateWorker
        self.evaluate_worker = EvaluateWorker(
            config=self.config,
            executor=self.executor,
            ml_logger=self.ml_logger,
            prisma_client=self.prisma_client,
            shared_sampling_client=self.shared_sampling_client,
            renderer=self.solver_model.renderer,
            eval_tasks=prepared.eval_tasks,
            extra_eval_suites={
                s.name: [t.instruction for t in s.tasks]
                for s in self.seed_suites
                if s.for_eval
            },
            trigger=self.eval_trigger,
        )
        eval_worker_task = asyncio.create_task(self.evaluate_worker.run_loop())

        if self.config.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Skipping initial SFT because resume_trainer_path is provided: {self.config.model_loading_settings.resume_trainer_path}"
            )
        else:
            logger.info(f"Running initial SFT for {self.config.sft_epochs} epochs...")
            await self._run_sft_steps(sft_batch_iter)

            if self.config.save_sft_checkpoint:
                logger.info("Saving initial SFT checkpoint...")
                await self._save_checkpoint(
                    name="init_sft",
                    loop_state={"epochs": self.config.sft_epochs},
                )

                # Update reference client to the post-SFT state
                self.reference_client = await self.training_client.save_weights_and_get_sampling_client_async()
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

                batch_groups = await self._collect_valid_rl_batch()

                logger.info(
                    f"Running RL (GRPO) update on valid batch of size {len(batch_groups)}..."
                )
                await self._run_grpo_update(batch_groups)

                rl_step = iteration + 1
                if rl_step % self.config.rl_checkpoint_interval == 0:
                    logger.info(f"Saving RL checkpoint at step {rl_step}...")
                    await self._save_checkpoint(
                        name=f"rl_{rl_step:04d}",
                        loop_state={"rl_step": rl_step},
                    )

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
        for task, source in rl_tasks:
            await self.rl_tasks_queue.put((task, source))

        for suite in self.seed_suites:
            if not suite.for_rl:
                continue
            source = RLSource(id=suite.name, title=suite.name)
            for task in suite.tasks:
                await self.rl_tasks_queue.put((task, source))

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
                task, source = await self.rl_tasks_queue.get()
                # Get current shared client snapshot
                indexed_client = self.shared_sampling_client.get_client()

                group = await self._collect_rl_group(
                    system_prompt, task, source, indexed_client
                )
                if group:
                    await self.rl_results_queue.put(group)

                self.rl_tasks_queue.task_done()
                await self.rl_tasks_queue.put((task, source))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"RL worker error: {e}")
                await asyncio.sleep(1)

    async def _collect_rl_group(
        self, system_prompt: str, task: Task, source: RLSource, indexed_client: Any
    ) -> RLGroup | None:
        model_input = self.solver_model.renderer.build_generation_prompt(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=task.instruction),
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
                    task.instruction, reasoning, answer_text
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
                        "knowledge_id": source.id,
                        "knowledge_title": source.title,
                        "step": indexed_client.version,
                        "question": task.instruction,
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

    async def _collect_valid_rl_batch(self) -> list[RLGroup]:
        state = self._rl_batch_state
        logger.info(
            f"Collecting until {state.target_valid} valid RL groups are ready "
            f"(buffered={len(state.valid_buffer)}, window={len(state.window_groups)}/{state.metrics_window})..."
        )
        while not state.ready():
            group = await self.rl_results_queue.get()
            rollout_metrics = state.add(group)
            if rollout_metrics is not None:
                self.ml_logger.log_metrics(rollout_metrics)
        return state.pop_batch()

    async def _run_grpo_update(self, valid_groups: list[RLGroup]) -> None:
        if not valid_groups:
            return

        optimizer_params = OptimizerParams(
            adam_params=self.config.rl_adam_params,
            loss_fn=self.config.rl_loss_fn,
            num_steps=1,
            kl_penalty_coef=self.config.kl_penalty_coef,
            kl_discount_factor=self.config.kl_discount_factor,
        )

        res = await compute_grpo_loss(
            valid_groups,
            self.training_client,
            optimizer_params,
            kl_reference_client=self.reference_client,
        )

        if not res:
            return

        opt_future = await self.training_client.optim_step_async(
            self.config.rl_adam_params
        )
        await opt_future.result_async()

        metrics = {f"rl/{k}": v for k, v in res.metrics.items()}
        metrics["rl/train_batch_size"] = float(len(valid_groups))
        self.ml_logger.log_metrics(metrics)

        logger.info("Synchronizing sampling weights and triggering evaluation...")
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.shared_sampling_client.update_client(new_client)
        self.eval_trigger.set()

    async def _save_checkpoint(self, name: str, loop_state: dict[str, Any]) -> None:
        await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=name,
            log_path=str(self.log_dir),
            loop_state=loop_state,
            kind="both",
            ttl_seconds=self.config.ttl_seconds,
        )

    async def _run_sft_steps(self, batch_iter: Iterable[list[Datum]]) -> None:
        adam_params = self.config.adam_params
        for batch in batch_iter:
            fwd_future = await self.training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
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
            s_qras = await self._generate_and_cache(
                k, self.config.k_sft, "sft", verify=False, is_coding=True
            )
            e_qras = await self._generate_and_cache(
                k, self.config.k_eval, "eval", verify=False, is_coding=True
            )
            # If eval fails, we might not have e_qras[0]. Handle it.
            e_item = (k, e_qras[0]) if e_qras else None
            return s_qras, e_item

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.knowledge_list]
        )

        self._print_generation_summary(self.knowledge_list, results)

        # Save SFT QRAs to database for visualization
        for k, (s_qras, _) in zip(self.knowledge_list, results):
            for qra in s_qras:
                try:
                    # Check if already exists to avoid duplication on re-runs
                    existing = await self.prisma_client.simplesftqna.find_first(
                        where={
                            "simple_train_id": self.config.simple_train_id,
                            "knowledge_id": k.id,
                            "question": qra.question,
                        }
                    )
                    if not existing:
                        await self.prisma_client.simplesftqna.create(
                            data={
                                "simple_train_id": self.config.simple_train_id,
                                "knowledge_id": k.id,
                                "knowledge_title": k.title,
                                "question": qra.question,
                                "reasoning": qra.reasoning,
                                "answer": qra.answer,
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to record SFT QRA to DB: {e}")

        sft_qras = [q for r in results for q in r[0]]
        eval_tasks = [r[1] for r in results if r[1] is not None]
        return PreparedTasks(sft_qras=sft_qras, eval_tasks=eval_tasks)

    async def _prepare_rl_tasks(self) -> list[tuple[Task, RLSource]]:
        async def _prep_single(k: Knowledge):
            qras = await self._generate_and_cache(
                k, self.config.k_rl, "rl", verify=False, is_coding=True
            )
            source = RLSource(id=k.id, title=k.title)
            return [(Task(instruction=q.question), source) for q in qras]

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.knowledge_list]
        )
        return list(itertools.chain.from_iterable(results))

    @staticmethod
    async def _load_solved_seed_tasks(
        client: Prisma, experiment_id: str
    ) -> list[Task]:
        """Fetch root tasks directly under pseudo_root that are marked solved
        in the study experiment's graph."""
        experiment = await client.experiment.find_unique(
            where={"experiment_name": experiment_id}
        )
        if experiment is None or not experiment.graph_json:
            raise ValueError(f"Experiment '{experiment_id}' has no graph_json.")

        graph = experiment.graph_json
        pseudo_root_id = "pseudo_root"
        root_child_ids = {
            e["target"]
            for e in graph.get("edges", [])
            if e.get("type") == "decomposition" and e.get("source") == pseudo_root_id
        }
        tasks: list[Task] = []
        for node in graph.get("nodes", []):
            if node.get("type") != "task" or node["id"] not in root_child_ids:
                continue
            if not node.get("metadata", {}).get("is_solved"):
                continue
            instruction = node["metadata"]["instruction"]
            tasks.append(Task(id=node["id"], instruction=instruction))
        return tasks

    def _create_sft_batch_iterator(
        self, sft_qras: list[QRA], num_epochs: int
    ) -> Iterable[list[Datum]]:
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_loading_settings.model_name
        )
        if self.config.cpt:
            datums = [
                conversation_to_datum(
                    conversation=[
                        Message(
                            role="system",
                            content=f"これは{self.config.library_name}の教育資料から抜粋した練習問題です。",
                            trainable=False,
                        ),
                        Message(
                            role="example_question",
                            content=q.question,
                            trainable=False,
                        ),
                        Message(
                            role="example_answer",
                            content=q.answer,
                            trainable=True,
                        ),
                    ],
                    renderer=renderer,
                    train_on_what=TrainOnWhat.CUSTOMIZED,
                    max_length=None,
                )
                for q in sft_qras
            ]
        else:
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
        if not datums:
            return iter([])

        batch_size = self.config.sft_batch_size
        all_datums = datums * num_epochs
        num_batches = (len(all_datums) + batch_size - 1) // batch_size

        batches = []
        for i in range(num_batches):
            batch = all_datums[i * batch_size : (i + 1) * batch_size]
            if len(batch) < batch_size:
                # Pad with wrap-around from the start of the FULL dataset
                needed = batch_size - len(batch)
                batch.extend(all_datums[:needed])
            batches.append(batch)

        logger.info(f"Created {len(batches)} batches for {num_epochs} epochs.")
        return iter(batches)
