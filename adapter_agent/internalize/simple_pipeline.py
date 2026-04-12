import asyncio
import itertools
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import tinker
from coder_mcp.runtime import Runtime
from more_itertools import chunked
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from prisma import Prisma
from tinker.types.loss_fn_type import LossFnType
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
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import OptimizerParams
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class RuntimeExecutionResult:
    execution_output: str
    tree_output: str
    exit_success: bool


@dataclass
class VerificationOutcome:
    success: bool
    execution_output: str
    verification_output: str


@dataclass
class EvalResult:
    success_count: int
    total_count: int


@dataclass
class PreparedTasks:
    sft_qras: list[QRA]
    eval_tasks: list[tuple[Knowledge, QRA]]


@dataclass
class PipelineConfig:
    runtime_pool_size: int
    rl_worker_count: int
    eval_concurrency: int
    generation_concurrency: int
    simple_train_id: str
    knowledge_list: list[Knowledge]
    model_name: str
    library_name: str
    runtime_settings: RuntimeSettings
    lora_rank: int = 32
    k_sft: int = 32
    k_eval: int = 1
    eval_rollout: int = 4
    init_sft_steps: int = 5
    iter_sft_steps: int = 1
    sft_batch_size: int = 32
    adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-4)
    )
    max_iterations: int = 50
    cache_dir: Path = Path(".cache/simple_internalizer")
    k_rl: int = 4
    rl_rollout: int = 8
    rl_adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-5)
    )
    rl_loss_fn: LossFnType = "importance_sampling"
    rl_batch_size: int = 48
    rl_worker_stagger_s: float = 2.0
    extra_eval_suites: dict[str, list[str]] = field(default_factory=dict)
    stop_grpo: bool = False


class SimplePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        rust_doc_analyzer: AsyncRustDocAnalyzer,
        service_client: tinker.ServiceClient,
        training_client: tinker.TrainingClient,
        solver_model: TinkerModel,
        generator: GeneratorAgent,
        verifier: Verifier,
        ml_logger: MLLogger,
        prisma_client: Prisma,
    ):
        self.config = config
        self.analyzer = rust_doc_analyzer
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_pool = RuntimePool(
            config.runtime_settings, max_size=config.runtime_pool_size
        )

        self.service_client = service_client
        self.training_client = training_client
        self.solver_model = solver_model
        self.generator = generator
        self.verifier = verifier
        self.ml_logger = ml_logger
        self.prisma_client = prisma_client
        self.current_step = 0
        self.rl_tasks_queue: asyncio.Queue[tuple[Knowledge, QRA]] = asyncio.Queue()
        self.rl_results_queue: asyncio.Queue[RLGroup] = asyncio.Queue()

    @classmethod
    async def create(
        cls, config: PipelineConfig, rust_doc_analyzer: AsyncRustDocAnalyzer
    ) -> "SimplePipeline":
        gemini = get_gemini()
        generator = GeneratorAgent(model=gemini, rust_doc_analyzer=rust_doc_analyzer)
        verifier = Verifier(model=gemini, rust_doc_analyzer=rust_doc_analyzer)

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
        tinker_model, _, _ = setup_tinkermodel(
            model_name=config.model_name,
            service_client=service_client,
        )

        log_dir = Path("logs") / "simple_internalizer" / config.model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": config.model_name,
                "library": config.library_name,
                "init_sft_steps": config.init_sft_steps,
                "iter_sft_steps": config.iter_sft_steps,
                "sft_batch_size": config.sft_batch_size,
                "lora_rank": config.lora_rank,
            },
        )

        db_manager = PostgresDB()
        await db_manager.connect()
        client = await db_manager.get_client()
        await client.simpletrainrun.upsert(
            where={"id": config.simple_train_id},
            data={"create": {"id": config.simple_train_id}, "update": {}},
        )

        return cls(
            config=config,
            rust_doc_analyzer=rust_doc_analyzer,
            service_client=service_client,
            training_client=training_client,
            solver_model=tinker_model,
            generator=generator,
            verifier=verifier,
            ml_logger=ml_logger,
            prisma_client=client,
        )

    async def _generate_and_cache(
        self, knowledge: Knowledge, count: int, prefix: str
    ) -> list[QRA]:
        cache_file = self.cache_dir / f"{knowledge.id}_{prefix}_{count}.pkl"
        if cache_file.exists():
            logger.info(
                f"Loading {count} {prefix} QRAs for '{knowledge.title}' from cache."
            )
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        logger.info(f"Generating {count} {prefix} QRAs for '{knowledge.title}'...")
        qras: list[QRA] = []
        sem = asyncio.Semaphore(self.config.generation_concurrency)

        async def _gen() -> QRA:
            async with sem:
                while True:
                    qra = await self.generator.generate_sft(knowledge)
                    if qra:
                        return qra
                    await asyncio.sleep(0.1)  # Avoid tight loop if generator fails

        results = await asyncio.gather(*[_gen() for _ in range(count)])
        qras.extend(results)

        with open(cache_file, "wb") as f:
            pickle.dump(qras, f)

        return qras

    async def run(self) -> None:
        prepared = await self._prepare_knowledge_tasks()
        sft_batch_iter = self._create_sft_batch_iterator(prepared.sft_qras)

        logger.info(f"Running initial {self.config.init_sft_steps} steps of SFT...")
        await self._run_sft_steps(sft_batch_iter, self.config.init_sft_steps)

        spawner_task = None
        worker_tasks = []

        if not self.config.stop_grpo:
            # Transition to RL: Start workers
            spawner_task, worker_tasks = await self._start_rl_workers()

        try:
            for iteration in range(self.config.max_iterations):
                logger.info(
                    f"--- Iteration {iteration + 1} (RL{' STOPPED' if self.config.stop_grpo else ''}) ---"
                )

                # Evaluation (synchronous with start of iteration)
                await asyncio.gather(
                    self._run_evaluation(prepared.eval_tasks),
                    self._run_extra_evaluations(),
                )

                if self.config.stop_grpo:
                    continue

                # Batch Collection: Wait for at least one batch
                batch_groups = await self._collect_rl_batch(self.config.rl_batch_size)

                logger.info("Running RL (GRPO) update on collected batch...")
                await self._run_grpo_update(batch_groups)

                # Backlog processing: consume all remaining full batches
                while self.rl_results_queue.qsize() >= self.config.rl_batch_size:
                    qsize = self.rl_results_queue.qsize()
                    logger.info(
                        f"Backlog detected: {qsize} samples in queue. "
                        f"Draining another batch of {self.config.rl_batch_size}..."
                    )
                    batch_groups = []
                    for _ in range(self.config.rl_batch_size):
                        batch_groups.append(self.rl_results_queue.get_nowait())

                    await self._run_grpo_update(batch_groups)

        finally:
            if spawner_task:
                spawner_task.cancel()
            for t in worker_tasks:
                t.cancel()

            tasks_to_wait = []
            if spawner_task:
                tasks_to_wait.append(spawner_task)
            tasks_to_wait.extend(worker_tasks)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _start_rl_workers(self) -> tuple[asyncio.Task, list[asyncio.Task]]:
        """Prepares RL tasks and starts background workers with staggering."""
        rl_tasks = await self._prepare_rl_tasks()
        for k, qra in rl_tasks:
            await self.rl_tasks_queue.put((k, qra))

        num_workers = self.config.rl_worker_count
        logger.info(
            f"Starting {num_workers} RL background workers with {self.config.rl_worker_stagger_s}s stagger..."
        )
        worker_tasks: list[asyncio.Task] = []

        async def _staggered_spawner():
            for i in range(num_workers):
                worker_tasks.append(asyncio.create_task(self._rl_worker()))
                if i < num_workers - 1:
                    await asyncio.sleep(self.config.rl_worker_stagger_s)

        spawner_task = asyncio.create_task(_staggered_spawner())
        return spawner_task, worker_tasks

    async def _collect_rl_batch(self, batch_size: int) -> list[RLGroup]:
        """Collects a specified number of RL groups from the results queue."""
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
                    {"status/collected_rl_groups": len(batch_groups)},
                    step=self.current_step,
                )
        return batch_groups

    async def _rl_worker(self) -> None:
        """Background worker that continuously performs rollouts."""
        system_prompt = self._get_solver_system_prompt(self.config.library_name)
        while True:
            try:
                k, qra = await self.rl_tasks_queue.get()
                group = await self._collect_rl_group(system_prompt, qra, k)
                if group:
                    await self.rl_results_queue.put(group)

                # Always put task back for revolving queue
                self.rl_tasks_queue.task_done()
                await self.rl_tasks_queue.put((k, qra))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"RL worker encountered error: {e}")
                await asyncio.sleep(1)

    async def _run_grpo_update(self, valid_groups: list[RLGroup]) -> None:
        """Helper to perform the GRPO update and weight sync."""
        optimizer_params = OptimizerParams(
            adam_params=self.config.rl_adam_params,
            loss_fn=self.config.rl_loss_fn,
            num_steps=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
        )

        res = await compute_grpo_loss(
            valid_groups, self.training_client, optimizer_params
        )

        if res:
            opt_future = await self.training_client.optim_step_async(
                self.config.rl_adam_params
            )
            await opt_future.result_async()

            metrics = {f"rl/{k}": v for k, v in res.metrics.items()}
            metrics["rl/valid_groups"] = len(valid_groups)
            self.ml_logger.log_metrics(metrics, step=self.current_step)
            self.current_step += 1

            logger.info("Synchronizing sampling weights...")
            new_client = (
                await self.training_client.save_weights_and_get_sampling_client_async()
            )
            self.solver_model.sampling_client = new_client

    async def _prepare_knowledge_tasks(self) -> PreparedTasks:
        logger.info(
            f"Preparing tasks for {len(self.config.knowledge_list)} knowledge items..."
        )

        async def _prep_single(k: Knowledge):
            s_qras = await self._generate_and_cache(k, self.config.k_sft, "sft")
            e_qras = await self._generate_and_cache(k, self.config.k_eval, "eval")
            return s_qras, (k, e_qras[0])

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.config.knowledge_list]
        )

        sft_qras = []
        eval_qras = []
        for s, e in results:
            sft_qras.extend(s)
            eval_qras.append(e)

        logger.info(
            f"Loaded {len(sft_qras)} SFT tasks and {len(eval_qras)} EVAL tasks."
        )
        return PreparedTasks(sft_qras=sft_qras, eval_tasks=eval_qras)

    async def _prepare_rl_tasks(self) -> list[tuple[Knowledge, QRA]]:
        logger.info(
            f"Preparing RL tasks for {len(self.config.knowledge_list)} knowledge items..."
        )

        async def _prep_single(k: Knowledge):
            tasks = await self._generate_and_cache(k, self.config.k_rl, "rl")
            return [(k, t) for t in tasks]

        results = await asyncio.gather(
            *[_prep_single(k) for k in self.config.knowledge_list]
        )
        rl_tasks = list(itertools.chain.from_iterable(results))

        logger.info(f"Prepared {len(rl_tasks)} RL tasks.")
        return rl_tasks

    async def _collect_rl_group(
        self, system_prompt: str, qra: QRA, knowledge: Knowledge
    ) -> RLGroup | None:
        model_input = self.solver_model.renderer.build_generation_prompt(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=qra.question),
            ]
        )

        sample_results = await self.solver_model.sampling_client.sample_async(
            prompt=model_input,
            num_samples=self.config.rl_rollout,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )

        async def _verify_single_rollout(seq) -> tuple[Trajectory, float]:
            tokens = seq.tokens
            logprobs = seq.logprobs
            ac = TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)
            transition = Transition(
                ob=model_input, ac=ac, reward=0.0, episode_done=True
            )
            trajectory = Trajectory(transitions=[transition], final_ob=model_input)

            msg, ok = self.solver_model.renderer.parse_response(tokens)
            content = msg.get("content") if ok else None
            is_success = False
            exec_out_str = ""
            verif_out_str = ""
            reasoning = ""
            answer_text = ""

            if ok and content:
                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "thinking":
                            reasoning += part["thinking"]
                        elif part["type"] == "text":
                            answer_text += part["text"]
                else:
                    answer_text = str(content)

                outcome = await self._run_execution_and_verification(
                    qra.question, reasoning, answer_text
                )
                is_success, exec_out_str, verif_out_str = (
                    outcome.success,
                    outcome.execution_output,
                    outcome.verification_output,
                )
            else:
                verif_out_str = "Parse failed."

            # Record to Prisma
            try:
                await self.prisma_client.simpletrajectory.create(
                    data={
                        "simple_train_id": self.config.simple_train_id,
                        "knowledge_id": knowledge.id,
                        "knowledge_title": knowledge.title,
                        "step": self.current_step,
                        "question": qra.question,
                        "reasoning": reasoning,
                        "answer": answer_text,
                        "success": is_success,
                        "execution_output": exec_out_str,
                        "verification_output": verif_out_str,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record RL trajectory: {e}")

            return trajectory, (1.0 if is_success else 0.0)

        # Parallelize the 8 rollouts' verification
        rollout_results = await asyncio.gather(
            *[_verify_single_rollout(seq) for seq in sample_results.sequences]
        )

        trajectories = [r[0] for r in rollout_results]
        rewards = [r[1] for r in rollout_results]

        if all(r == rewards[0] for r in rewards):
            return None

        return RLGroup(trajectories=trajectories, rewards=rewards)

    def _create_sft_batch_iterator(self, sft_qras: list[QRA]):
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_name
        )

        datums = [
            conversation_to_datum(
                conversation=[
                    Message(role="user", content=qra.question),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingPart(type="thinking", thinking=qra.reasoning),
                            TextPart(type="text", text=f"\\n\\n{qra.answer}"),
                        ],
                    ),
                ],
                renderer=renderer,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_length=None,
            )
            for qra in sft_qras
        ]

        logger.info(
            f"Splitting {len(datums)} datums into batches of {self.config.sft_batch_size}."
        )
        batches = list(chunked(datums, self.config.sft_batch_size))
        if not batches:
            raise ValueError("No batches created. Aborting.")

        return itertools.cycle(batches)

    async def _run_sft_steps(self, batch_iter, num_steps: int) -> None:
        if num_steps == 0:
            return

        adam_params = self.config.adam_params
        fwd_future = await self.training_client.forward_backward_async(
            data=next(batch_iter), loss_fn="cross_entropy"
        )
        opt_future = await self.training_client.optim_step_async(adam_params)

        for i in range(num_steps):
            if i + 1 < num_steps:
                next_fwd = await self.training_client.forward_backward_async(
                    data=next(batch_iter), loss_fn="cross_entropy"
                )
                next_opt = await self.training_client.optim_step_async(adam_params)
            else:
                next_fwd = next_opt = None

            assert fwd_future is not None
            assert opt_future is not None

            fwd_res = await fwd_future.result_async()
            await opt_future.result_async()

            metrics_to_log = {f"sft/{k}": v for k, v in fwd_res.metrics.items()}
            self.ml_logger.log_metrics(metrics_to_log, step=self.current_step)
            self.current_step += 1

            fwd_future = next_fwd
            opt_future = next_opt

        logger.info("Synchronizing sampling client with newly trained weights...")
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.solver_model.sampling_client = new_client

    def _get_solver_system_prompt(self, library_name: str) -> str:
        return f"""<Role>
You are an expert Rust engineer.
Your task is to solve the programming challenge using the `{library_name}` library.
</Role>

<Guidelines>
1. Write high-quality, idiomatic Rust code.
2. Ensure your solution is complete and self-contained.
3. Ensure that your code produces clear output during execution so that its correctness can be easily verified from the execution results.
4. Your response should include a natural language explanation, and the complete code MUST be enclosed in a ```rust ... ``` code block.
</Guidelines>
"""

    async def _run_evaluation(self, eval_data: list[tuple[Knowledge, QRA]]) -> None:
        logger.info(f"Running evaluation on {len(eval_data)} tasks...")
        system_prompt = self._get_solver_system_prompt(self.config.library_name)

        results = await gather_with_semaphore(
            [
                self._evaluate_single_task(system_prompt, qra, k.id, k.title)
                for k, qra in eval_data
            ],
            max_concurrent=self.config.eval_concurrency,
        )

        total_success = sum(r.success_count for r in results)
        total_rollouts = sum(r.total_count for r in results)

        success_ratio = total_success / total_rollouts if total_rollouts > 0 else 0.0
        self.ml_logger.log_metrics(
            {
                "eval/success_ratio": success_ratio,
                "eval/total_success": total_success,
                "eval/total_rollouts": total_rollouts,
            },
            step=self.current_step,
        )

    async def _run_extra_evaluations(self) -> None:
        if not self.config.extra_eval_suites:
            return

        system_prompt = self._get_solver_system_prompt(self.config.library_name)

        all_eval_tasks = []
        for suite_name, instructions in self.config.extra_eval_suites.items():
            for instr in instructions:
                qra = QRA(question=instr, reasoning="", answer="")
                all_eval_tasks.append((suite_name, qra))

        logger.info(
            f"Running extra evaluation for {len(all_eval_tasks)} total tasks across {len(self.config.extra_eval_suites)} suites concurrently..."
        )

        async def _eval_with_suite(s_name: str, q: QRA) -> tuple[str, EvalResult]:
            res = await self._evaluate_single_task(
                system_prompt,
                q,
                knowledge_id=s_name,
                knowledge_title=s_name,
                rollouts=1,
            )
            return s_name, res

        results = await gather_with_semaphore(
            [_eval_with_suite(s_name, q) for s_name, q in all_eval_tasks],
            max_concurrent=self.config.eval_concurrency,
        )

        suite_metrics = {
            s: {"success": 0, "rollouts": 0}
            for s in self.config.extra_eval_suites.keys()
        }
        for s_name, res in results:
            suite_metrics[s_name]["success"] += res.success_count
            suite_metrics[s_name]["rollouts"] += res.total_count

        metrics_to_log = {}
        for s_name, stats in suite_metrics.items():
            success = stats["success"]
            rollouts = stats["rollouts"]
            success_ratio = success / rollouts if rollouts > 0 else 0.0
            metrics_to_log[f"eval_{s_name}/success_ratio"] = success_ratio
            metrics_to_log[f"eval_{s_name}/total_success"] = success
            metrics_to_log[f"eval_{s_name}/total_rollouts"] = rollouts

        if metrics_to_log:
            self.ml_logger.log_metrics(metrics_to_log, step=self.current_step)

    async def _evaluate_single_task(
        self,
        system_prompt: str,
        qra: QRA,
        knowledge_id: str,
        knowledge_title: str,
        rollouts: int | None = None,
    ) -> EvalResult:
        model_input = self.solver_model.renderer.build_generation_prompt(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=qra.question),
            ]
        )

        num_samples = rollouts if rollouts is not None else self.config.eval_rollout
        sample_results = await self.solver_model.sampling_client.sample_async(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )

        success_count = 0
        for seq in sample_results.sequences:
            msg, ok = self.solver_model.renderer.parse_response(seq.tokens)
            content = msg.get("content") if ok else None
            if not ok or not content:
                continue

            reasoning = ""
            answer_text = ""

            if isinstance(content, list):
                for part in content:
                    if part["type"] == "thinking":
                        reasoning += part["thinking"]
                    elif part["type"] == "text":
                        answer_text += part["text"]
            else:
                answer_text = str(content)

            outcome = await self._run_execution_and_verification(
                qra.question, reasoning, answer_text
            )
            is_success, exec_out_str, verif_out_str = (
                outcome.success,
                outcome.execution_output,
                outcome.verification_output,
            )
            if is_success:
                success_count += 1

            try:
                await self.prisma_client.simpletrajectory.create(
                    data={
                        "simple_train_id": self.config.simple_train_id,
                        "knowledge_id": knowledge_id,
                        "knowledge_title": knowledge_title,
                        "step": self.current_step,
                        "question": qra.question,
                        "reasoning": reasoning,
                        "answer": answer_text,
                        "success": is_success,
                        "execution_output": exec_out_str,
                        "verification_output": verif_out_str,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record trajectory: {e}")

        return EvalResult(success_count=success_count, total_count=num_samples)

    async def _run_rust_code_in_runtime(
        self, runtime: Runtime, code: str
    ) -> RuntimeExecutionResult:
        await runtime.set_content("src/main.rs", code)
        execution_output, exit_success = await runtime.run_cargo()
        tree_output = await runtime.tree()
        return RuntimeExecutionResult(
            execution_output=execution_output,
            tree_output=tree_output,
            exit_success=exit_success,
        )

    async def _run_execution_and_verification(
        self, question: str, reasoning: str, answer_text: str
    ) -> VerificationOutcome:
        code = extract_rust_code(answer_text)
        if not code:
            return VerificationOutcome(
                success=False,
                execution_output="",
                verification_output="No Rust code block found.",
            )

        try:

            async def _run_closure(runtime: Runtime) -> RuntimeExecutionResult:
                return await self._run_rust_code_in_runtime(runtime, code)

            exec_res = await self.runtime_pool.execute_with_retry(_run_closure)

            if not exec_res.exit_success:
                return VerificationOutcome(
                    success=False,
                    execution_output=exec_res.execution_output,
                    verification_output="Compilation or execution failed.",
                )

            ans = QRA(question=question, reasoning=reasoning, answer=answer_text)
            res = await self.verifier.verify(
                qa=ans,
                tree_structure=exec_res.tree_output,
                execution_output=exec_res.execution_output,
                main_rs_content=answer_text,
            )
            return VerificationOutcome(
                success=res.success,
                execution_output=exec_res.execution_output,
                verification_output=res.reasoning,
            )
        except Exception as e:
            logger.error(f"Execution/verification failed: {e}")
            return VerificationOutcome(
                success=False, execution_output="", verification_output=str(e)
            )
