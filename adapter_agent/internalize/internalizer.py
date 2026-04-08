import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Self, cast

import ray
import tinker
from agents.extensions.models.litellm_model import LitellmModel
from more_itertools import chunked
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from ray.actor import ActorHandle
from tinker import SamplingClient
from tinker_cookbook.renderers import (
    Message,
    Renderer,
    TextPart,
    ThinkingPart,
    TrainOnWhat,
)
from tinker_cookbook.rl import Trajectory
from tinker_cookbook.rl.types import TokensWithLogprobs, Transition
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import VerificationResult, Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.grpo import compute_grpo_loss
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge, Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    ModelLoadingSettings,
    OptimizerParams,
    SFTOptimizerParams,
)
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)


class InternalizationTask(Task):
    knowledge_id: str


@dataclass
class SingleRolloutResult:
    question: str
    reasoning: str
    answer: str
    execution_output: str
    main_rs_content: str
    success: bool
    verification_reasoning: str
    trajectory: Optional[Trajectory] = None

    @classmethod
    def parse_failed(cls, question: str, trajectory: Trajectory) -> Self:
        return cls(
            question=question,
            reasoning="Parse Error",
            answer="N/A",
            execution_output="The model output could not be parsed as a valid message.",
            main_rs_content="",
            success=False,
            verification_reasoning="Parse failure.",
            trajectory=trajectory,
        )


@dataclass
class GroupRolloutResult:
    trajectories: list[SingleRolloutResult]
    knowledge_id: str
    current_sampling_version: int = 0

    def to_rlgroup(self) -> RLGroup:
        """Transform this rollout group into an RLGroup for GRPO training."""
        trajectories = [res.trajectory for res in self.trajectories]
        # Binary rewards based on success
        rewards = [1.0 if res.success else 0.0 for res in self.trajectories]
        return RLGroup(trajectories=trajectories, rewards=rewards)


@dataclass
class GlobalState:
    sampling_client: SharedSamplingClient
    # Knowledge items being tracked
    knowledges: list[Knowledge] = field(default_factory=list)

    # QRAGenerator state: knowledge_id -> target count and current pool
    qra_targets: dict[str, int] = field(default_factory=dict)
    qra_pool: dict[str, list[QRA]] = field(default_factory=dict)

    # RemoteWorker state: Tasks and Results
    task_queue: list[InternalizationTask] = field(default_factory=list)
    result_queue: list[GroupRolloutResult] = field(default_factory=list)

    # --- Orchestration State ---
    initialized_knowledge: set[str] = field(default_factory=set)
    failed_knowledge: set[str] = field(default_factory=set)
    # knowledge_id -> latest success ratio (RL trajectories since last SFT)
    performance_map: dict[str, float] = field(default_factory=dict)
    # knowledge_id -> sampling client version at the time of the last SFT update
    last_sft_version: dict[str, int] = field(default_factory=dict)
    # knowledge_id -> list of results accumulated since the last SFT update
    trajectory_buffer: dict[str, list[SingleRolloutResult]] = field(
        default_factory=dict
    )

    # Configs (passed by Internalizer on start)
    sft_threshold: float = 0.5
    k_rl: int = 16
    over_gen_factor: float = 1.5

    async def update_sampling_client(self, sampling_client: SamplingClient) -> None:
        await self.sampling_client.update_client(sampling_client)

    async def get_sampling_client(self) -> IndexedSamplingClient:
        return await self.sampling_client.get_client()

    async def get_active_knowledges(self) -> list[Knowledge]:
        return self.knowledges

    # --- QRA Generation Interfaces ---
    async def get_pool_status(self) -> dict[str, int]:
        """Return the current size of QRA pools per knowledge ID."""
        return {kid: len(pool) for kid, pool in self.qra_pool.items()}

    async def get_target_pool_size(self) -> int:
        """Return the target number of QRAs per knowledge pool."""
        return int(self.k_rl * self.over_gen_factor)

    async def push_qras(self, knowledge_id: str, qras: list[QRA]) -> None:
        """Replenish a specific knowledge pool."""
        if knowledge_id not in self.qra_pool:
            self.qra_pool[knowledge_id] = []
        self.qra_pool[knowledge_id].extend(qras)
        logger.info(f"Pool for '{knowledge_id}' updated: {len(self.qra_pool[knowledge_id])} QRAs ready.")

    async def pop_qras(self, knowledge_id: str, count: int) -> list[QRA]:
        """Get QRAs for training. Called by Internalizer."""
        if knowledge_id not in self.qra_pool:
            return []

        batch = self.qra_pool[knowledge_id][:count]
        self.qra_pool[knowledge_id] = self.qra_pool[knowledge_id][count:]
        return batch

    # --- Rollout Interfaces ---
    async def push_tasks(self, tasks: list[InternalizationTask]) -> None:
        """Add new tasks to the queue."""
        self.task_queue.extend(tasks)

    async def pop_task(self) -> InternalizationTask | None:
        """Get a task from the queue. Called by RemoteWorkers."""
        if not self.task_queue:
            return None
        return self.task_queue.pop(0)

    async def push_result(self, result: GroupRolloutResult) -> None:
        """Add a completed rollout result and update performance metrics."""
        self.result_queue.append(result)

        # Update orchestration state
        k_id = result.knowledge_id
        if k_id not in self.trajectory_buffer:
            self.trajectory_buffer[k_id] = []

        # Only buffer if it matches or is newer than the version after the last SFT
        # results from older versions are discarded for success ratio calculation
        if result.current_sampling_version >= self.last_sft_version.get(k_id, 0):
            self.trajectory_buffer[k_id].extend(result.trajectories)
            # Recalculate performance_map for this knowledge
            buffer = self.trajectory_buffer[k_id]
            if buffer:
                success_count = sum(1 for t in buffer if t.success)
                self.performance_map[k_id] = success_count / len(buffer)

    async def pop_all_results(self) -> list[GroupRolloutResult]:
        """Get all completed results."""
        results = list(self.result_queue)
        self.result_queue.clear()
        return results

    async def pop_rl_batch(
        self, min_batch_size: int
    ) -> list[GroupRolloutResult] | None:
        """Returns results only if the queue size exceeds min_batch_size."""
        if len(self.result_queue) < min_batch_size:
            return None

        # For RL, we usually consume all available results as a single update to stay on-policy
        return await self.pop_all_results()

    async def pop_sft_batch(self, min_batch_size: int) -> list[QRA] | None:
        """
        Returns a combined batch of QRAs if:
        1. Total available QRAs meets min_batch_size.
        2. At least one underperforming knowledge (success < threshold) has enough data.
        Returns None if conditions aren't met, ensuring no QRAs are popped prematurely.
        """
        total_available = sum(len(pool) for pool in self.qra_pool.values())
        if total_available < min_batch_size:
            return None

        underperforming = [
            k_id
            for k_id, success in self.performance_map.items()
            if success < self.sft_threshold
        ]

        if not underperforming:
            return None

        # At this point, we are guaranteed that total_available >= min_batch_size,
        # so we can safely pop exactly one batch without losing data.
        batch: list[QRA] = []

        # Phase 1: Pull from underperforming items with diversity
        await self._fill_diverse(batch, underperforming, min_batch_size)

        # Phase 2: If still needed, fill the rest from all available pools with diversity
        if len(batch) < min_batch_size:
            all_pool_ids = [
                k_id for k_id, pool in self.qra_pool.items() if len(pool) > 0
            ]
            await self._fill_diverse(batch, all_pool_ids, min_batch_size)

        return batch

    async def _fill_diverse(
        self, batch: list[QRA], target_ids: list[str], min_batch_size: int
    ) -> None:
        """Pulls one QRA from each target knowledge ID until full or exhausted."""
        active_ids = [tid for tid in target_ids if len(self.qra_pool.get(tid, [])) > 0]
        while len(batch) < min_batch_size and active_ids:
            for k_id in list(active_ids):
                if len(batch) >= min_batch_size:
                    break
                pulled = await self.pop_qras(k_id, 1)
                if pulled:
                    batch.extend(pulled)
                if len(self.qra_pool.get(k_id, [])) == 0:
                    active_ids.remove(k_id)

    async def mark_sft_update(self, knowledge_ids: list[str], version: int) -> None:
        """Reset performance tracking for specified knowledges after SFT."""
        for k_id in knowledge_ids:
            self.last_sft_version[k_id] = version
            self.trajectory_buffer[k_id] = []

    async def replenish_tasks(self) -> None:
        """Promote QRAs to task_queue based on performance metrics."""
        for k in self.knowledges:
            success = self.performance_map.get(k.id, 0.0)

            # Count how many rollouts are currently in flight or buffered
            in_flight = sum(
                1
                for r in self.result_queue
                if r.knowledge_id == k.id
                and r.current_sampling_version >= self.last_sft_version.get(k.id, 0)
            )
            buffer_count = len(self.trajectory_buffer.get(k.id, []))
            total_current = in_flight + buffer_count

            # If success is low or we don't have enough recent samples, promote more
            if success < 1.0 and total_current < self.k_rl:
                needed = self.k_rl - total_current
                qras = await self.pop_qras(k.id, needed)
                tasks = [
                    InternalizationTask(instruction=q.question, knowledge_id=k.id)
                    for q in qras
                ]
                self.task_queue.extend(tasks)

    async def get_knowledge_ids(self) -> list[str]:
        return [k.id for k in self.knowledges]


@dataclass
class RemoteWorker:
    global_state: ActorHandle[GlobalState]
    library: Library
    model_name: str
    runtime_settings: RuntimeSettings
    k_rollout: int

    # Non-picklable internal state (initialized in setup)
    verifier_model: Optional[LitellmModel] = field(init=False, default=None)
    rust_doc_analyzer: Optional[AsyncRustDocAnalyzer] = field(init=False, default=None)
    verifier: Optional[Verifier] = field(init=False, default=None)
    _renderer: Optional[Renderer] = field(init=False, default=None)

    async def setup(self) -> None:
        """Initialize non-picklable components."""
        if self.verifier is not None:
            return

        logger.info("Initializing RemoteWorker internal components...")
        self.verifier_model = get_gemini()
        self.rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
            self.library.local_path, skip_init=True
        )
        self.verifier = Verifier(
            model=self.verifier_model, rust_doc_analyzer=self.rust_doc_analyzer
        )

        # Initialize student's renderer (local mode)
        # Using setup_tinkermodel as it handles tokenizer/renderer setup canonically.
        _, _, renderer = setup_tinkermodel(self.model_name)
        self._renderer = renderer

    async def get_renderer(self):
        return self._renderer

    async def run_loop(self) -> None:
        """
        Main loop for the worker: constantly pull tasks from GlobalState and perform rollouts.
        """
        await self.setup()
        while True:
            # 1. Get current sampling client and version
            indexed_client: IndexedSamplingClient = (
                await self.global_state.get_sampling_client.remote()
            )

            # 2. Pop task
            task: InternalizationTask | None = await self.global_state.pop_task.remote()
            if task is None:
                await asyncio.sleep(1)
                continue

            # 3. Perform rollout and push
            result = await self.rollout_task_group(
                task, indexed_client.client, indexed_client.version
            )
            await self.global_state.push_result.remote(result)

    async def rollout_task_group(
        self,
        task: InternalizationTask,
        client: SamplingClient,
        sampling_version: int,
    ) -> GroupRolloutResult:
        """Sample multiple trajectories and verify them."""
        assert self._renderer is not None, (
            "Renderer not initialized. Call setup() first."
        )
 
        # 1. Prepare prompt
        system_prompt = self._get_solver_system_prompt()
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task.instruction),
        ]
 
        # Build real ModelInput (observation)
        ob = self._renderer.build_generation_prompt(messages)
 
        # 2. Sample
        sample_results = await client.sample_async(
            prompt=ob,
            num_samples=self.k_rollout,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )
 
        # 3. Process & Verify in parallel
        coros = []
 
        for seq in sample_results.sequences:
            tokens = seq.tokens
            logprobs = seq.logprobs
            ac = TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs)
            transition = Transition(ob=ob, ac=ac, reward=0.0, episode_done=True)
            trajectory = Trajectory(transitions=[transition], final_ob=ob)
 
            # Parse results correctly using the student's renderer
            msg, success = self._renderer.parse_response(tokens)
 
            if not success:
                coros.append(self._to_async_parse_failed(task.instruction, trajectory))
            else:
                qra = self._parse_qra_from_message(task.instruction, msg)
                coros.append(
                    self._verify_with_execution(task.instruction, qra, trajectory)
                )
 
        results = await asyncio.gather(*coros)
        return GroupRolloutResult(
            trajectories=results,
            knowledge_id=task.knowledge_id,
            current_sampling_version=sampling_version,
        )
 
    async def _to_async_parse_failed(self, question: str, trajectory: Trajectory) -> SingleRolloutResult:
        return SingleRolloutResult.parse_failed(question, trajectory)

    async def _verify_with_execution(
        self, question: str, rollout: QRA, trajectory: Trajectory
    ) -> SingleRolloutResult:
        """Run Rust code and verify."""
        assert self.verifier is not None
        try:
            async with self.runtime_settings.build_runtime() as runtime:
                await runtime.set_content("src/main.rs", rollout.answer)
                execution_output, exit_success = await runtime.run_cargo()
                
                if not exit_success:
                    return SingleRolloutResult(
                        question=question,
                        reasoning=rollout.reasoning,
                        answer=rollout.answer,
                        execution_output=execution_output,
                        main_rs_content=rollout.answer,
                        success=False,
                        verification_reasoning="Compilation or runtime execution failed.",
                        trajectory=trajectory
                    )
                
                tree_output = await runtime.tree()
                
                verification_result = await self.verifier.verify(
                    qa=rollout,
                    tree_structure=tree_output,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer
                )
                
                return SingleRolloutResult(
                    question=question,
                    reasoning=rollout.reasoning,
                    answer=rollout.answer,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer,
                    success=verification_result.success,
                    verification_reasoning=verification_result.reasoning,
                    trajectory=trajectory
                )
        except Exception as e:
            logger.error(f"Rollout verification error: {e}")
            return SingleRolloutResult.parse_failed(question, trajectory)

    def _get_solver_system_prompt(self) -> str:
        return f"""<Role>
You are an expert Rust engineer.
Your task is to solve the programming challenge using the `{self.library.name}` library.
</Role>

<Guidelines>
1. Write high-quality, idiomatic Rust code.
2. Ensure your solution is complete and self-contained.
3. Ensure that your code produces clear output during execution so that its correctness can be easily verified from the execution results.
</Guidelines>
"""

    def _parse_qra_from_message(self, question: str, message: Message) -> QRA:
        content = message["content"]
        reasoning = ""
        answer_text = ""

        if isinstance(content, list):
            for part in content:
                # TypedDict check
                if part.get("type") == "thinking":
                    reasoning += str(part.get("thinking", ""))
                elif part.get("type") == "text":
                    answer_text += str(part.get("text", ""))
        else:
            answer_text = content

        if not reasoning:
            reasoning = "N/A"

        return QRA(
            question=question,
            reasoning=reasoning,
            answer=extract_rust_code(answer_text),
        )


@dataclass
class QRAGenerator:
    global_state: ActorHandle[GlobalState]
    library: Library
    runtime_settings: RuntimeSettings

    # Internal state
    generator: Optional[GeneratorAgent] = field(init=False, default=None)
    verifier: Optional[Verifier] = field(init=False, default=None)
    doc_analyzer: Optional[AsyncRustDocAnalyzer] = field(init=False, default=None)

    async def setup(self) -> None:
        """Initialize generator agent and verifier."""
        logger.info("Initializing QRAGenerator internal components...")
        self.doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
            self.library.local_path, skip_init=True
        )
        self.generator = GeneratorAgent(
            model=get_gemini(), rust_doc_analyzer=self.doc_analyzer
        )
        self.verifier = Verifier(
            model=get_gemini(), rust_doc_analyzer=self.doc_analyzer
        )

    async def run_loop(self) -> None:
        """
        Background loop: replenishment of task pools.
        """
        await self.setup()
        while True:
            await self.replenish_all_pools()
            await asyncio.sleep(5)
 
    async def replenish_all_pools(self) -> None:
        """Analyze pool status and trigger replenishment for all knowledges concurrently."""
        assert self.generator and self.verifier, "QRAGenerator not initialized. Call setup() first."
        
        knowledges = await self.global_state.get_active_knowledges.remote()
        pool_status = await self.global_state.get_pool_status.remote()
        target_size = await self.global_state.get_target_pool_size.remote()
        
        # Parallelize across knowledges with a limit (e.g. 8 knowledges at once)
        await gather_with_semaphore(
            [self._replenish_knowledge_pool(k, pool_status.get(k.id, 0), target_size) for k in knowledges],
            max_concurrent=8
        )
 
    async def _replenish_knowledge_pool(self, k: Knowledge, current_size: int, target_size: int) -> None:
        """Generate and verify QRAs if the pool is below target size."""
        if current_size >= target_size:
            return
 
        batch_size = min(4, target_size - current_size)
        logger.info(f"Replenishing '{k.title}': {current_size}/{target_size}")
 
        valid_qras = await self._generate_verified_batch(k, batch_size)
        if valid_qras:
            await self.global_state.push_qras.remote(k.id, valid_qras)
 
    async def _generate_verified_batch(self, k: Knowledge, batch_size: int) -> list[QRA]:
        """Generate multiple QRAs and verify them in parallel."""
        assert self.generator and self.verifier
        
        # Parallelize generation and verification within the batch
        # We use a smaller concurrency limit here or rely on the outer semaphore
        raw_results = await gather_with_semaphore(
            [self._generate_and_verify_single(k) for _ in range(batch_size)],
            max_concurrent=4
        )
        return [q for q in raw_results if q is not None]

    async def _generate_and_verify_single(self, k: Knowledge) -> Optional[QRA]:
        """Helper to generate and verify one QRA for concurrent execution."""
        assert self.generator and self.verifier
        qra = await self.generator.generate_sft(k)
        if qra:
            verify_res = await self._verify_with_execution(k.title, qra)
            if verify_res.success:
                return qra
        return None

    async def _verify_with_execution(self, question: str, rollout: QRA) -> VerificationResult:
        """Run Rust code and verify for newly generated tasks."""
        assert self.verifier is not None
        try:
            async with self.runtime_settings.build_runtime() as runtime:
                await runtime.set_content("src/main.rs", rollout.answer)
                execution_output, exit_success = await runtime.run_cargo()
                
                if not exit_success:
                    return VerificationResult(success=False, reasoning="Cargo execution failed.")
                
                tree_output = await runtime.tree()
                
                return await self.verifier.verify(
                    qa=rollout,
                    tree_structure=tree_output,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer
                )
        except Exception as e:
            logger.error(f"Execution verification failed during QRA generation: {e}")
            return VerificationResult(success=False, reasoning=str(e))


@dataclass
class Internalizer:
    global_state: ActorHandle[GlobalState]
    workers: list[ActorHandle[RemoteWorker]]
    qra_generator: ActorHandle[QRAGenerator]

    # Training Configuration
    model_loading_settings: ModelLoadingSettings
    sft_optimizer_params: SFTOptimizerParams
    rl_optimizer_params: OptimizerParams
    min_sft_batch_size: int
    min_rl_batch_size: int

    max_iterations: int = 100

    # Training state
    training_client: tinker.TrainingClient | None = field(default=None, init=False)
    ml_logger: Optional[MLLogger] = None

    @classmethod
    async def start(
        cls,
        knowledges: list[Knowledge],
        sampling_client: SharedSamplingClient,
        library: Library,
        model_name: str,
        runtime_settings: RuntimeSettings,
        sft_optimizer_params: SFTOptimizerParams,
        rl_optimizer_params: OptimizerParams,
        min_sft_batch_size: int,
        min_rl_batch_size: int,
        k_rollout: int = 16,
        sft_threshold: float = 0.5,
        k_rl: int = 16,
        num_workers: int = 4,
        max_iterations: int = 100,
    ) -> "Internalizer":
        """
        Initialize Ray actors and return a started Internalizer.
        """
        # 1. Start GlobalState actor
        GlobalStateActor = ray.remote(GlobalState)
        global_state = cast(
            ActorHandle[GlobalState],
            GlobalStateActor.remote(
                sampling_client,
                knowledges,
                sft_threshold=sft_threshold,
                k_rl=k_rl,
            ),
        )

        # 2. Start QRAGenerator actor
        QRAGeneratorActor = ray.remote(QRAGenerator)
        qra_generator = cast(
            ActorHandle[QRAGenerator],
            QRAGeneratorActor.remote(global_state, library, runtime_settings),
        )

        # 3. Start RemoteWorker actors
        RemoteWorkerActor = ray.remote(RemoteWorker)
        workers = [
            cast(
                ActorHandle[RemoteWorker],
                RemoteWorkerActor.remote(
                    global_state,
                    library=library,
                    model_name=model_name,
                    runtime_settings=runtime_settings,
                    k_rollout=k_rollout,
                ),
            )
            for _ in range(num_workers)
        ]

        # 4. Logger (W&B)
        log_dir = Path("logs") / "internalizer" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": model_name,
                "library": library.name,
                "num_workers": num_workers,
                "sft_threshold": sft_threshold,
            },
        )

        # 4. Create Internalizer instance
        return cls(
            global_state=global_state,
            workers=workers,
            qra_generator=qra_generator,
            model_loading_settings=ModelLoadingSettings(
                model_name=model_name,
                lora_rank=32,
            ),
            sft_optimizer_params=sft_optimizer_params,
            rl_optimizer_params=rl_optimizer_params,
            min_sft_batch_size=min_sft_batch_size,
            min_rl_batch_size=min_rl_batch_size,
            max_iterations=max_iterations,
            ml_logger=ml_logger,
        )

    async def run(self) -> None:
        """Entry point for the internalizer."""
        await self.initialize()
        await self.main_loop()

    async def initialize(self) -> None:
        """Setup training client and start background actors."""
        logger.info(f"Initializing internalizer (max_iterations={self.max_iterations})")
        await self.setup_training()

        # Start workers and generator in the background
        for worker in self.workers:
            worker.run_loop.remote()
        self.qra_generator.run_loop.remote()

    async def main_loop(self) -> None:
        """
        Main reactive loop for the internalization process.
        Prioritize RL updates for off-policy data, then SFT updates.
        """
        for iteration in range(self.max_iterations):
            logger.info(f"--- Monitoring Turn {iteration + 1} ---")
            
            # Overall Status Logging
            status: dict[str, float] = await self.global_state.get_status_summary.remote()
            if status and self.ml_logger:
                mean_success = sum(status.values()) / len(status)
                self.ml_logger.log_metrics({
                    "iteration": iteration + 1,
                    "overall/mean_task_success": mean_success,
                    "overall/active_knowledges": len(status),
                })

            # 1. RL Update (Priority): Use pop_rl_batch to ensure efficiency
            rollout_results: (
                list[GroupRolloutResult] | None
            ) = await self.global_state.pop_rl_batch.remote(self.min_rl_batch_size)
            if rollout_results:
                await self.rl_step(rollout_results)
                continue

            # 2. SFT Update (Secondary): Use pop_sft_batch to ensure efficiency
            sft_dataset: (
                list[QRA] | None
            ) = await self.global_state.pop_sft_batch.remote(self.min_sft_batch_size)
            if sft_dataset:
                await self.sft_step(sft_dataset)
                continue

            # 3. Idle Phase: Replenish tasks and wait if no action was taken
            await self.replenish_tasks()
            await asyncio.sleep(5)

        logger.info("Internalization process complete.")

    async def replenish_tasks(self) -> None:
        """
        Convert verified QRAs from the pool into active InternalizationTasks.
        """
        pool_status: dict[str, int] = await self.global_state.get_pool_status.remote()
        
        for kid, count in pool_status.items():
            if count > 0:
                # Take some QRAs and convert them
                qras = await self.global_state.pop_qras.remote(kid, count)
                tasks = [
                    InternalizationTask(instruction=q.question, knowledge_id=kid)
                    for q in qras
                ]
                await self.global_state.push_tasks.remote(tasks)
                logger.info(f"Pushed {len(tasks)} new tasks for knowledge '{kid}'.")
            await asyncio.sleep(5)

    async def rl_step(self, results: list[GroupRolloutResult]) -> None:
        """Perform combined GRPO update using collected trajectories."""
        # 1. Flatten results
        rl_groups: list[RLGroup] = [r.to_rlgroup() for r in results]
        
        # 2. Compute aggregate metrics
        total_trajectories = sum(len(g.trajectories) for g in rl_groups)
        total_reward = sum(sum(g.rewards) for g in rl_groups)
        mean_reward = total_reward / total_trajectories if total_trajectories > 0 else 0.0
        
        logger.info(f"RL STEP: results={len(results)}, trajectories={total_trajectories}, mean_reward={mean_reward:.4f}")

        # 3. Update model
        await self.exec_rl(rl_groups)
        
        # 4. Log metrics
        if self.ml_logger:
            self.ml_logger.log_metrics({
                "rl/trajectories": total_trajectories,
                "rl/mean_reward": mean_reward,
                "rl/total_reward": total_reward,
            })

        # replenish_tasks is handled by GlobalState in the background or during monitoring
        await self.global_state.replenish_tasks.remote()

    async def sft_step(self, sft_dataset: list[QRA]) -> None:
        """
        Perform a single SFT step: train on the provided QRA dataset.
        """
        logger.info(f"Triggering SFT update with {len(sft_dataset)} samples.")
        await self.exec_sft(sft_dataset)

        # Inform GlobalState about the SFT update to update version thresholds
        indexed_client: IndexedSamplingClient = (
            await self.global_state.get_sampling_client.remote()
        )

        # Mark updated knowledges to current version
        updated_ids = list(
            set(q.knowledge_id for q in sft_dataset if hasattr(q, "knowledge_id"))
        )
        if updated_ids:
            await self.global_state.mark_sft_update.remote(
                updated_ids,
                indexed_client.version,
            )

    async def setup_training(self) -> None:
        """Initialize Tinker training client."""
        logger.info("Setting up Tinker TrainingClient...")
        service_client = tinker.ServiceClient()
        self.training_client = service_client.create_lora_training_client(
            base_model=self.model_loading_settings.model_name,
            rank=self.model_loading_settings.lora_rank,
        )
        if self.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Loading trainer from {self.model_loading_settings.resume_trainer_path}"
            )
            self.training_client.load_state(
                path=self.model_loading_settings.resume_trainer_path
            )

    async def exec_sft(self, qras: list[QRA]) -> None:
        """Execute SFT update."""
        assert self.training_client is not None
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.model_loading_settings.model_name
        )
        data = [
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
            for q in qras
        ]
        
        logger.info(f"SFT STEP: updating with {len(data)} samples")
        
        # Simple SFT loop (one epoch for now)
        for batch in chunked(data, self.sft_optimizer_params.batch_size):
            fwd_bwd = await self.training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
            )
            optim = await self.training_client.optim_step_async(
                self.sft_optimizer_params.adam_params
            )
            
            res = await fwd_bwd.result_async()
            await optim.result_async()
            
            if self.ml_logger:
                self.ml_logger.log_metrics({
                    "sft/loss": res.loss if hasattr(res, "loss") else 0.0,
                    "sft/batch_samples": len(batch),
                })
        
        await self._sync_sampling_weights()

    async def exec_rl(self, dataset: list[RLGroup]) -> None:
        """Execute RL (GRPO) training step."""
        logger.info(f"Triggering GRPO training for batch of {len(dataset)} groups...")
        assert self.training_client is not None
        grpo_res = await compute_grpo_loss(
            dataset, self.training_client, self.rl_optimizer_params
        )
        logger.info("GRPO loss computed, performing optimizer step...")
        optim_future = await self.training_client.optim_step_async(
            self.rl_optimizer_params.adam_params
        )
        await optim_future.result_async()
        
        if self.ml_logger and hasattr(grpo_res, "metrics"):
            self.ml_logger.log_metrics({
                f"rl/{k}": v for k, v in grpo_res.metrics.items()
            })
            
        await self._sync_sampling_weights()

    async def _sync_sampling_weights(self) -> None:
        """Sync updated weights to GlobalState's SharedSamplingClient."""
        if self.training_client:
            logger.info("Syncing updated weights to sampling model...")
            new_sampling_client = (
                await self.training_client.save_weights_and_get_sampling_client_async()
            )
            await self.global_state.update_sampling_client.remote(new_sampling_client)
