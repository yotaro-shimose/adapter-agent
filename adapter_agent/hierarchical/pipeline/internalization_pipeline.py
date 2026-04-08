import asyncio
import logging
from dataclasses import dataclass
from typing import Self

import tinker
import wandb
from more_itertools import chunked
from oai_utils.agent import AgentsSDKModel
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.solver import SolverAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import compute_grpo_loss
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.rl.config import (
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    SFTOptimizerParams,
)
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import Knowledge
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)
# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class PipelineConfig:
    knowledge_list: list[Knowledge]
    runtime_settings: RuntimeSettings
    model_loading_settings: ModelLoadingSettings
    sft_optimizer_params: SFTOptimizerParams
    rl_optimizer_params: OptimizerParams
    experiment_settings: ExperimentSettings
    sft_threshold: float = 0.5
    max_sft_knowledge: int = 8
    k_sft: int = 16
    k_init_sft: int = 16
    init_sft_epochs: int = 6
    k_rl: int = 16
    k_rollout: int = 16
    over_generation_factor: float = 1.5
    max_generation_attempts: int = 3
    concurrency: int = 8
    task_gen_concurrency: int = 8
    max_iterations: int = 10
    stop_at_100: bool = True


@dataclass
class RLTrajectory:
    question: str
    reasoning: str
    answer: str
    execution_output: str
    verification_reasoning: str
    success: bool

    @classmethod
    def failed(cls, question: str) -> Self:
        return cls(
            question=question,
            reasoning="Parse Error",
            answer="N/A",
            execution_output="The model output could not be parsed as a valid message.",
            verification_reasoning="Parse failure.",
            success=False,
        )


@dataclass
class RolloutMetadata:
    ac: TokensWithLogprobs
    qra: QRA | None


@dataclass
class SFTStepResult:
    representative_sample: RLTrajectory | None
    qra_list: list[QRA]


@dataclass
class TaskRolloutResult:
    num_success: int
    num_rollouts: int
    group: RLGroup | None
    sample: RLTrajectory | None

    @classmethod
    def from_results(
        cls,
        model_input,
        results: list[RLTrajectory],
        metadata: list[RolloutMetadata],
    ) -> Self:
        trajectories = []
        rewards = []
        sample: RLTrajectory | None = None

        for i, res in enumerate(results):
            reward = 1.0 if res.success else 0.0
            rewards.append(reward)
            trajectories.append(
                Trajectory(
                    transitions=[
                        Transition(
                            ob=model_input,
                            ac=metadata[i].ac,
                            reward=reward,
                            episode_done=True,
                        )
                    ],
                    final_ob=tinker.ModelInput.empty(),
                )
            )
            if sample is None:
                sample = res

        group = (
            RLGroup(trajectories=trajectories, rewards=rewards)
            if len(set(rewards)) > 1
            else None
        )
        return cls(
            num_success=sum(1 for res in results if res.success),
            num_rollouts=len(results),
            group=group,
            sample=sample,
        )


@dataclass
class RLStepResult:
    rollout_success_ratio: float
    task_success_ratio: float
    total_success: int
    total_rollouts: int
    task_success: int
    num_tasks: int
    sample: RLTrajectory | None

    @classmethod
    def from_task_rollouts(cls, task_results: list[TaskRolloutResult]) -> Self:
        total_success = sum(tr.num_success for tr in task_results)
        total_rollouts = sum(tr.num_rollouts for tr in task_results)
        task_success = sum(1 for tr in task_results if tr.num_success > 0)
        num_tasks = len(task_results)

        rl_sample = next(
            (tr.sample for tr in task_results if tr.sample and tr.sample.success), None
        )
        if not rl_sample and task_results:
            rl_sample = task_results[0].sample

        result = cls(
            rollout_success_ratio=total_success / total_rollouts
            if total_rollouts > 0
            else 0.0,
            task_success_ratio=task_success / num_tasks if num_tasks > 0 else 0.0,
            total_success=total_success,
            total_rollouts=total_rollouts,
            task_success=task_success,
            num_tasks=num_tasks,
            sample=rl_sample,
        )
        return result


class InternalizationPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        generator_model: AgentsSDKModel,
        verifier_model: AgentsSDKModel,
        rust_doc_analyzer: AsyncRustDocAnalyzer,
        library_name: str,
    ):
        self.config = config
        self.rust_doc_analyzer = rust_doc_analyzer
        self.library_name = library_name

        # Concurrency
        self.task_gen_semaphore = asyncio.Semaphore(self.config.task_gen_concurrency)

        # Agents
        self.generator = GeneratorAgent(
            model=generator_model, rust_doc_analyzer=rust_doc_analyzer
        )
        self.verifier = Verifier(
            model=verifier_model, rust_doc_analyzer=rust_doc_analyzer
        )

        # Training/Solver (Initialized in setup)
        self.training_client: tinker.TrainingClient | None = None
        self.service_client: tinker.ServiceClient | None = None
        self.ml_logger: MLLogger | None = None
        self._samples_table: wandb.Table | None = None

        # Performance Tracking: Maps knowledge title to current task success ratio
        self.performance_map: dict[str, float] = {
            k.id: 0.0 for k in self.config.knowledge_list
        }

        # Knowledge Pools (QRA, RLTrajectory) and Background Generators
        self.knowledge_pools: dict[str, list[tuple[QRA, RLTrajectory]]] = {
            k.id: [] for k in self.config.knowledge_list
        }
        self.generator_tasks: dict[str, asyncio.Task] = {}
        self.failed_knowledge: set[str] = set()
        self.completed_knowledge: set[str] = set()
        self.initialized_knowledge: set[str] = set()

    async def setup(self):
        """
        Initialize Tinker clients and solver agents.
        """
        if self.service_client:
            return

        logger.info("Setting up Tinker ServiceClient and TrainingClient...")
        self.service_client = tinker.ServiceClient()

        # 1. Setup Training Client for student (Qwen-3-8B)
        self.training_client = (
            await self.service_client.create_lora_training_client_async(
                base_model=self.config.model_loading_settings.model_name,
                rank=self.config.model_loading_settings.lora_rank,
            )
        )
        if self.config.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Loading trainer from {self.config.model_loading_settings.resume_trainer_path}..."
            )
            await self.training_client.load_state_async(
                path=self.config.model_loading_settings.resume_trainer_path
            )
        else:
            logger.warning("No trainer path provided, starting from scratch.")

        # 2. Setup Sampling Model (TinkerModel wrapper for agents)
        student_tinker_model, _, _ = setup_tinkermodel(
            model_name=self.config.model_loading_settings.model_name,
            path=self.config.model_loading_settings.resume_sampler_path,
            service_client=self.service_client,
        )

        # 3. Solver Agent (Student)
        self.solver = SolverAgent[TinkerModel](
            model=student_tinker_model, rust_doc_analyzer=self.rust_doc_analyzer
        )

        # 4. Logger (W&B)
        log_root = self.config.experiment_settings.log_root()
        log_root.mkdir(parents=True, exist_ok=True)
        self.ml_logger = ml_log.setup_logging(
            log_dir=str(log_root),
            wandb_project=self.config.experiment_settings.wandb_project,
            config=self.config,
        )

        if self.ml_logger:
            self._samples_table = wandb.Table(
                columns=["Iteration", "Question", "Reasoning", "Code", "Success"]
            )

        logger.info("Pipeline components successfully initialized.")

    async def run(self):
        """
        Main loop: Orchestrate SFT and RL across multiple knowledge items.
        """
        await self.setup()
        logger.info(
            f"Starting API internalization for {len(self.config.knowledge_list)} items."
        )

        # 1. Start background generators for all initial knowledge items
        for k in self.config.knowledge_list:
            if k.id not in self.generator_tasks:
                self.generator_tasks[k.id] = asyncio.create_task(
                    self._run_background_generator(k)
                )

        iteration = 0
        try:
            while iteration < self.config.max_iterations:
                iteration += 1
                logger.info(f"--- Iteration {iteration} ---")

                if not await self._process_iteration(iteration):
                    break

        finally:
            # Cleanup: stop all background generators
            logger.info("Cleaning up background generators...")
            for task in self.generator_tasks.values():
                task.cancel()
            await asyncio.gather(*self.generator_tasks.values(), return_exceptions=True)
            self.generator_tasks.clear()

    async def _process_iteration(self, iteration: int) -> bool:
        """
        Execute a single iteration of SFT and RL.
        Returns True to continue, False to stop.
        """
        # 1. Synchronize: Wait for all active items to have ready pools
        await self._wait_for_task_readiness(self.config.knowledge_list)

        # 2. Filter knowledge: Remove completed or failed items
        active_knowledge = self._filter_active_knowledge()
        self.config.knowledge_list = active_knowledge

        if not self.config.knowledge_list:
            logger.warning("No knowledge items remaining to process.")
            return False

        # 3. Identify knowledge items needing SFT
        init_candidates = [
            k
            for k in self.config.knowledge_list
            if k.id not in self.initialized_knowledge
        ]
        regular_candidates = [
            k
            for k in self.config.knowledge_list
            if k.id in self.initialized_knowledge
            and self.performance_map[k.id] < self.config.sft_threshold
        ]

        # Prioritize init tasks, then regular tasks, up to max_sft_knowledge
        sft_targets_init = init_candidates[: self.config.max_sft_knowledge]
        remaining_slots = self.config.max_sft_knowledge - len(sft_targets_init)
        sft_targets_regular = (
            regular_candidates[:remaining_slots] if remaining_slots > 0 else []
        )

        sft_result: SFTStepResult | None = None

        # Execute Initial SFT
        if sft_targets_init:
            logger.info(
                f"Selected {len(sft_targets_init)} items for INITIAL SFT (k={self.config.k_init_sft}, epochs={self.config.init_sft_epochs})."
            )
            sft_result = await self.execute_sft_step(
                sft_targets_init, self.knowledge_pools, is_init=True
            )
            # Mark as initialized
            for k in sft_targets_init:
                self.initialized_knowledge.add(k.id)

        # Execute Regular SFT
        if sft_targets_regular:
            logger.info(
                f"Selected {len(sft_targets_regular)} items for REGULAR SFT (k={self.config.k_sft})."
            )
            regular_sft_result = await self.execute_sft_step(
                sft_targets_regular, self.knowledge_pools, is_init=False
            )
            if sft_result is None:
                sft_result = regular_sft_result
            else:
                sft_result.qra_list.extend(regular_sft_result.qra_list)

        # 4. RL Phase: Exploration and Correction for ALL knowledge items in PARALLEL
        logger.info(
            f"Running RL Phase for {len(self.config.knowledge_list)} items concurrently."
        )

        # Collect rollouts from all knowledge items
        all_task_results_per_knowledge: list[
            list[TaskRolloutResult]
        ] = await asyncio.gather(
            *[
                self._collect_rl_rollouts(k, self.knowledge_pools[k.id])
                for k in self.config.knowledge_list
            ]
        )

        # Consume RL tasks from pools
        for k in self.config.knowledge_list:
            self.knowledge_pools[k.id] = self.knowledge_pools[k.id][
                self.config.k_rl :
            ]

        # Flatten all results for a single combined GRPO update
        all_results_flat = [
            tr for k_results in all_task_results_per_knowledge for tr in k_results
        ]

        if self._should_perform_grpo(all_results_flat):
            await self._perform_grpo_update(all_results_flat)

        # Sync sampling weights once per iteration after the aggregated update
        await self._sync_sampling_weights()

        # Process metrics and summarized results per knowledge
        iteration_metrics = {}
        total_rl_success = 0
        total_rl_rollouts = 0
        total_rl_task_success = 0
        total_rl_tasks = 0

        for i, (knowledge, k_results) in enumerate(
            zip(self.config.knowledge_list, all_task_results_per_knowledge)
        ):
            # Summarize results for this specific knowledge item
            rl_result = self._summarize_rl_results(k_results)

            # Update performance map with the latest task success ratio
            self.performance_map[knowledge.id] = rl_result.task_success_ratio

            # Accumulate per-knowledge metrics
            iteration_metrics[f"rl/{knowledge.title}/rollout_success_ratio"] = (
                rl_result.rollout_success_ratio
            )
            iteration_metrics[f"rl/{knowledge.title}/task_success_ratio"] = (
                rl_result.task_success_ratio
            )

            # Accumulate for aggregate metrics
            total_rl_success += rl_result.total_success
            total_rl_rollouts += rl_result.total_rollouts
            total_rl_task_success += rl_result.task_success
            total_rl_tasks += rl_result.num_tasks

            # self._update_wandb_table(iteration, rl_result.sample)

            # Visualization / Logging (for terminal feedback)
            # self._visualize_iteration(
            #     iteration,
            #     sft_result.representative_sample if sft_result else None,
            #     rl_result.sample,
            #     include_header=(i == 0),
            #     include_sft=(i == 0 and sft_result is not None),
            # )

        # 5. Overall Progress
        if total_rl_rollouts > 0:
            iteration_metrics["rl/rollout_success_ratio"] = (
                total_rl_success / total_rl_rollouts
            )
        if total_rl_tasks > 0:
            iteration_metrics["rl/task_success_ratio"] = (
                total_rl_task_success / total_rl_tasks
            )

        mean_task_success = sum(self.performance_map.values()) / len(
            self.performance_map
        )
        iteration_metrics["rl/mean_task_success"] = mean_task_success

        # Single point for logging all RL-related metrics in this iteration
        if self.ml_logger:
            self.ml_logger.log_metrics(iteration_metrics)

        logger.info(
            f"Iteration {iteration} complete. Mean Task Success: {mean_task_success:.2%}"
        )

        if not self.config.stop_at_100:
            return True

        return mean_task_success < 1.0

    def _filter_active_knowledge(self) -> list[Knowledge]:
        """
        Filter knowledge items, removing completed or failed ones.
        """
        active_knowledge = []
        for k in self.config.knowledge_list:
            if self.config.stop_at_100 and self.performance_map.get(k.id, 0.0) >= 1.0:
                logger.info(f"KNOWLEDGE COMPLETED: '{k.title}' reached 100% success.")
                self.completed_knowledge.add(k.id)
                # Stop generator
                if k.id in self.generator_tasks:
                    self.generator_tasks[k.id].cancel()
            elif k.id in self.failed_knowledge:
                logger.error(
                    f"KNOWLEDGE REJECTED: '{k.title}' failed to generate enough valid tasks."
                )
            else:
                active_knowledge.append(k)
        return active_knowledge

    async def _run_background_generator(self, knowledge: Knowledge):
        """
        Background loop to continuously replenish the task pool for a specific knowledge item.
        Exits only when the knowledge is no longer being actively processed.
        """
        knowledge_id = knowledge.id
        logger.info(f"Starting background task generator for '{knowledge.title}'.")
        consecutive_failures = 0

        try:
            while True:
                # 1. Determine target pool size dynamically
                required_count, target_count = self._get_required_task_counts(knowledge_id)

                current_pool = self.knowledge_pools.get(knowledge_id, [])
                if len(current_pool) < target_count:
                    # 2. Generate a small batch to replenish the pool
                    batch_size = max(4, target_count - len(current_pool))
                    # Cap batch size to avoid overwhelming with a single item
                    batch_size = min(batch_size, 8)

                    logger.debug(
                        f"Replenishing '{knowledge.title}': current={len(current_pool)}, target={target_count}, batch={batch_size}"
                    )

                    tasks_raw_maybe = await asyncio.gather(
                        *[
                            self._generate_qra_with_semaphore(knowledge)
                            for _ in range(batch_size)
                        ]
                    )
                    tasks_raw = [t for t in tasks_raw_maybe if t is not None]

                    if not tasks_raw:
                        consecutive_failures += 1
                    else:
                        # 3. Verify the generated batch
                        verification_results = await gather_with_semaphore(
                            [
                                self._verify_with_execution(q.question, q)
                                for q in tasks_raw
                            ],
                            max_concurrent=self.config.concurrency,
                        )

                        valid_tasks = [
                            (tasks_raw[i], res)
                            for i, res in enumerate(verification_results)
                            if res.success
                        ]

                        if valid_tasks:
                            self.knowledge_pools[knowledge_id].extend(valid_tasks)
                            logger.info(
                                f"Pool for '{knowledge.title}' updated: {len(self.knowledge_pools[knowledge_id])} tasks available (Required: {required_count}, Buffer Target: {target_count})."
                            )
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1

                    if consecutive_failures >= self.config.max_generation_attempts:
                        logger.error(
                            f"Background generator for '{knowledge.title}' failed repeatedly. Flagging for rejection."
                        )
                        self.failed_knowledge.add(knowledge_id)
                        break
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.info(f"Background generator for '{knowledge.title}' cancelled.")
        except Exception as e:
            logger.exception(
                f"Fatal error in background generator for '{knowledge.title}': {e}"
            )
        finally:
            # Cleanup: ensure pool is cleared or marked if knowledge is removed
            pass

    async def _wait_for_task_readiness(self, active_knowledge: list[Knowledge]):
        """
        Wait until all active knowledge items have enough tasks for the current iteration.
        Returns early if any active knowledge item has failed repeatedly.
        """
        logger.info(
            "Waiting for all active knowledge items to have ready task pools..."
        )
        while True:
            all_ready = True
            for k in active_knowledge:
                # 1. Safety check: Did this knowledge item recently fail to generate tasks?
                if k.id in self.failed_knowledge:
                    logger.warning(
                        f"Wait interrupted for '{k.title}' as it was marked as failed."
                    )
                    return

                # 2. Check current pool size
                needs_sft = (
                    self.performance_map.get(k.id, 0.0) < self.config.sft_threshold
                )
                required = (self.config.k_sft if needs_sft else 0) + self.config.k_rl

                if len(self.knowledge_pools.get(k.id, [])) < required:
                    logger.debug(
                        f"Waiting for '{k.title}': {len(self.knowledge_pools[k.id])}/{required} tasks."
                    )
                    all_ready = False
                    break

            if all_ready:
                logger.info("All task pools are ready for the current iteration.")
                break

            await asyncio.sleep(2)

    async def _generate_qra_with_semaphore(self, knowledge: Knowledge) -> QRA | None:
        """
        Generate a QRA triplet for SFT bootstrapping, limited by task_gen_semaphore.
        """
        async with self.task_gen_semaphore:
            return await self.generator.generate_sft(knowledge)

    async def execute_sft_step(
        self,
        targets: list[Knowledge],
        pools: dict[str, list[tuple[QRA, RLTrajectory]]],
        is_init: bool = False,
    ) -> SFTStepResult:
        """
        Perform supervised fine-tuning across multiple knowledge targets.
        """
        logger.info(f"Executing SFT Step for {len(targets)} targets...")
        assert self.training_client is not None, "TrainingClient must be initialized"

        combined_qra_list = []
        representative_sample: RLTrajectory | None = None

        k_val = self.config.k_init_sft if is_init else self.config.k_sft
        epochs = self.config.init_sft_epochs if is_init else None

        for k in targets:
            pool = pools[k.id]
            # Take tasks and consume them
            sft_bundle = pool[:k_val]
            pools[k.id] = pool[k_val:]

            sft_tasks = [bundle[0] for bundle in sft_bundle]
            combined_qra_list.extend(sft_tasks)

            # Pick a representative sample for visualization (use existing trajectory)
            if not representative_sample and sft_bundle:
                representative_sample = sft_bundle[0][1]

        if combined_qra_list:
            await self._perform_sft_training(combined_qra_list, num_epochs=epochs)

        return SFTStepResult(
            representative_sample=representative_sample, qra_list=combined_qra_list
        )

    async def _perform_sft_training(self, qra_list: list[QRA], num_epochs: int | None = None):
        logger.info(
            f"Fine-tuning student on {len(qra_list)} total VALID QRA samples..."
        )

        assert self.training_client is not None
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_loading_settings.model_name
        )

        datums = [
            conversation_to_datum(
                conversation=[
                    Message(role="user", content=qra.question),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingPart(type="thinking", thinking=qra.reasoning),
                            TextPart(type="text", text=f"\n\n{qra.answer}"),
                        ],
                    ),
                ],
                renderer=renderer,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_length=None,
            )
            for qra in qra_list
        ]

        await self._run_sft_training_loop(datums, num_epochs=num_epochs)

    async def _run_sft_training_loop(self, datums: list, num_epochs: int | None = None):
        params = self.config.sft_optimizer_params
        epochs = num_epochs if num_epochs is not None else params.num_epochs
        batches = list(chunked(datums, params.batch_size))

        assert self.training_client is not None
        logger.info("Initializing SFT training loop with look-ahead...")
        fwd_bwd_future = await self.training_client.forward_backward_async(
            data=batches[0], loss_fn="cross_entropy"
        )
        optim_future = await self.training_client.optim_step_async(params.adam_params)

        for epoch in range(epochs):
            for step, _ in enumerate(batches):
                # Enqueue next
                next_ep, next_st = self._get_next_batch_idx(
                    epoch, step, len(batches), params.num_epochs
                )
                if next_ep is not None and next_st is not None:
                    next_fwd_bwd_future = (
                        await self.training_client.forward_backward_async(
                            data=batches[next_st], loss_fn="cross_entropy"
                        )
                    )
                    next_optim_future = await self.training_client.optim_step_async(
                        params.adam_params
                    )
                else:
                    next_fwd_bwd_future = next_optim_future = None

                # Await current
                if fwd_bwd_future is not None:
                    fwd_bwd_result = await fwd_bwd_future.result_async()
                    assert optim_future is not None
                    await optim_future.result_async()
                    self._log_sft_metrics(
                        epoch, step, len(batches), fwd_bwd_result, params.num_epochs
                    )

                fwd_bwd_future, optim_future = next_fwd_bwd_future, next_optim_future

        logger.info(f"SFT Step complete after {params.num_epochs} epochs.")

    def _get_next_batch_idx(
        self, epoch: int, step: int, num_batches: int, num_epochs: int
    ) -> tuple[int | None, int | None]:
        if step + 1 < num_batches:
            return epoch, step + 1
        elif epoch + 1 < num_epochs:
            return epoch + 1, 0
        return None, None

    def _log_sft_metrics(
        self, epoch: int, step: int, num_batches: int, result, total_epochs: int
    ):
        self._log_metrics(f"sft/epoch_{epoch + 1}/step_{step + 1}", result.metrics)
        self._log_metrics("sft", result.metrics)

        loss = result.metrics.get("loss")
        if loss is None:
            loss = next(
                (v for k, v in result.metrics.items() if k.startswith("loss")), "N/A"
            )

        if step % 5 == 0:
            logger.info(
                f"SFT Epoch {epoch + 1}/{total_epochs}, Step {step + 1}/{num_batches}: loss={loss}"
            )

    async def _collect_rl_rollouts(
        self, knowledge: Knowledge, pool: list[tuple[QRA, RLTrajectory]]
    ) -> list[TaskRolloutResult]:
        """
        Rollout solutions for pool tasks and verify them.
        Returns a list of TaskRolloutResults for potential training.
        """
        logger.info(f"Collecting RL rollouts for '{knowledge.title}'...")

        # Use the remaining tasks in the pool for RL (capped at k_rl)
        # We only need the QA part for rollouts
        tasks = [bundle[0] for bundle in pool[: self.config.k_rl]]

        task_results = await gather_with_semaphore(
            [self._rollout_and_verify_task_group(t) for t in tasks],
            max_concurrent=self.config.concurrency,
        )

        return task_results

    async def _rollout_and_verify_task_group(self, qa: QA) -> TaskRolloutResult:
        assert isinstance(self.solver.model, TinkerModel)
        library_name = self.library_name
        system_prompt = self._get_solver_system_prompt(library_name)

        model_input = self.solver.model.renderer.build_generation_prompt(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=qa.question),
            ]
        )

        sample_results = await self.solver.model.sampling_client.sample_async(
            prompt=model_input,
            num_samples=self.config.k_rollout,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )

        verification_tasks = []
        metadata_list: list[RolloutMetadata] = []

        for seq in sample_results.sequences:
            ac = TokensWithLogprobs(tokens=seq.tokens, maybe_logprobs=seq.logprobs)
            msg, success = self.solver.model.renderer.parse_response(ac.tokens)

            if not success:
                verification_tasks.append(self._wrap_failed_trajectory(qa.question))
                metadata_list.append(RolloutMetadata(ac=ac, qra=None))
            else:
                qra = self._parse_qra_from_message(qa.question, msg)
                verification_tasks.append(self._verify_with_execution(qa.question, qra))
                metadata_list.append(RolloutMetadata(ac=ac, qra=qra))

        results = await asyncio.gather(*verification_tasks)
        return TaskRolloutResult.from_results(model_input, results, metadata_list)

    def _parse_qra_from_message(self, question: str, message: Message) -> QRA:
        content = message["content"]
        reasoning = ""
        answer_text = ""

        if isinstance(content, list):
            for part in content:
                if part["type"] == "thinking":
                    reasoning += part["thinking"]
                elif part["type"] == "text":
                    answer_text += part["text"]
        else:
            answer_text = content

        # DRY: Extract reasoning from answer text if thinking part was missing
        if not reasoning:
            reasoning = answer_text.split("```rust")[0].strip()

        return QRA(
            question=question,
            reasoning=reasoning,
            answer=extract_rust_code(answer_text),
        )

    def _get_required_task_counts(self, knowledge_id: str) -> tuple[int, int]:
        """
        Calculate required and target (with buffer) task counts for a knowledge item.
        """
        if knowledge_id not in self.initialized_knowledge:
            required_count = self.config.k_init_sft + self.config.k_rl
        else:
            needs_sft = (
                self.performance_map.get(knowledge_id, 0.0) < self.config.sft_threshold
            )
            required_count = (self.config.k_sft if needs_sft else 0) + self.config.k_rl

        target_count = int(required_count * self.config.over_generation_factor)
        return required_count, target_count

    async def _wrap_failed_trajectory(self, question: str) -> RLTrajectory:
        """
        Wrap a failed trajectory creation in an async call for uniform handling.
        """
        return RLTrajectory.failed(question)

    def _should_perform_grpo(self, task_results: list[TaskRolloutResult]) -> bool:
        return any(tr.group is not None for tr in task_results)

    async def _perform_grpo_update(self, task_results: list[TaskRolloutResult]):
        assert self.training_client is not None
        groups = [tr.group for tr in task_results if tr.group is not None]

        logger.info(f"Triggering GRPO training for batch of {len(groups)} groups...")
        grpo_result = await compute_grpo_loss(
            groups, self.training_client, self.config.rl_optimizer_params
        )

        optim_future = await self.training_client.optim_step_async(
            self.config.rl_optimizer_params.adam_params
        )
        await optim_future.result_async()

        if hasattr(grpo_result, "metrics"):
            self._log_metrics("rl", grpo_result.metrics)

    def _summarize_rl_results(
        self, task_results: list[TaskRolloutResult]
    ) -> RLStepResult:
        return RLStepResult.from_task_rollouts(task_results)

    def _log_metrics(self, prefix: str, metrics: dict):
        """
        Helper to log metrics with a prefix to the initialized ml_logger.
        """
        if self.ml_logger:
            self.ml_logger.log_metrics({f"{prefix}/{k}": v for k, v in metrics.items()})

    def _update_wandb_table(self, iteration: int, rl_sample: RLTrajectory | None):
        """
        Add a row to the persistent WandB samples table.
        """
        if self.ml_logger and rl_sample and self._samples_table is not None:
            self._samples_table.add_data(
                iteration,
                rl_sample.question,
                rl_sample.reasoning,
                rl_sample.answer,
                rl_sample.success,
            )
            wandb.log({"eval/samples": self._samples_table})

    async def _sync_sampling_weights(self):
        """
        Sync updated weights from training client to the solver model.
        """
        if self.training_client and self.solver:
            logger.info("Syncing updated weights to sampling model...")
            new_sampling_client = (
                await self.training_client.save_weights_and_get_sampling_client_async()
            )
            self.solver.model.sampling_client = new_sampling_client
            logger.info("Sampling weights successfully updated.")

    def _get_solver_system_prompt(self, library_name: str) -> str:
        return f"""<Role>
You are an expert Rust engineer.
Your task is to solve the programming challenge using the `{library_name}` library.
</Role>

<Guidelines>
1. Write high-quality, idiomatic Rust code.
2. Ensure your solution is complete and self-contained.
3. Ensure that your code produces clear output during execution so that its correctness can be easily verified from the execution results.
</Guidelines>
"""

    async def _verify_with_execution(self, question: str, rollout: QRA) -> RLTrajectory:
        """
        Actually run the Rust code and verify it.
        Returns a full RLTrajectory.
        """
        try:
            async with self.config.runtime_settings.build_runtime() as runtime:
                # 1. Write the solution code
                await runtime.set_content("src/main.rs", rollout.answer)

                # 2. Compile and Run
                execution_output, exit_success = await runtime.run_cargo()

                if not exit_success:
                    return RLTrajectory(
                        question=question,
                        reasoning=rollout.reasoning,
                        answer=rollout.answer,
                        execution_output=execution_output,
                        verification_reasoning="Compilation or runtime execution failed. Early exit before LLM verification.",
                        success=False,
                    )

                # 3. Get tree structure
                tree_output = await runtime.tree()

                # 4. Verify
                verification_result = await self.verifier.verify(
                    qa=rollout,
                    tree_structure=tree_output,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer,
                )
                return RLTrajectory(
                    question=question,
                    reasoning=rollout.reasoning,
                    answer=rollout.answer,
                    execution_output=execution_output,
                    verification_reasoning=verification_result.reasoning,
                    success=verification_result.success,
                )
        except Exception as e:
            logger.exception(
                f"Execution/Verification failed for question: {question[:50]}..."
            )
            return RLTrajectory(
                question=question,
                reasoning=rollout.reasoning,
                answer=rollout.answer,
                execution_output=str(e),
                verification_reasoning="System error during execution.",
                success=False,
            )

    def _visualize_iteration(
        self,
        iteration: int,
        sft_sample: RLTrajectory | None,
        rl_sample: RLTrajectory | None,
        include_header: bool = True,
        include_sft: bool = True,
    ):
        """
        Log a representative sample from SFT and RL phases using Rich for a better UX.
        """
        console = Console()

        if include_header:
            console.print()
            console.print(
                Rule(
                    f"[bold cyan]Iteration {iteration} Summary[/bold cyan]",
                    style="cyan",
                )
            )
            console.print()

        if include_sft and sft_sample:
            console.print(
                self._create_sample_panel(
                    sft_sample, "Phase 1: Teacher (SFT) Sample", "magenta"
                )
            )
            console.print()

        if rl_sample:
            console.print(
                self._create_sample_panel(
                    rl_sample,
                    "Phase 2: Student (RL) Sample",
                    "green" if rl_sample.success else "yellow",
                )
            )
            console.print()

        console.print(Rule(style="cyan"))
        console.print()

    def _create_sample_panel(self, sample: RLTrajectory, title: str, color: str) -> Panel:
        """
        Create a Rich Panel for a single trajectory sample.
        """
        success_tag = (
            "[bold green]✅ SUCCESS[/bold green]"
            if sample.success
            else "[bold red]❌ FAILURE[/bold red]"
        )

        content = Group(
            Markdown(f"### Question\n{sample.question}"),
            Rule(style="dim"),
            Markdown(f"### Thought\n> {sample.reasoning}"),
            Rule(style="dim"),
            Markdown(f"### Solution\n```rust\n{sample.answer}\n```"),
            Rule(style="dim"),
            Markdown(f"### Execution Output\n```text\n{sample.execution_output}\n```"),
            Rule(style="dim"),
            Markdown(f"### Verification Judgment\n> {sample.verification_reasoning}"),
        )

        return Panel(
            content,
            title=f"{title} - {success_tag}",
            border_style=color,
            expand=False,
        )
