import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Self

import ray
import tinker
from agents.extensions.models.litellm_model import LitellmModel
from coder_mcp.runtime import Runtime
from more_itertools import chunked
from oai_utils.tinker import setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from ray.actor import ActorProxy
from tinker import SamplingClient
from tinker.types.loss_fn_type import LossFnType
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

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.grpo import compute_grpo_loss
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.internalize_toobig.global_state import (
    GlobalState,
    InternalizationKnowledge,
    KnowledgeMasteryManager,
)
from adapter_agent.internalize_toobig.qra_generator import QRAGenerator
from adapter_agent.internalize_toobig.types import (
    GroupRolloutResult,
    InternalizationQRA,
    InternalizationTask,
    MasteryConfig,
    SingleRolloutResult,
)
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    ModelLoadingSettings,
    OptimizerParams,
)
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)


@dataclass
class RemoteWorker:
    global_state: ActorProxy[GlobalState]
    library: Library
    model_name: str
    runtime_settings: RuntimeSettings
    k_rollout: int
    task_per_worker: int

    # Non-picklable internal state (initialized in setup)
    verifier_model: Optional[LitellmModel] = field(init=False, default=None)
    rust_doc_analyzer: Optional[AsyncRustDocAnalyzer] = field(init=False, default=None)
    verifier: Optional[Verifier] = field(init=False, default=None)
    renderer: Optional[Renderer] = field(init=False, default=None)
    runtime_pool: Optional[RuntimePool] = field(init=False, default=None)

    async def setup(self) -> None:
        """Initialize non-picklable components."""
        setup_base_loglevel()
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
        self.renderer = renderer

        self.runtime_pool = RuntimePool(
            self.runtime_settings, max_size=self.task_per_worker * self.k_rollout
        )

    def __repr__(self) -> str:
        return f"RemoteWorker(model={self.model_name})"

    async def run_loop(self) -> None:
        """
        Main loop for the worker: starts multiple concurrent task loops.
        """
        await self.setup()
        loops = [self._single_run_loop(i) for i in range(self.task_per_worker)]
        await asyncio.gather(*loops)

    async def _single_run_loop(self, loop_id: int) -> None:
        """
        Single task loop: pull tasks from GlobalState and perform rollouts.
        """
        logger.info(f"Starting RemoteWorker task loop {loop_id} (actor={self})")
        while True:
            task: InternalizationTask | None = None
            try:
                # 1. Get current sampling client and version
                indexed_client: IndexedSamplingClient = (
                    await self.global_state.get_sampling_client.remote()  # type: ignore[assignment]
                )

                # 2. Pop task
                task = await self.global_state.pop_rollout_task.remote()  # type: ignore[assignment]
                if task is None:
                    await asyncio.sleep(1)
                    continue

                # 3. Perform rollout and push
                result = await self.rollout_task_group(
                    task, indexed_client.client, indexed_client.version
                )
                await self.global_state.push_rollout_result.remote(result)  # type: ignore[attr-defined]
            except Exception as e:
                logger.exception(f"RemoteWorker error in loop {loop_id}: {e}")
                if task:
                    # Notify GlobalState to release the task slot
                    await self.global_state.report_rollout_failure.remote(  # type: ignore[attr-defined]
                        task.knowledge_id, task.id
                    )
                await asyncio.sleep(5)

    async def rollout_task_group(
        self,
        task: InternalizationTask,
        client: SamplingClient,
        sampling_version: int,
    ) -> GroupRolloutResult:
        """Sample multiple trajectories and verify them."""
        assert self.renderer is not None, (
            "Renderer not initialized. Call setup() first."
        )

        # 1. Prepare prompt
        system_prompt = self._get_solver_system_prompt()
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task.instruction),
        ]

        # Build real ModelInput (observation)
        ob = self.renderer.build_generation_prompt(messages)

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
            msg, success = self.renderer.parse_response(tokens)

            if not success:
                coros.append(self._to_async_parse_failed(task.instruction, trajectory))
            else:
                qra = self._parse_qra_from_message(task.instruction, msg)
                coros.append(self._verify_with_execution(qra, trajectory))

        results = await asyncio.gather(*coros)
        return GroupRolloutResult(
            task_id=task.id,
            trajectories=results,
            knowledge_id=task.knowledge_id,
            current_sampling_version=sampling_version,
        )

    async def _to_async_parse_failed(
        self, question: str, trajectory: Trajectory
    ) -> SingleRolloutResult:
        return SingleRolloutResult.parse_failed(question, trajectory)

    async def _verify_with_execution(
        self, rollout: QRA, trajectory: Trajectory
    ) -> SingleRolloutResult:
        """Run Rust code and verify."""
        assert self.verifier is not None
        assert self.runtime_pool is not None

        async def _run(runtime: Runtime) -> SingleRolloutResult:
            code = extract_rust_code(rollout.answer)
            await runtime.set_content("src/main.rs", code)
            execution_output, exit_success = await runtime.run_cargo()

            if not exit_success:
                return SingleRolloutResult(
                    question=rollout.question,
                    reasoning=rollout.reasoning,
                    answer=rollout.answer,
                    execution_output=execution_output,
                    main_rs_content=rollout.answer,
                    success=False,
                    verification_reasoning="Compilation or runtime execution failed.",
                    trajectory=trajectory,
                )

            tree_output = await runtime.tree()

            verification_result = await self.verifier.verify(
                qa=rollout,
                tree_structure=tree_output,
                execution_output=execution_output,
                main_rs_content=rollout.answer,
            )

            return SingleRolloutResult(
                question=rollout.question,
                reasoning=rollout.reasoning,
                answer=rollout.answer,
                execution_output=execution_output,
                main_rs_content=rollout.answer,
                success=verification_result.success,
                verification_reasoning=verification_result.reasoning,
                trajectory=trajectory,
            )

        try:
            return await self.runtime_pool.execute_with_retry(_run)
        except Exception as e:
            logger.error(f"Rollout verification error: {e}")
            return SingleRolloutResult.parse_failed(rollout.question, trajectory)

    def _get_solver_system_prompt(self) -> str:
        return f"""<Role>
You are an expert Rust engineer.
Your task is to solve the programming challenge using the `{self.library.name}` library.
</Role>

<Guidelines>
1. Write high-quality, idiomatic Rust code.
2. Ensure your solution is complete and self-contained.
3. Ensure that your code produces clear output during execution so that its correctness can be easily verified from the execution results.
4. Your response should include a natural language explanation, and the complete code MUST be enclosed in a ```rust ... ``` code block.
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
            answer=answer_text,
        )


@dataclass
class Internalizer:
    global_state: ActorProxy[GlobalState]
    workers: list[ActorProxy[RemoteWorker]]
    qra_generator: ActorProxy[QRAGenerator]
    training_client: tinker.TrainingClient  # Mandatory

    # Training Configuration
    model_loading_settings: ModelLoadingSettings
    sft_batch_size: int
    sft_adam_params: tinker.AdamParams
    rl_adam_params: tinker.AdamParams
    rl_loss_fn: LossFnType
    min_sft_batch_size: int
    min_rl_batch_size: int
    task_per_worker: int = 4

    max_iterations: int = 100
    rl_step_count: int = field(init=False, default=0)
    generator_concurrency: int = 16
    status_log_frequency: int = 5

    @classmethod
    async def start(
        cls,
        knowledges: list[Knowledge],
        sampling_client: SharedSamplingClient,
        library: Library,
        model_name: str,
        runtime_settings: RuntimeSettings,
        sft_batch_size: int,
        sft_adam_params: tinker.AdamParams,
        rl_adam_params: tinker.AdamParams,
        rl_loss_fn: LossFnType,
        min_sft_batch_size: int,
        min_rl_batch_size: int,
        k_rollout: int = 8,
        task_per_worker: int = 4,
        studying_threshold: float = 0.2,
        success_threshold: float = 0.5,
        overgen_factor: float = 1.5,
        k_sft: int = 16,
        k_rl: int = 16,
        num_workers: int = 4,
        max_iterations: int = 50,
        generator_concurrency: int = 16,
        status_log_frequency: int = 5,
    ) -> Self:
        """
        Initialize Ray actors and return a started Internalizer.
        """
        setup_base_loglevel()
        # 0. Prep Mastery Manager
        mastery_config = MasteryConfig(
            studying_threshold=studying_threshold,
            success_threshold=success_threshold,
            overgen_factor=overgen_factor,
            k_sft=k_sft,
            k_rl=k_rl,
        )

        internalization_knowledges = {
            k.id: InternalizationKnowledge(
                knowledge=k,
                qras=[],
                groups=[],
                running_qra_generation=0,
                running_sft={},
                running_rl={},
                mastery_config=mastery_config,
            )
            for k in knowledges
        }

        knowledge_manager = KnowledgeMasteryManager(
            knowledges=internalization_knowledges,
            mastery_config=mastery_config,
        )

        # 1. Start GlobalState actor
        GlobalStateActor = ray.remote(GlobalState)
        global_state = GlobalStateActor.remote(
            sampling_client,
            knowledge_manager,
        )

        # 2. Start QRAGenerator actor
        QRAGeneratorActor = ray.remote(QRAGenerator)
        qra_generator: ActorProxy[QRAGenerator] = QRAGeneratorActor.remote(
            global_state,
            library=library,
            verifier=None,
            runtime_settings=runtime_settings,
            num_concurrent_generations=generator_concurrency,
        )

        # 3. Start RemoteWorker actors
        RemoteWorkerActor = ray.remote(RemoteWorker)
        workers = [
            RemoteWorkerActor.remote(
                global_state,
                library=library,
                model_name=model_name,
                runtime_settings=runtime_settings,
                k_rollout=k_rollout,
                task_per_worker=task_per_worker,
            )
            for _ in range(num_workers)
        ]

        # 4. Logger Setup via GlobalState
        log_dir = Path("logs") / "internalizer" / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        await global_state.setup_logging.remote(  # type: ignore[attr-defined]
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": model_name,
                "library": library.name,
                "num_workers": num_workers,
                "studying_threshold": studying_threshold,
                "success_threshold": success_threshold,
                "overgen_factor": overgen_factor,
                "k_sft": k_sft,
                "k_rl": k_rl,
                "task_per_worker": task_per_worker,
            },
        )

        # 4. Training Client Setup
        logger.info("Setting up Tinker TrainingClient...")
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=32,
        )

        # 5. Create Internalizer instance
        return cls(
            global_state=global_state,
            workers=workers,
            qra_generator=qra_generator,
            training_client=training_client,
            model_loading_settings=ModelLoadingSettings(
                model_name=model_name,
                lora_rank=32,
            ),
            sft_batch_size=sft_batch_size,
            sft_adam_params=sft_adam_params,
            rl_adam_params=rl_adam_params,
            rl_loss_fn=rl_loss_fn,
            min_sft_batch_size=min_sft_batch_size,
            min_rl_batch_size=min_rl_batch_size,
            max_iterations=max_iterations,
            generator_concurrency=generator_concurrency,
            status_log_frequency=status_log_frequency,
            task_per_worker=task_per_worker,
        )

    async def run(self) -> None:
        """Entry point for the internalizer."""
        logger.info(f"Starting internalizer (max_iterations={self.max_iterations})")

        # Start workers and generator in the background
        for worker in self.workers:
            worker.run_loop.remote()  # type: ignore[attr-defined]
        self.qra_generator.run_loop.remote()  # type: ignore[attr-defined]

        await self.main_loop()

    async def main_loop(self) -> None:
        """
        Main reactive loop for the internalization process.
        Prioritize RL updates for off-policy data, then SFT updates.
        """
        iteration_count = 0
        while self.rl_step_count < self.max_iterations:
            if iteration_count % self.status_log_frequency == 0:
                await self.global_state.report_detailed_status.remote()  # type: ignore[attr-defined]

            logger.debug(
                f"--- Monitoring (RL Steps: {self.rl_step_count}/{self.max_iterations}) ---"
            )

            # 1. RL Update (Priority): Use pop_rl_batch to ensure efficiency
            rollout_results: (
                list[RLGroup] | None
            ) = await self.global_state.pop_rl_batch.remote(self.min_rl_batch_size)  # type: ignore[assignment]
            if rollout_results:
                await self.rl_step(rollout_results)
                iteration_count += 1
                continue

            # 2. SFT Update (Secondary): Use pop_sft_batch to ensure efficiency
            sft_dataset: (
                list[InternalizationQRA] | None
            ) = await self.global_state.pop_sft_batch.remote(self.min_sft_batch_size)  # type: ignore[assignment]
            if sft_dataset:
                await self.sft_step(sft_dataset)
                iteration_count += 1
                continue

            # 3. Idle Phase: Wait if no action was taken
            await asyncio.sleep(5)
            iteration_count += 1

        logger.info("Internalization process complete.")

    async def rl_step(self, rl_groups: list[RLGroup]) -> None:
        """Perform combined GRPO update using collected trajectories."""

        # 2. Compute aggregate metrics
        total_trajectories = sum(len(g.trajectories) for g in rl_groups)
        total_reward = sum(sum(g.rewards) for g in rl_groups)
        mean_reward = (
            total_reward / total_trajectories if total_trajectories > 0 else 0.0
        )

        logger.info(
            f"RL STEP: groups={len(rl_groups)}, trajectories={total_trajectories}, mean_reward={mean_reward:.4f}"
        )

        # 3. Update model
        await self.exec_rl(rl_groups)
        self.rl_step_count += 1

        # 4. Log metrics
        current_version = await self.global_state.get_current_version.remote()  # type: ignore[attr-defined]
        await self.global_state.log_metrics.remote(  # type: ignore[attr-defined]
            {
                "rl/trajectories": total_trajectories,
                "rl/mean_reward": mean_reward,
                "rl/total_reward": total_reward,
            },
            step=current_version,
        )

    async def sft_step(self, sft_dataset: list[InternalizationQRA]) -> None:
        """
        Perform a single SFT step: train on the provided QRA dataset.
        """
        logger.info(f"Triggering SFT update with {len(sft_dataset)} samples.")
        await self.exec_sft(sft_dataset)

        # Mark updated knowledges to current version
        await self.global_state.report_sft_results.remote(sft_dataset)  # type: ignore[attr-defined]

    async def exec_sft(self, qras: list[InternalizationQRA]) -> None:
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
        for batch in chunked(data, self.sft_batch_size):
            fwd_bwd = await self.training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
            )
            optim = await self.training_client.optim_step_async(self.sft_adam_params)

            res = await fwd_bwd.result_async()
            await optim.result_async()

            # Log all available metrics from the server
            current_version = await self.global_state.get_current_version.remote()
            metrics = {f"sft/{k}": v for k, v in res.metrics.items()}
            metrics["sft/batch_samples"] = len(batch)
            await self.global_state.log_metrics.remote(metrics, step=current_version)  # type: ignore[attr-defined]

        await self._sync_sampling_weights()

    async def exec_rl(self, dataset: list[RLGroup]) -> None:
        """Execute RL (GRPO) training step."""
        logger.info(f"Triggering GRPO training for batch of {len(dataset)} groups...")
        assert self.training_client is not None

        # Reconstruct minimal OptimizerParams for compute_grpo_loss
        minimal_params = OptimizerParams(
            adam_params=self.rl_adam_params,
            loss_fn=self.rl_loss_fn,
            num_steps=1,  # Not used by compute_grpo_loss
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
        )

        grpo_res = await compute_grpo_loss(
            dataset, self.training_client, minimal_params
        )
        logger.info("GRPO loss computed, performing optimizer step...")
        optim_future = await self.training_client.optim_step_async(self.rl_adam_params)
        await optim_future.result_async()

        if hasattr(grpo_res, "metrics"):
            current_version = await self.global_state.get_current_version.remote()  # type: ignore[attr-defined]
            await self.global_state.log_metrics.remote(  # type: ignore[attr-defined]
                {f"rl/{k}": v for k, v in grpo_res.metrics.items()},
                step=current_version,
            )

        await self._sync_sampling_weights()

    async def _sync_sampling_weights(self) -> None:
        """Sync updated weights to GlobalState's SharedSamplingClient."""
        logger.info("Syncing updated weights to sampling model...")
        new_sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        await self.global_state.update_sampling_client.remote(new_sampling_client)  # type: ignore[attr-defined]
