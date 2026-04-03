import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

from more_itertools import chunked


import tinker
from oai_utils.agent import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QA, QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.solver import SolverAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.rl.config import (
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    SFTOptimizerParams,
)
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)
# Suppress noisy LiteLLM logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@dataclass
class PipelineConfig:
    api_doc_path: Path
    runtime_settings: RuntimeSettings
    model_loading_settings: ModelLoadingSettings
    sft_optimizer_params: SFTOptimizerParams
    rl_optimizer_params: OptimizerParams
    experiment_settings: ExperimentSettings
    k_sft: int = 16
    k_rl: int = 16
    k_rollout: int = 16
    concurrency: int = 8
    max_iterations: int = 10


@dataclass
class RLTrajectory:
    question: str
    reasoning: str
    answer: str
    execution_output: str
    verification_reasoning: str
    success: bool


class InternalizationPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        generator_model: AgentsSDKModel,
        verifier_model: AgentsSDKModel,
        rust_doc_analyzer: AsyncRustDocAnalyzer,
    ):
        self.config = config
        self.rust_doc_analyzer = rust_doc_analyzer

        # Agents
        self.generator = GeneratorAgent(
            model=generator_model, rust_doc_analyzer=rust_doc_analyzer
        )
        self.verifier = Verifier(
            model=verifier_model, rust_doc_analyzer=rust_doc_analyzer
        )

        # Training/Solver (Initialized in setup)
        self.training_client: Optional[tinker.TrainingClient] = None
        self.service_client: Optional[tinker.ServiceClient] = None
        self.ml_logger: Optional[MLLogger] = None

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
            import wandb

            self._samples_table = wandb.Table(
                columns=["Iteration", "Question", "Reasoning", "Code", "Success"]
            )

        logger.info("Pipeline components successfully initialized.")

    async def run(self):
        """
        Main loop: SFT -> RL -> (SFT/RL)
        """
        await self.setup()
        logger.info(f"Starting API internalization for doc: {self.config.api_doc_path}")

        rollout_success_ratio = 0.0
        task_success_ratio = 0.0
        iteration = 0

        while iteration < self.config.max_iterations and rollout_success_ratio < 1.0:
            iteration += 1
            logger.info(f"--- Iteration {iteration} ---")

            # Step 1: Conditional SFT (Warm up / Recovery)
            # Skip SFT if task success ratio is at least 50%
            sft_sample = None
            if task_success_ratio < 0.5:
                sft_sample, sft_tasks = await self.execute_sft_step()
                eval_tasks = [
                    QA(question=t.question, answer=t.answer)
                    for t in sft_tasks[: self.config.k_rl]
                ]
            else:
                logger.info(
                    f"Skipping SFT Step (Task Success: {task_success_ratio:.2%}). Generating fresh tasks..."
                )
                # Still generate tasks using the teacher to maintain quality
                fresh_tasks = await self.generate_teacher_tasks(self.config.k_rl)
                eval_tasks = [QA(question=t.question, answer=t.answer) for t in fresh_tasks]

            # Step 2: RL (Exploration / Correction)
            (rollout_success_ratio, task_success_ratio, rl_sample) = (
                await self.execute_rl_step(eval_tasks=eval_tasks)
            )

            # Step 3: W&B Iteration Summary
            self._log_metrics(
                "overall",
                {
                    "rollout_success_ratio": rollout_success_ratio,
                    "task_success_ratio": task_success_ratio,
                },
            )
            self._update_wandb_table(iteration, rl_sample)

            # Step 4: Visualization / Logging
            self._visualize_iteration(iteration, sft_sample, rl_sample)

            logger.info(
                f"Iteration {iteration} complete. "
                f"Rollout Success: {rollout_success_ratio:.2%}, "
                f"Task Success: {task_success_ratio:.2%}"
            )


    async def generate_teacher_tasks(self, count: int) -> List[QRA]:
        """
        Generate and verify high-quality QRA tasks using the teacher model.
        """
        logger.info(f"Teacher generating {count} high-quality tasks...")
        topic_hint = self.config.api_doc_path.stem

        # 1. Generate
        async def generate_task(_):
            return await self.generator.generate_sft(topic_hint=topic_hint)

        tasks_raw: List[QRA] = await self._run_parallel(generate_task, range(count))

        # 2. Verify
        logger.info(f"Verifying {len(tasks_raw)} teacher-generated tasks...")

        async def verify_task(qra: QRA):
            return await self._verify_with_execution(qra.question, qra)

        results: List[RLTrajectory] = await self._run_parallel(verify_task, tasks_raw)

        # 3. Handle failures? For now, we return all, but verifier result in QRA is important.
        # Actually, let's filter only successful ones to ensure RL always has valid data.
        valid_tasks = [tasks_raw[i] for i, res in enumerate(results) if res.success]

        # If too few were valid, we might want to try again or just proceed with what we have.
        if len(valid_tasks) < count:
            logger.warning(
                f"Only {len(valid_tasks)}/{count} teacher tasks were valid (verified)."
            )

        return valid_tasks

    async def execute_sft_step(self) -> tuple[Optional[RLTrajectory], List[QRA]]:
        """
        Generate QRA tasks and perform supervised fine-tuning.
        """
        logger.info("Executing SFT Step...")
        assert self.training_client is not None, "TrainingClient must be initialized"

        # 1. Generate & Verify teacher tasks
        qra_list = await self.generate_teacher_tasks(self.config.k_sft)
        teacher_success_ratio = len(qra_list) / self.config.k_sft if self.config.k_sft else 0
        self._log_metrics("sft", {"teacher_success_ratio": teacher_success_ratio})

        # Find a representative successful sample for logging
        # We need to re-verify or just take the first one if we want RLTrajectory
        # Actually, generate_teacher_tasks returns QRA, not RLTrajectory.
        # Let's perform a single verification again for the representative if needed,
        # or update generate_teacher_tasks to return both.
        # For simplicity, we just use the first valid QRA to create a dummy RLTrajectory
        # for logging purposes in visualize_iteration.
        representative_sft_sample = None
        if qra_list:
            # We'll just run one verification to get the full RLTrajectory record
            representative_sft_sample = await self._verify_with_execution(
                qra_list[0].question, qra_list[0]
            )

        # 3. Train on successful tasks
        if qra_list:
            logger.info(
                f"Fine-tuning student on {len(qra_list)}/{self.config.k_sft} VALID QRA samples..."
            )

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

            batch_size = self.config.sft_optimizer_params.batch_size
            num_epochs = self.config.sft_optimizer_params.num_epochs
            adam_params = self.config.sft_optimizer_params.adam_params

            batches = list(chunked(datums, batch_size))

            # Helper to get next batch index (epoch, step)
            def get_next_batch_idx(ep: int, st: int) -> tuple[int | None, int | None]:
                if st + 1 < len(batches):
                    return ep, st + 1
                elif ep + 1 < num_epochs:
                    return ep + 1, 0
                return None, None

            # 1. Enqueue the very first batch
            logger.info("Initializing SFT training loop with look-ahead...")
            fwd_bwd_future = await self.training_client.forward_backward_async(
                data=batches[0], loss_fn="cross_entropy"
            )
            optim_future = await self.training_client.optim_step_async(adam_params)

            # 2. Asynchronous look-ahead loop
            for epoch in range(num_epochs):
                for step, _ in enumerate(batches):
                    # Enqueue NEXT step before awaiting current
                    next_ep, next_st = get_next_batch_idx(epoch, step)
                    if next_ep is not None and next_st is not None:
                        next_fwd_bwd_future = (
                            await self.training_client.forward_backward_async(
                                data=batches[next_st], loss_fn="cross_entropy"
                            )
                        )
                        next_optim_future = (
                            await self.training_client.optim_step_async(adam_params)
                        )
                    else:
                        next_fwd_bwd_future = None
                        next_optim_future = None

                    # Await CURRENT step results
                    assert fwd_bwd_future is not None
                    assert optim_future is not None

                    fwd_bwd_result = await fwd_bwd_future.result_async()
                    await optim_future.result_async()


                    # Log metrics
                    # Prefix by epoch for trend tracking
                    self._log_metrics(
                        f"sft/epoch_{epoch + 1}/step_{step + 1}", fwd_bwd_result.metrics
                    )
                    # Global SFT logs
                    self._log_metrics("sft", fwd_bwd_result.metrics)

                    # Find appropriate loss metric (handle 'loss', 'loss:sum', etc.)
                    current_loss = fwd_bwd_result.metrics.get("loss")
                    if current_loss is None:
                        for k, v in fwd_bwd_result.metrics.items():
                            if k.startswith("loss"):
                                current_loss = v
                                break

                    if step % 5 == 0:
                        logger.info(
                            f"SFT Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{len(batches)}: "
                            f"loss={current_loss if current_loss is not None else 'N/A'}"
                        )


                    # Move next to current
                    fwd_bwd_future = next_fwd_bwd_future
                    optim_future = next_optim_future


            logger.info(f"SFT Step complete after {num_epochs} epochs.")

        return (representative_sft_sample, qra_list)

    async def execute_rl_step(
        self, eval_tasks: Optional[List[QA]] = None
    ) -> tuple[float, float, Optional[RLTrajectory]]:
        """
        Generate evaluation tasks, rollout solutions, verify (with execution), and train GRPO if success exists.
        Returns the (rollout_success_ratio, task_success_ratio, sample_trajectory).
        """

        logger.info("Executing RL Step...")

        # 1. Obtain Evaluation Tasks
        if eval_tasks is None:
            logger.info(f"Generating {self.config.k_rl} RL eval tasks...")
            modules = self.rust_doc_analyzer.get_modules()

            async def generate_rl_task(i: int) -> QA:
                topic_hint = modules[i % len(modules)] if modules else None
                logger.info(
                    f"Starting RL eval task generation {i + 1}/{self.config.k_rl} (Hint: {topic_hint})"
                )
                return await self.generator.generate_rl(topic_hint=topic_hint)

            eval_tasks = await self._run_parallel(
                generate_rl_task, range(self.config.k_rl)
            )
        else:
            logger.info(f"Re-using {len(eval_tasks)} tasks from SFT for RL Step.")

        # 2. Rollout & Verify for each task
        total_rollouts = 0
        total_successes = 0
        rl_sample: RLTrajectory | None = None
        all_rl_groups: List[RLGroup] = []

        library_name = self.config.api_doc_path.stem
        system_prompt = self._get_solver_system_prompt(library_name)

        async def process_task(qa: QA):
            task_label = f"Task: {qa.question[:40]}..."
            logger.info(f"[{task_label}] Rolling out solutions...")

            assert self.solver is not None
            assert isinstance(self.solver.model, TinkerModel)

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=qa.question),
            ]
            model_input = self.solver.model.renderer.build_generation_prompt(messages)

            # Sample k solutions
            sample_results = await self.solver.model.sampling_client.sample_async(
                prompt=model_input,
                num_samples=self.config.k_rollout,
                sampling_params=tinker.SamplingParams(
                    include_logprobs=True,
                ),
            )

            rollout_metadata = []
            verification_tasks = []

            for seq in sample_results.sequences:
                ac = TokensWithLogprobs(tokens=seq.tokens, maybe_logprobs=seq.logprobs)
                assistant_message, parse_success = (
                    self.solver.model.renderer.parse_response(ac.tokens)
                )

                if not parse_success:

                    async def failed_result():
                        return RLTrajectory(
                            question=qa.question,
                            reasoning="Parse Error",
                            answer="N/A",
                            execution_output="The model output could not be parsed as a valid message.",
                            verification_reasoning="Parse failure.",
                            success=False,
                        )

                    verification_tasks.append(failed_result())
                    r_qra = QRA(
                        question=qa.question, reasoning="Parse Error", answer=""
                    )
                    rollout_metadata.append({"ac": ac, "qra": r_qra})
                    continue

                # Extract reasoning and answer content
                content = assistant_message["content"]
                reasoning = ""
                answer_text = ""

                if isinstance(content, list):
                    for part in content:
                        if part["type"] == "thinking":
                            reasoning += part["thinking"]
                        elif part["type"] == "text":
                            answer_text += part["text"]

                    # If thinking part is empty, fallback to extracting reasoning from start of text
                    if not reasoning and answer_text:
                        reasoning = answer_text.split("```rust")[0].strip()
                else:
                    answer_text = content
                    reasoning = answer_text.split("```rust")[0].strip()

                code = extract_rust_code(answer_text)
                r_qra = QRA(question=qa.question, reasoning=reasoning, answer=code)

                verification_tasks.append(
                    self._verify_with_execution(qa.question, r_qra)
                )

                # Prepare metadata for trajectory building
                rollout_metadata.append({"ac": ac, "qra": r_qra})

            # Parallel Verification per task (inner parallelism)
            results: List[RLTrajectory] = await asyncio.gather(*verification_tasks)

            final_trajectories = []
            rewards = []
            task_rl_sample: RLTrajectory | None = None

            for i, res in enumerate(results):
                reward = 1.0 if res.success else 0.0
                rewards.append(reward)

                ac_action = cast(TokensWithLogprobs, rollout_metadata[i]["ac"])
                transition = Transition(
                    ob=model_input,
                    ac=ac_action,
                    reward=reward,
                    episode_done=True,
                )

                final_trajectories.append(
                    Trajectory(
                        transitions=[transition], final_ob=tinker.ModelInput.empty()
                    )
                )

                if task_rl_sample is None and res.success:
                    task_rl_sample = res

            if task_rl_sample is None and results:
                task_rl_sample = results[0]

            num_success = sum(1 for res in results if res.success)

            logger.info(
                f"[{task_label}] Completion: {num_success}/{self.config.k_rollout} successful."
            )

            group: RLGroup | None = None
            if final_trajectories:
                if len(set(rewards)) > 1:
                    group = RLGroup(trajectories=final_trajectories, rewards=rewards)
                else:
                    logger.info(
                        f"[{task_label}] Skipping GRPO group: all rewards are equal ({rewards[0] if rewards else 'N/A'})."
                    )

            return num_success, len(results), group, task_rl_sample

        # Use semaphore for task-level concurrency control (outer parallelism)
        task_results = await self._run_parallel(process_task, eval_tasks)

        for n_success, n_rollouts, group, task_rl_sample in task_results:
            total_successes += n_success
            total_rollouts += n_rollouts
            if group:
                all_rl_groups.append(group)
            # Pick first success for sample logging
            if rl_sample is None and task_rl_sample and task_rl_sample.success:
                rl_sample = task_rl_sample

        if rl_sample is None and task_results:
            # Fallback to first non-None sample if no success in any task
            for res in task_results:
                if res[3] is not None:
                    rl_sample = res[3]
                    break

        # 3. Trigger GRPO training for the entire batch
        if self.training_client and all_rl_groups:
            has_success = any(r > 0 for g in all_rl_groups for r in g.rewards)
            if has_success:
                logger.info(
                    f"Triggering GRPO training for batch of {len(all_rl_groups)} groups..."
                )

                from adapter_agent.hierarchical.grpo import compute_grpo_loss

                grpo_result = await compute_grpo_loss(
                    all_rl_groups, self.training_client, self.config.rl_optimizer_params
                )

                # Optimization step
                optim_future = await self.training_client.optim_step_async(
                    self.config.rl_optimizer_params.adam_params
                )
                await optim_future.result_async()

                # Check for metrics
                loss_val = "N/A"
                if hasattr(grpo_result, "metrics"):
                    loss_val = grpo_result.metrics.get("loss", "N/A")
                    self._log_metrics("rl", grpo_result.metrics)

                logger.info(f"GRPO Batch Update complete. Loss: {loss_val}")
            else:
                logger.info(
                    "Skipping GRPO batch update: no successful samples found in any group."
                )

        success_ratio = total_successes / total_rollouts if total_rollouts > 0 else 0.0
        task_successes = sum(1 for res in task_results if res[0] > 0)
        task_success_ratio = task_successes / len(task_results) if task_results else 0.0

        # Log Metrics
        if self.ml_logger:
            self.ml_logger.log_metrics(
                {
                    "rl/rollout_success_ratio": success_ratio,
                    "rl/task_success_ratio": task_success_ratio,
                }
            )

        # 4. Sync Sampling Weights
        await self._sync_sampling_weights()

        return success_ratio, task_success_ratio, rl_sample


    async def _run_parallel(self, coro_func, items: list | range) -> list:
        """
        Execute an async function over a list of items with concurrency control.
        """
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def sem_task(item):
            async with semaphore:
                return await coro_func(item)

        tasks = [sem_task(item) for item in items]
        return await asyncio.gather(*tasks)

    def _log_metrics(self, prefix: str, metrics: dict):
        """
        Helper to log metrics with a prefix to the initialized ml_logger.
        """
        if self.ml_logger:
            self.ml_logger.log_metrics({f"{prefix}/{k}": v for k, v in metrics.items()})

    def _update_wandb_table(self, iteration: int, rl_sample: Optional[RLTrajectory]):
        """
        Add a row to the persistent WandB samples table.
        """
        if self.ml_logger and rl_sample and self._samples_table is not None:
            import wandb

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
    ):
        """
        Log a representative sample from SFT and RL phases using Rich for a better UX.
        """
        from rich.console import Console, Group
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.rule import Rule

        console = Console()

        def create_sample_panel(sample: RLTrajectory, title: str, color: str):
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
                Markdown(
                    f"### Execution Output\n```text\n{sample.execution_output}\n```"
                ),
                Rule(style="dim"),
                Markdown(
                    f"### Verification Judgment\n> {sample.verification_reasoning}"
                ),
            )

            return Panel(
                content,
                title=f"{title} - {success_tag}",
                border_style=color,
                expand=False,
            )

        console.print()
        console.print(
            Rule(f"[bold cyan]Iteration {iteration} Summary[/bold cyan]", style="cyan")
        )
        console.print()

        if sft_sample:
            console.print(
                create_sample_panel(
                    sft_sample, "Phase 1: Teacher (SFT) Sample", "magenta"
                )
            )
            console.print()

        if rl_sample:
            console.print(
                create_sample_panel(
                    rl_sample,
                    "Phase 2: Student (RL) Sample",
                    "green" if rl_sample.success else "yellow",
                )
            )
            console.print()

        console.print(Rule(style="cyan"))
        console.print()
