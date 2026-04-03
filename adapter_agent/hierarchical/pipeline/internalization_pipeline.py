import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

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
from adapter_agent.util.parsing import extract_rust_code

logger = logging.getLogger(__name__)
# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


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


@dataclass
class RLStepResult:
    rollout_success_ratio: float
    task_success_ratio: float
    sample: RLTrajectory | None


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
        self.training_client: tinker.TrainingClient | None = None
        self.service_client: tinker.ServiceClient | None = None
        self.ml_logger: MLLogger | None = None

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

            # Phase 1: Conditional SFT (Warm up / Recovery)
            # Skip SFT if task success ratio is at least 50%
            sft_result: SFTStepResult | None = None
            if task_success_ratio < 0.5:
                sft_result = await self.execute_sft_step()
                eval_tasks = [
                    QA(question=t.question, answer=t.answer)
                    for t in sft_result.qra_list[: self.config.k_rl]
                ]
            else:
                logger.info(
                    f"Skipping SFT Step (Task Success: {task_success_ratio:.2%}). Generating fresh tasks..."
                )
                fresh_tasks = await self.generate_teacher_tasks(self.config.k_rl)
                eval_tasks = [
                    QA(question=t.question, answer=t.answer) for t in fresh_tasks
                ]

            # Phase 2: RL (Exploration / Correction)
            rl_result = await self.execute_rl_step(eval_tasks=eval_tasks)

            rollout_success_ratio = rl_result.rollout_success_ratio
            task_success_ratio = rl_result.task_success_ratio

            # Phase 3: W&B Iteration Summary
            self._log_metrics(
                "overall",
                {
                    "rollout_success_ratio": rollout_success_ratio,
                    "task_success_ratio": task_success_ratio,
                },
            )
            self._update_wandb_table(iteration, rl_result.sample)

            # Phase 4: Visualization / Logging
            self._visualize_iteration(
                iteration,
                sft_result.representative_sample if sft_result else None,
                rl_result.sample,
            )

            logger.info(
                f"Iteration {iteration} complete. "
                f"Rollout Success: {rollout_success_ratio:.2%}, "
                f"Task Success: {task_success_ratio:.2%}"
            )

    async def generate_teacher_tasks(self, count: int) -> list[QRA]:
        """
        Generate and verify high-quality QRA tasks using the teacher model.
        """
        logger.info(f"Teacher generating {count} high-quality tasks...")
        topic_hint = self.config.api_doc_path.stem

        tasks = [
            self.generator.generate_sft(topic_hint=topic_hint) for _ in range(count)
        ]
        tasks_raw: list[QRA] = await gather_with_semaphore(
            tasks, max_concurrent=self.config.concurrency
        )

        # 2. Verify
        logger.info(f"Verifying {len(tasks_raw)} teacher-generated tasks...")
        verification_tasks = [
            self._verify_with_execution(qra.question, qra) for qra in tasks_raw
        ]
        results: list[RLTrajectory] = await gather_with_semaphore(
            verification_tasks, max_concurrent=self.config.concurrency
        )

        # 3. Handle failures? For now, we return all, but verifier result in QRA is important.
        # Actually, let's filter only successful ones to ensure RL always has valid data.
        valid_tasks = [tasks_raw[i] for i, res in enumerate(results) if res.success]

        # If too few were valid, we might want to try again or just proceed with what we have.
        if len(valid_tasks) < count:
            logger.warning(
                f"Only {len(valid_tasks)}/{count} teacher tasks were valid (verified)."
            )

        return valid_tasks

    async def execute_sft_step(self) -> SFTStepResult:
        """
        Generate QRA tasks and perform supervised fine-tuning.
        """
        logger.info("Executing SFT Step...")
        assert self.training_client is not None, "TrainingClient must be initialized"

        qra_list = await self.generate_teacher_tasks(self.config.k_sft)
        teacher_success_ratio = (
            len(qra_list) / self.config.k_sft if self.config.k_sft else 0
        )
        self._log_metrics("sft", {"teacher_success_ratio": teacher_success_ratio})

        representative_sample = None
        if qra_list:
            representative_sample = await self._verify_with_execution(
                qra_list[0].question, qra_list[0]
            )
            await self._perform_sft_training(qra_list)

        return SFTStepResult(
            representative_sample=representative_sample, qra_list=qra_list
        )

    async def _perform_sft_training(self, qra_list: list[QRA]):
        logger.info(
            f"Fine-tuning student on {len(qra_list)}/{self.config.k_sft} VALID QRA samples..."
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

        await self._run_sft_training_loop(datums)

    async def _run_sft_training_loop(self, datums: list):
        params = self.config.sft_optimizer_params
        batches = list(chunked(datums, params.batch_size))

        assert self.training_client is not None
        logger.info("Initializing SFT training loop with look-ahead...")
        fwd_bwd_future = await self.training_client.forward_backward_async(
            data=batches[0], loss_fn="cross_entropy"
        )
        optim_future = await self.training_client.optim_step_async(params.adam_params)

        for epoch in range(params.num_epochs):
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

    async def execute_rl_step(self, eval_tasks: list[QA] | None = None) -> RLStepResult:
        """
        Generate evaluation tasks, rollout solutions, verify (with execution), and train GRPO if success exists.
        Returns an RLStepResult.
        """
        logger.info("Executing RL Step...")

        tasks = await self._ensure_eval_tasks(eval_tasks)
        task_results = await gather_with_semaphore(
            [self._rollout_and_verify_task_group(t) for t in tasks],
            max_concurrent=self.config.concurrency,
        )

        if self._should_perform_grpo(task_results):
            await self._perform_grpo_update(task_results)

        await self._sync_sampling_weights()

        return self._summarize_rl_results(task_results)

    async def _ensure_eval_tasks(self, eval_tasks: list[QA] | None) -> list[QA]:
        if eval_tasks is not None:
            logger.info(f"Re-using {len(eval_tasks)} tasks from SFT for RL Step.")
            return eval_tasks

        logger.info(f"Generating {self.config.k_rl} RL eval tasks...")
        modules = self.rust_doc_analyzer.get_modules()

        async def generate_rl_task(i: int) -> QA:
            topic_hint = modules[i % len(modules)] if modules else None
            return await self.generator.generate_rl(topic_hint=topic_hint)

        return await gather_with_semaphore(
            [generate_rl_task(i) for i in range(self.config.k_rl)],
            max_concurrent=self.config.concurrency,
        )

    async def _rollout_and_verify_task_group(self, qa: QA) -> TaskRolloutResult:
        assert isinstance(self.solver.model, TinkerModel)
        library_name = self.config.api_doc_path.stem
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

                async def failed_result():
                    return self._create_failed_trajectory(qa.question)

                verification_tasks.append(failed_result())
                metadata_list.append(RolloutMetadata(ac=ac, qra=None))
            else:
                qra = self._parse_qra_from_message(qa.question, msg)
                verification_tasks.append(self._verify_with_execution(qa.question, qra))
                metadata_list.append(RolloutMetadata(ac=ac, qra=qra))

        results = await asyncio.gather(*verification_tasks)
        return self._build_task_rollout_result(qa, model_input, results, metadata_list)

    def _create_failed_trajectory(self, question: str) -> RLTrajectory:
        return RLTrajectory(
            question=question,
            reasoning="Parse Error",
            answer="N/A",
            execution_output="The model output could not be parsed as a valid message.",
            verification_reasoning="Parse failure.",
            success=False,
        )

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
            if not reasoning and answer_text:
                reasoning = answer_text.split("```rust")[0].strip()
        else:
            answer_text = content
            reasoning = answer_text.split("```rust")[0].strip()

        return QRA(
            question=question,
            reasoning=reasoning,
            answer=extract_rust_code(answer_text),
        )

    def _build_task_rollout_result(
        self,
        qa: QA,
        model_input,
        results: list[RLTrajectory],
        metadata: list[RolloutMetadata],
    ) -> TaskRolloutResult:
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
            if sample is None or (res.success and not sample.success):
                sample = res

        group = (
            RLGroup(trajectories=trajectories, rewards=rewards)
            if len(set(rewards)) > 1
            else None
        )
        return TaskRolloutResult(
            num_success=sum(1 for res in results if res.success),
            num_rollouts=len(results),
            group=group,
            sample=sample,
        )

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
        total_success = sum(tr.num_success for tr in task_results)
        total_rollouts = sum(tr.num_rollouts for tr in task_results)
        task_success = sum(1 for tr in task_results if tr.num_success > 0)

        rl_sample = next(
            (tr.sample for tr in task_results if tr.sample and tr.sample.success), None
        )
        if not rl_sample and task_results:
            rl_sample = task_results[0].sample

        result = RLStepResult(
            rollout_success_ratio=total_success / total_rollouts
            if total_rollouts > 0
            else 0.0,
            task_success_ratio=task_success / len(task_results)
            if task_results
            else 0.0,
            sample=rl_sample,
        )

        self._log_metrics(
            "rl",
            {
                "rollout_success_ratio": result.rollout_success_ratio,
                "task_success_ratio": result.task_success_ratio,
            },
        )
        return result

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
    ):
        """
        Log a representative sample from SFT and RL phases using Rich for a better UX.
        """
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
