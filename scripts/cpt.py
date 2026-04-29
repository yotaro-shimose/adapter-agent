import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import tinker
from dotenv import load_dotenv
from more_itertools import chunked
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from pydantic import BaseModel
from tinker import Datum, SampledSequence
from tinker_cookbook.renderers import Message, ToolCall, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.util.logger_util import (
    ClockCycleFilteredLogger,
    setup_base_loglevel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    success: bool
    message: Message | None
    execution_output: str = ""
    verification_output: str = ""


@dataclass
class PreparedCPTTasks:
    cpt_datums: list[Datum]
    eval_tasks: list[tuple[Knowledge, QRA]]


class CPTConfig(BaseModel):
    """Configuration for the Continual Pretraining (CPT) pipeline."""

    version: str = "lab_verification"
    limit: int | None = None
    granular_id: str | None = None
    model_name: str = "Qwen/Qwen3-8B"
    batch_size: int = 256
    epochs: int = 3
    num_cycles: int = 5  # Number of [SFT -> Eval] blocks to run
    learning_rate: float = 1e-3
    lora_rank: int = 32
    max_length: int = 2048
    rust_doc_path: Path = Path("repositories/numrs/target/doc/numrs2.json")
    wandb_project: str | None = "cpt_wiki"
    eval_rollout: int = 8
    runtime_pool_size: int = 400
    system_prompt: str = "You are an expert Rust developer using the numrs2 library. Solve the user's task by writing a complete Rust program. Your answer MUST include the code enclosed in a ```rust ... ``` code block. The code must be correct, idiomatic Rust, and solve the problem completely."


class CPTPipeline:
    """Encapsulates the CPT training workflow with cyclic evaluation."""

    def __init__(
        self,
        config: CPTConfig,
        db: Prisma,
        service_client: tinker.ServiceClient,
        training_client: tinker.TrainingClient,
        solver_model: TinkerModel,
        generator: GeneratorAgent,
        executor: InternalizeExecutor,
        runtime_pool: RuntimePool,
        ml_logger: MLLogger,
        log_dir: Path,
    ):
        self.config = config
        self.db = db
        self.service_client = service_client
        self.training_client = training_client
        self.sampling_client = solver_model.sampling_client
        self.solver_model = solver_model
        self.generator = generator
        self.executor = executor
        self.runtime_pool = runtime_pool
        self.ml_logger = ml_logger
        self.log_dir = log_dir

    @classmethod
    async def create(cls, config: CPTConfig) -> Self:
        """Factory method for CPTPipeline initialization."""
        db = Prisma()
        await db.connect()

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )

        solver_model, _, _ = setup_tinkermodel(
            model_name=config.model_name,
            service_client=service_client,
        )

        gemini = get_gemini()
        analyzer = await AsyncRustDocAnalyzer.create_from_json(config.rust_doc_path)
        generator = GeneratorAgent(model=gemini, rust_doc_analyzer=analyzer)
        verifier = Verifier(model=gemini, rust_doc_analyzer=analyzer)

        runtime_settings = RuntimeSettings.cloudrun_numrs2()
        runtime_pool = RuntimePool(runtime_settings, max_size=config.runtime_pool_size)
        executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)

        log_dir = (
            Path("logs/cpt")
            / f"{config.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ClockCycleFilteredLogger(
            ml_log.setup_logging(
                log_dir=str(log_dir),
                wandb_project=config.wandb_project,
                config=config.model_dump(),
            )
        )

        return cls(
            config=config,
            db=db,
            service_client=service_client,
            training_client=training_client,
            solver_model=solver_model,
            generator=generator,
            executor=executor,
            runtime_pool=runtime_pool,
            ml_logger=ml_logger,
            log_dir=log_dir,
        )

    async def load_knowledge(self) -> list[Knowledge]:
        """Loads knowledge articles from Wiki or Granular DB in parallel."""
        knowledge_list = []

        if self.config.granular_id:
            logger.info(
                f"Loading granular knowledge from run ID: {self.config.granular_id}"
            )
            granulars = await self.db.granularknowledge.find_many(
                where={"simple_train_id": self.config.granular_id}
            )
            for g in granulars:
                knowledge_list.append(
                    Knowledge(id=g.id, title=g.title, content=g.content)
                )
        else:
            wiki_manager = WikiManager(self.db, version=self.config.version)
            titles = await wiki_manager.ls(path="api/")
            if self.config.limit:
                titles = titles[: self.config.limit]

            logger.info(
                f"Loading {len(titles)} articles from Wiki version {self.config.version}..."
            )
            tasks = [wiki_manager.read(title) for title in titles]
            contents = await asyncio.gather(*tasks)

            knowledge_list = [
                Knowledge(title=title, content=content)
                for title, content in zip(titles, contents)
                if content
            ]

        logger.info(f"Successfully loaded {len(knowledge_list)} knowledge items.")
        return knowledge_list

    def _create_cpt_datum(self, k: Knowledge, log: bool = False) -> Datum:
        """Constructs a single CPT training datum."""
        tool_call_id = f"call_{k.id or k.title[:8]}"
        conversation = [
            Message(
                role="user",
                content=f"Please provide information related to {k.title}.",
                trainable=False,
            ),
            Message(
                role="assistant",
                content=f"<wiki_read>{k.title}</wiki_read>",
                trainable=False,
                tool_calls=[
                    ToolCall(
                        id=tool_call_id,
                        function=ToolCall.FunctionBody(
                            name="wiki_read", arguments=json.dumps({"title": k.title})
                        ),
                    )
                ],
            ),
            Message(
                role="tool",
                content=k.content,
                trainable=True,
                tool_call_id=tool_call_id,
                name="wiki_read",
            ),
        ]
        if log:
            log_trajectory(conversation)
        return conversation_to_datum(
            conversation=conversation,
            renderer=self.solver_model.renderer,
            train_on_what=TrainOnWhat.CUSTOMIZED,
            max_length=self.config.max_length,
        )

    async def prepare_tasks(self, knowledge_list: list[Knowledge]) -> PreparedCPTTasks:
        """Prepares CPT datums and Evaluation Tasks in parallel."""
        logger.info(f"Preparing tasks for {len(knowledge_list)} knowledge items...")

        async def _prep_single(k: Knowledge):
            datum = self._create_cpt_datum(k, log=True)
            qra = await self.generator.generate_sft(k)
            return datum, (k, qra) if qra else None

        results = await asyncio.gather(*[_prep_single(k) for k in knowledge_list])
        cpt_datums = [r[0] for r in results]
        eval_tasks = [r[1] for r in results if r[1] is not None]

        logger.info(
            f"Task preparation complete: {len(cpt_datums)} CPT datums, {len(eval_tasks)} eval tasks."
        )
        return PreparedCPTTasks(cpt_datums=cpt_datums, eval_tasks=eval_tasks)

    def _extract_reasoning_and_answer(self, message: Message) -> tuple[str, str]:
        """Extracts thinking/reasoning and text/answer from a Renderer message."""
        content = message.get("content", "")
        reasoning, answer_text = "", ""

        if isinstance(content, list):
            for part in content:
                if part.get("type") == "thinking":
                    reasoning += str(part.get("thinking", ""))
                elif part.get("type") == "text":
                    answer_text += str(part.get("text", ""))
        else:
            answer_text = str(content)

        return reasoning.strip(), answer_text.strip()

    async def _evaluate_single_task(
        self,
        task: tuple[Knowledge, QRA],
    ) -> tuple[bool, int]:
        """Evaluates a single task with multi-sample verification in parallel."""
        k, qra = task
        tool_input_messages = [
            Message(role="system", content=self.config.system_prompt),
            Message(role="user", content=qra.question),
        ]
        prompt = self.solver_model.renderer.build_generation_prompt(tool_input_messages)

        rollouts = await self.sampling_client.sample_async(
            prompt=prompt,
            num_samples=self.config.eval_rollout,
            sampling_params=tinker.SamplingParams(),
        )

        async def _verify_sequence(seq: SampledSequence) -> RolloutResult:
            message, ok = self.solver_model.renderer.parse_response(seq.tokens)
            if not ok:
                return RolloutResult(success=False, message=None)

            reasoning, answer_text = self._extract_reasoning_and_answer(message)
            outcome = await self.executor.run_execution_and_verification(
                qra.question, reasoning, answer_text
            )
            return RolloutResult(
                success=outcome.success,
                message=message,
                execution_output=outcome.execution_output,
                verification_output=outcome.verification_output,
            )

        # Verify all rollouts for this task in parallel
        verification_tasks = [_verify_sequence(seq) for seq in rollouts.sequences]
        verification_results: list[RolloutResult] = await asyncio.gather(
            *verification_tasks
        )

        any_success = any(res.success for res in verification_results)
        success_count = sum(1 for res in verification_results if res.success)
        self._log_evaluation_result(k, tool_input_messages, verification_results)

        return any_success, success_count

    def _log_evaluation_result(
        self,
        knowledge: Knowledge,
        tool_input_messages: list[Message],
        verification_results: list[RolloutResult],
    ) -> None:
        """Logs the final task status and visualizes the representative trajectory."""
        success_idx = next(
            (i for i, res in enumerate(verification_results) if res.success), -1
        )
        any_success = success_idx != -1

        if any_success:
            logger.info(
                f"  Task '{knowledge.title}': ✅ SUCCESS (at rollout {success_idx})"
            )
            res = verification_results[success_idx]
            msg = res.message
            if msg:
                log_trajectory(
                    tool_input_messages
                    + [
                        msg,
                        Message(
                            role="tool",
                            content=f"Execution Output:\n{res.execution_output}\n\nVerification Output:\n{res.verification_output}",
                            name="verifier",
                        ),
                    ]
                )
        else:
            logger.info(f"  Task '{knowledge.title}': ❌ FAILED all rollouts")
            # Log the baseline (first) trajectory if it exists
            if verification_results and (msg := verification_results[0].message):
                res = verification_results[0]
                log_trajectory(
                    tool_input_messages
                    + [
                        msg,
                        Message(
                            role="tool",
                            content=f"Execution Output:\n{res.execution_output}\n\nVerification Output:\n{res.verification_output}",
                            name="verifier",
                        ),
                    ]
                )

    async def evaluate(self, eval_tasks: list[tuple[Knowledge, QRA]]) -> None:
        """Evaluates the trained model in parallel across tasks and samples."""
        logger.info(f"Starting parallel evaluation on {len(eval_tasks)} tasks...")
        # Synchronize weights and update sampling client fields
        tasks = [self._evaluate_single_task(task) for task in eval_tasks]
        results = await asyncio.gather(*tasks)

        success_at_least_one = sum(1 for r in results if r[0])
        total_success_samples = sum(r[1] for r in results)
        total_samples = len(eval_tasks) * self.config.eval_rollout

        pass_at_n = success_at_least_one / len(eval_tasks) if eval_tasks else 0
        sample_success_rate = (
            total_success_samples / total_samples if total_samples else 0
        )

        logger.info(
            f"Evaluation Complete.\n"
            f"  Pass@{self.config.eval_rollout}: {pass_at_n:.2%}\n"
            f"  Sample Success Rate: {sample_success_rate:.2%}"
        )
        self.ml_logger.log_metrics(
            {
                "eval/pass_at_n": pass_at_n,
                "eval/sample_success_rate": sample_success_rate,
                "eval/num_tasks": len(eval_tasks),
                "eval/total_samples": total_samples,
            }
        )

    async def _run_sft(self, datums: list[Datum], num_epochs: int) -> None:
        """Runs SFT training for multiple epochs with look-ahead logic across the full dataset."""
        logger.info(f"--- Starting SFT Training ({num_epochs} epochs) ---")
        all_datums = datums * num_epochs
        batch_iter = list(chunked(all_datums, self.config.batch_size))
        adam_params = tinker.AdamParams(learning_rate=self.config.learning_rate)

        if not batch_iter:
            return

        # Enqueue first batch
        fwd_future = await self.training_client.forward_backward_async(
            data=batch_iter[0], loss_fn="cross_entropy"
        )
        opt_future = await self.training_client.optim_step_async(adam_params)

        for i in range(len(batch_iter)):
            # Enqueue next batch
            if i + 1 < len(batch_iter):
                next_fwd_future = await self.training_client.forward_backward_async(
                    data=batch_iter[i + 1], loss_fn="cross_entropy"
                )
                next_opt_future = await self.training_client.optim_step_async(
                    adam_params
                )
            else:
                next_fwd_future = next_opt_future = None

            # Await current batch
            if fwd_future is not None and opt_future is not None:
                fwd_res = await fwd_future.result_async()
                await opt_future.result_async()

                self.ml_logger.log_metrics(
                    {f"cpt/{k}": v for k, v in fwd_res.metrics.items()}
                )

            if i % 10 == 0:
                epoch_hint = (
                    (i * self.config.batch_size // len(datums)) + 1 if datums else 1
                )
                logger.info(
                    f"Epoch {epoch_hint} Step {i}/{len(batch_iter)} - Loss: {fwd_res.metrics.get('loss', 'N/A')}"
                )

            # Shift futures
            fwd_future, opt_future = next_fwd_future, next_opt_future
        self.sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )

    async def run(self) -> None:
        """Executes the cyclic CPT pipeline: [SFT (N Epochs) -> Evaluation] x M Cycles."""
        knowledge_list = await self.load_knowledge()
        if not knowledge_list:
            logger.warning("No knowledge items found. Exiting.")
            return

        tasks = await self.prepare_tasks(knowledge_list)

        for cycle in range(self.config.num_cycles):
            logger.info(f"--- Cycle {cycle + 1}/{self.config.num_cycles} ---")
            await self._run_sft(tasks.cpt_datums, self.config.epochs)
            logger.info(f"Cycle {cycle + 1} training complete. Running evaluation...")
            await self.evaluate(tasks.eval_tasks)

        checkpoint_path = self.log_dir / "final_state"
        logger.info(f"Saving state to {checkpoint_path}...")
        await self.training_client.save_state_async(str(checkpoint_path))

    async def close(self) -> None:
        """Cleans up resources, closing database and network sessions."""
        logger.info("Closing resources...")
        await self.runtime_pool.close_all()
        await self.db.disconnect()
        # ServiceClient manages internal thread/loop and requires explicit close
        if hasattr(self.service_client, "close"):
            self.service_client.close()
        logger.info("All resources closed.")


async def main():
    """Main entry point for the CPT config app."""
    load_dotenv()
    setup_base_loglevel()
    config = CPTConfig(
        limit=None,
        epochs=3,
        num_cycles=5,
        wandb_project=None,
        eval_rollout=8,
        version="study_20260416_133659",
        granular_id="granular_prep_20260414_085004",
    )
    pipeline = await CPTPipeline.create(config)
    try:
        await pipeline.run()
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
