import asyncio
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import tinker
from dotenv import load_dotenv
from more_itertools import chunked
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from prisma import Prisma
from prisma.models import OpenBookQA, OpenBookTrajectory
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from tinker_cookbook.renderers import (
    Message,
    TextPart,
    ThinkingPart,
    TrainOnWhat,
)
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import TinkerMessage
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.rewirer import log_trajectory
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.async_rust_doc_analyzer import (
    AsyncRustDocAnalyzer,
)
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.util.logger_util import ClockCycleFilteredLogger

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class SamplerConfig:
    experiment_name: str = "test-openbook-1"
    granular_id: str = "granular_prep_20260414_085004"
    limit: int = 10
    model_name: str = "Qwen/Qwen3-8B"
    rust_doc_json: Path = Path("repositories/numrs/target/doc/numrs2.json")
    max_parallel_runtimes: int = 400
    # SFT/Eval settings
    sft_epochs: int = 20
    sft_batch_size: int = 256
    learning_rate: float = 1e-3
    eval_every: int = 5  # Evaluation every N optimizer steps
    train_holdout_split: float = 0.833  # 5:1 ratio split
    lora_rank: int = 32
    # Scaling settings
    k_sft: int = 8
    k_eval: int = 2
    max_retries: int = 3
    use_bridge: bool = True  # If True, force Logical Bridge protocol
    qa_cache_experiment: str | None = None  # Experiment name to reuse QAs from
    include_reasoning_in_sft: bool = (
        True  # If False, reasoning is removed from training data
    )


@dataclass(kw_only=True)
class OpenBookPipeline:
    config: SamplerConfig
    db: Prisma
    analyzer: AsyncRustDocAnalyzer
    executor: InternalizeExecutor
    generator: GeneratorAgent
    tinker_model: TinkerModel
    training_client: tinker.TrainingClient
    ml_logger: MLLogger

    @classmethod
    async def create(cls, config: SamplerConfig) -> "OpenBookPipeline":
        """Initialize all resources and returns a fully constructed sampler."""
        load_dotenv()
        db = Prisma()
        await db.connect()

        if not config.rust_doc_json.exists():
            raise FileNotFoundError(f"RustDoc JSON not found at {config.rust_doc_json}")

        analyzer = await AsyncRustDocAnalyzer.create_from_json(config.rust_doc_json)
        gemini = get_gemini()
        generator = GeneratorAgent(model=gemini, rust_doc_analyzer=analyzer)
        verifier = Verifier(model=gemini, rust_doc_analyzer=analyzer)

        runtime_settings = RuntimeSettings.cloudrun_numrs2()
        runtime_pool = RuntimePool(
            runtime_settings, max_size=config.max_parallel_runtimes
        )
        executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)

        service_client = tinker.ServiceClient()
        tinker_model, _, _ = setup_tinkermodel(
            model_name=config.model_name,
            service_client=service_client,
            path=None,
        )

        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )

        log_dir = Path("logs") / "openbook_rejection" / config.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ClockCycleFilteredLogger(
            ml_log.setup_logging(
                log_dir=str(log_dir),
                wandb_project="internalization",
                config=asdict(config),
            )
        )

        await db.openbookexperiment.upsert(
            where={"experiment_name": config.experiment_name},
            data={
                "create": {"experiment_name": config.experiment_name},
                "update": {},
            },
        )

        return cls(
            config=config,
            db=db,
            analyzer=analyzer,
            executor=executor,
            generator=generator,
            tinker_model=tinker_model,
            training_client=training_client,
            ml_logger=ml_logger,
        )

    async def prepare_qa_pairs(self) -> list[OpenBookQA]:
        """Fetch granular knowledge and ensure verified QA pairs exist for each."""
        # Global check: If ANY QAs exist for this experiment, don't generate more
        existing_qas = await self.db.openbookqa.find_many(
            where={"experiment_name": self.config.experiment_name}
        )
        if existing_qas:
            logger.info(
                f"Resuming experiment '{self.config.experiment_name}': {len(existing_qas)} QAs found. Skipping QA replenishment."
            )
            return existing_qas

        granulars = await self.db.granularknowledge.find_many(
            where={"simple_train_id": self.config.granular_id}, take=self.config.limit
        )
        logger.info(f"Loaded {len(granulars)} granular knowledge items.")

        with self._create_progress() as progress:
            total_target = len(granulars) * (self.config.k_sft + self.config.k_eval)
            task = progress.add_task("Generating/Verifying QAs...", total=total_target)
            results = await asyncio.gather(
                *[self._process_single_knowledge(g, progress, task) for g in granulars]
            )
            # Flatten the list of lists
            all_qas = [qa for knowledge_results in results for qa in knowledge_results]
            return all_qas

    async def prepare_datasets(
        self,
    ) -> tuple[list[OpenBookTrajectory], list[OpenBookTrajectory]]:
        """Fetch all successful trajectories and split into Train and Holdout sets."""
        trajectories = await self.db.openbooktrajectory.find_many(
            where={
                "experiment_name": self.config.experiment_name,
                "success": True,
            }
        )

        # Re-fetch QAs to get knowledge_id mapping
        qas = await self.db.openbookqa.find_many(
            where={"experiment_name": self.config.experiment_name}
        )
        qa_to_kn = {qa.id: qa.knowledge_id for qa in qas}

        from collections import defaultdict

        kn_groups = defaultdict(list)
        for t in trajectories:
            kn_id = qa_to_kn.get(t.qa_id)
            if kn_id:
                kn_groups[kn_id].append(t)

        train_data = []
        holdout_data = []

        import random

        for kn_id, items in kn_groups.items():
            random.shuffle(items)
            # Prioritize evaluation data (up to k_eval)
            eval_items = items[: self.config.k_eval]
            train_items = items[self.config.k_eval :]

            # PERSIST TAGS TO DB
            for t in eval_items:
                await self.db.openbooktrajectory.update(
                    where={"id": t.id}, data={"dataset": "holdout"}
                )
            for t in train_items:
                await self.db.openbooktrajectory.update(
                    where={"id": t.id}, data={"dataset": "train"}
                )

            holdout_data.extend(eval_items)
            train_data.extend(train_items)

        logger.info(
            f"Datasets prepared: {len(train_data)} train, {len(holdout_data)} holdout samples from {len(kn_groups)} knowledge items."
        )
        return train_data, holdout_data

    async def run_sft_and_eval(
        self,
        train_data: list[OpenBookTrajectory],
        holdout_data: list[OpenBookTrajectory],
    ):
        """Run the SFT training loop with periodic evaluation."""
        logger.info(
            f"Starting SFT and Evaluation loop for {self.config.sft_epochs} epochs."
        )
        datums = self._create_sft_datums(train_data)
        if not datums:
            logger.warning("No SFT datums created. Skipping training.")
            return

        batches = list(chunked(datums, self.config.sft_batch_size))
        total_steps = len(batches) * self.config.sft_epochs
        adam_params = tinker.AdamParams(learning_rate=self.config.learning_rate)

        # Initial evaluation
        logger.info("Initial evaluation: Triggering periodic evaluation...")
        await self.evaluate_closed_book(train_data[:20], holdout_data)

        # 1. Enqueue the very first batch
        fwd_bwd_future = await self.training_client.forward_backward_async(
            data=batches[0], loss_fn="cross_entropy"
        )
        optim_future = await self.training_client.optim_step_async(adam_params)

        step_counter = 0
        for epoch in range(self.config.sft_epochs):
            for step_in_epoch, _ in enumerate(batches):
                step_counter += 1

                # Enqueue NEXT step
                next_step = step_counter
                if next_step < total_steps:
                    next_batch_idx = next_step % len(batches)
                    next_fwd_bwd_future = (
                        await self.training_client.forward_backward_async(
                            data=batches[next_batch_idx], loss_fn="cross_entropy"
                        )
                    )
                    next_optim_future = await self.training_client.optim_step_async(
                        adam_params
                    )
                else:
                    next_fwd_bwd_future = None
                    next_optim_future = None

                # Await CURRENT results
                if fwd_bwd_future is not None and optim_future is not None:
                    fwd_bwd_res = await fwd_bwd_future.result_async()
                    await optim_future.result_async()

                    # Log metrics to ML Logger
                    self.ml_logger.log_metrics(
                        {f"sft/{k}": v for k, v in fwd_bwd_res.metrics.items()}
                    )

                    # Check for evaluation trigger (every N steps)
                    if step_counter % self.config.eval_every == 0:
                        # Sync weights before evaluation
                        logger.info(
                            f"Step {step_counter}: Triggering periodic evaluation..."
                        )
                        new_client = await self.training_client.save_weights_and_get_sampling_client_async()
                        self.tinker_model.sampling_client = new_client
                        await self.evaluate_closed_book(train_data[:20], holdout_data)

                fwd_bwd_future = next_fwd_bwd_future
                optim_future = next_optim_future

        logger.info("SFT training phase completed.")

    def _create_sft_datums(
        self, trajectories: list[OpenBookTrajectory]
    ) -> list[tinker.Datum]:
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_name
        )
        datums = []
        for t in trajectories:
            user_content = self._get_user_prompt(t.question, t.hint)

            if self.config.include_reasoning_in_sft:
                assistant_msg = Message(
                    role="assistant",
                    content=[
                        ThinkingPart(type="thinking", thinking=t.reasoning),
                        TextPart(type="text", text=f"\n\n{t.answer}"),
                    ],
                )
            else:
                assistant_msg = Message(role="assistant", content=t.answer)

            datum = conversation_to_datum(
                conversation=[
                    Message(role="user", content=user_content),
                    assistant_msg,
                ],
                renderer=renderer,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_length=None,
            )
            datums.append(datum)
        return datums

    async def evaluate_closed_book(
        self,
        train_subset: list[OpenBookTrajectory],
        holdout_data: list[OpenBookTrajectory],
    ):
        """Evaluate performance on both sets without hints in parallel."""
        logger.info("Evaluating Closed-Book performance...")

        with self._create_progress() as progress:
            results = await asyncio.gather(
                self._evaluate_set(train_subset, "Train (Subsample)", progress),
                self._evaluate_set(holdout_data, "Holdout", progress),
            )
            train_success, holdout_success = results

        self.ml_logger.log_metrics(
            {
                "eval/train_success_rate": train_success,
                "eval/holdout_success_rate": holdout_success,
            }
        )

    async def _evaluate_set(
        self, dataset: list[OpenBookTrajectory], name: str, progress: Progress
    ) -> float:
        if not dataset:
            return 0.0

        import random

        task = progress.add_task(f"Evaluating {name}...", total=len(dataset))

        async def _eval_one(item):
            try:
                # CLOSED-BOOK: No hint provided
                user_prompt = f"Question: {item.question}"
                messages = [
                    TinkerMessage(
                        role="system", content=self._get_closed_book_system_prompt()
                    ),
                    TinkerMessage(role="user", content=user_prompt),
                ]
                model_input = self.tinker_model.renderer.build_generation_prompt(
                    messages
                )

                sampling_res = await self.tinker_model.sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(),
                )

                if not sampling_res.sequences:
                    return False

                msg, ok = self.tinker_model.renderer.parse_response(
                    sampling_res.sequences[0].tokens
                )
                if not ok:
                    return False

                reasoning, answer_text = self._extract_parts(msg)
                outcome = await self.executor.run_execution_and_verification(
                    item.question, reasoning, answer_text
                )

                # Log some trajectories for visual check
                if random.random() < 0.2:  # Log 20% of eval trajectories
                    log_trajectory(
                        [
                            *messages,
                            msg,
                            TinkerMessage(
                                role="tool", content=outcome.execution_output
                            ),
                        ]
                    )

                return outcome.success
            finally:
                progress.advance(task)

        results = await asyncio.gather(*[_eval_one(item) for item in dataset])
        success_count = sum(1 for r in results if r)

        return success_count / len(dataset)

    async def sample_and_reject(self, qas: list[OpenBookQA]):
        """Perform open-book sampling for the given QA pairs."""
        logger.info(f"Ready to sample {len(qas)} QAs.")

        with self._create_progress() as progress:
            task = progress.add_task("Sampling trajectories...", total=len(qas))
            await asyncio.gather(
                *[self._sample_and_verify_single_qa(qa, progress, task) for qa in qas]
            )

    async def close(self):
        """Gracefully disconnect and cleanup."""
        await self.db.disconnect()
        if self.analyzer:
            await self.analyzer.__aexit__(None, None, None)

    # --- Internal Helpers ---

    async def _process_single_knowledge(
        self, g, progress: Progress, task_id: TaskID
    ) -> list[OpenBookQA]:
        target_count = self.config.k_sft + self.config.k_eval
        kn = Knowledge(id=g.id, title=g.title, content=g.content)

        # 1. Fetch QAs already in the current experiment
        final_qas = await self.db.openbookqa.find_many(
            where={
                "experiment_name": self.config.experiment_name,
                "knowledge_id": g.id,
            }
        )
        existing_questions = {qa.question for qa in final_qas}

        # 2. If needed, fetch and replicate QAs from the cache experiment
        if len(final_qas) < target_count and self.config.qa_cache_experiment:
            if self.config.qa_cache_experiment != self.config.experiment_name:
                cache_qas = await self.db.openbookqa.find_many(
                    where={
                        "experiment_name": self.config.qa_cache_experiment,
                        "knowledge_id": g.id,
                    }
                )
                for qa in cache_qas:
                    if len(final_qas) >= target_count:
                        break
                    if qa.question not in existing_questions:
                        # Replicate to the current experiment for data isolation
                        new_qa = await self.db.openbookqa.create(
                            data={
                                "experiment_name": self.config.experiment_name,
                                "knowledge_id": g.id,
                                "title": g.title,
                                "question": qa.question,
                                "answer": qa.answer,
                            }
                        )
                        final_qas.append(new_qa)
                        existing_questions.add(new_qa.question)

        # 3. Generate only if we have nothing at all
        if not final_qas:
            needed = target_count

            async def _attempt_single_qa():
                for attempt in range(self.config.max_retries + 1):
                    qa = await self.generator.generate_qa(kn)
                    if qa and qa.question not in existing_questions:
                        outcome = await self.executor.run_execution_and_verification(
                            qa.question, "", qa.answer
                        )
                        if outcome.success:
                            return await self.db.openbookqa.create(
                                data={
                                    "experiment_name": self.config.experiment_name,
                                    "knowledge_id": g.id,
                                    "title": g.title,
                                    "question": qa.question,
                                    "answer": qa.answer,
                                }
                            )
                    if attempt < self.config.max_retries:
                        logger.info(
                            f"Retrying QA generation for '{g.title}' (attempt {attempt + 1})"
                        )
                return None

            tasks = [_attempt_single_qa() for _ in range(needed)]
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            final_qas.extend(valid_results)

        # Advance progress for all slots, even if some generation tasks ultimately failed
        progress.advance(task_id, advance=target_count)
        return final_qas[:target_count]

    async def _sample_and_verify_single_qa(
        self, qa: OpenBookQA, progress: Progress, task_id: TaskID
    ):
        try:
            user_prompt = self._get_user_prompt(qa.question, qa.answer)
            messages = [
                TinkerMessage(
                    role="system", content=self._get_open_book_system_prompt()
                ),
                TinkerMessage(role="user", content=user_prompt),
            ]
            model_input = self.tinker_model.renderer.build_generation_prompt(messages)

            sampling_res = await self.tinker_model.sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(),
            )

            if not sampling_res.sequences:
                return

            msg, ok = self.tinker_model.renderer.parse_response(
                sampling_res.sequences[0].tokens
            )
            if not ok:
                return

            reasoning, answer_text = self._extract_parts(msg)
            outcome = await self.executor.run_execution_and_verification(
                qa.question, reasoning, answer_text
            )

            log_trajectory(
                [
                    *messages,
                    msg,
                    TinkerMessage(role="tool", content=outcome.execution_output),
                ]
            )

            await self.db.openbooktrajectory.create(
                data={
                    "experiment_name": self.config.experiment_name,
                    "qa_id": qa.id,
                    "question": qa.question,
                    "hint": qa.answer,
                    "reasoning": reasoning,
                    "answer": answer_text,
                    "success": outcome.success,
                    "execution_output": outcome.execution_output,
                    "verification_output": outcome.verification_output,
                }
            )
        finally:
            progress.advance(task_id)

    def _get_user_prompt(self, question: str, hint: str) -> str:
        return f"Question: {question}\n\nHint (Reference Answer):\n{hint}"

    def _extract_parts(self, msg: TinkerMessage) -> tuple[str, str]:
        reasoning = ""
        answer_text = ""
        content = msg.get("content")
        if content:
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "thinking":
                        reasoning += part["thinking"]
                    elif part["type"] == "text":
                        answer_text += part["text"]
            else:
                answer_text = str(content)
        return reasoning, answer_text

    def _get_open_book_system_prompt(self) -> str:
        if self.config.use_bridge:
            return """\
You are an expert software engineer.
You are given a programming Question and a Hint (the reference Answer).
Your task is to reach the exact solution by deeply internalizing the provided knowledge.

Final answer must include:
1. Generalization & Pattern Deduction:
   - Analyze the hint and extract knowledges that you should remember to solve similar problems in the future such as API usages, best practices or anti-patterns.
   - When you create a knowledge, you are encouraged to imagine several situations you use this knowledge.
   - To encourage recalling the knowledge, for each situation, I want you to start from situation and then derive the knowledge.
   - For example, you should first locate an API (say `random.random()` in python). Then you imagine a situation "I come up with a situation to use this knowledge. Suppose we are to estimate the value of pi using Monte-Carlo simulation. We can use `random.random()` to generate random numbers between 0 and 1 to..."
2. Answer to the question:
    - After thinking about the knowledge, you come back to the question. You start from re-stating the question in your own words.
    - You think about how your knowledge helps to solve this problem.
    - Then you write your final code solution wrapped in a single, fully executable ```rust ... ``` block (must be a complete, runnable program, not a snippet).

"""
        else:
            return """\
You are an expert software engineer.
You are given a programming Question and a Hint (the reference Answer).
Your task is to reach the exact solution. You should use the provided hint as a reference to help you solve the problem correctly.

Final answer must include your final code solution wrapped in a single, fully executable ```rust ... ``` block (must be a complete, runnable program, not a snippet).
"""

    def _get_closed_book_system_prompt(self) -> str:
        return """\
You are an expert software engineer.
You are given a programming Question.
Your task is to reach the exact solution by thinking through the problem step-by-step.

Final answer must include:
1. Re-statement of the problem in your own words.
2. Your step-by-step reasoning process.
3. Your final code solution wrapped in a single, fully executable ```rust ... ``` block (must be a complete, runnable program, not a snippet).
"""

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=Console(),
        )


async def main():
    config = SamplerConfig(experiment_name="with_bridge", use_bridge=True)
    sampler = await OpenBookPipeline.create(config)
    try:
        # Phase 1: Rejection Sampling (Open-Book with Hint)
        qas = await sampler.prepare_qa_pairs()
        if qas:
            logger.info("Starting Phase 1: Open-Book Rejection Sampling...")
            await sampler.sample_and_reject(qas)
        else:
            logger.warning("No QAs available for sampling.")

        # Phase 2: SFT and Evaluation (Closed-Book without Hint)
        logger.info("Starting Phase 2: SFT and Closed-Book Evaluation...")
        train_data, holdout_data = await sampler.prepare_datasets()
        if train_data:
            await sampler.run_sft_and_eval(train_data, holdout_data)
        else:
            logger.warning("No successful trajectories found for SFT.")

    finally:
        await sampler.close()


if __name__ == "__main__":
    asyncio.run(main())
