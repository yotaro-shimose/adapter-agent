import asyncio
import logging
from typing import Optional

import tinker
from oai_utils.async_utils import gather_with_semaphore
from prisma import Prisma
from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .executor import InternalizeExecutor
from .types import EvalResult, PipelineConfig, SeedSuite

logger = logging.getLogger(__name__)


class EvaluateWorker:
    def __init__(
        self,
        config: PipelineConfig,
        executor: InternalizeExecutor,
        ml_logger: MLLogger,
        prisma_client: Prisma,
        shared_sampling_client: SharedSamplingClient,
        renderer: Renderer,
        eval_suites: list[SeedSuite],
        trigger: asyncio.Event,
    ):
        self.config = config
        self.executor = executor
        self.ml_logger = ml_logger
        self.prisma_client = prisma_client
        self.shared_sampling_client = shared_sampling_client
        self.renderer = renderer
        self.eval_suites = eval_suites
        self.trigger = trigger
        self.system_prompt = self._get_solver_system_prompt(config.library_name)

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

    async def run_loop(self):
        logger.info("EvaluateWorker loop started.")
        while True:
            try:
                await self.trigger.wait()
                self.trigger.clear()

                # Snapshot the current model version
                snapshot: IndexedSamplingClient = (
                    self.shared_sampling_client.get_client()
                )
                version = snapshot.version
                logger.info(f"Starting evaluation cycle for model version {version}...")

                await self._run_evaluation(snapshot, version)
                logger.info(f"Evaluation cycle for version {version} completed.")

            except asyncio.CancelledError:
                logger.info("EvaluateWorker loop cancelled.")
                break
            except Exception as e:
                logger.exception(f"EvaluateWorker encountered error: {e}")
                await asyncio.sleep(5)

    async def _run_evaluation(
        self, snapshot: IndexedSamplingClient, version: int
    ) -> None:
        if not self.eval_suites:
            return

        flattened: list[tuple[str, str]] = [
            (s.name, t.instruction) for s in self.eval_suites for t in s.tasks
        ]
        logger.info(
            f"Running evaluation for {len(flattened)} total tasks across {len(self.eval_suites)} suites..."
        )

        async def _eval_one(suite_name: str, instruction: str) -> tuple[str, EvalResult]:
            res = await self._evaluate_single_task(
                snapshot, version, instruction, source=suite_name
            )
            return suite_name, res

        results = await gather_with_semaphore(
            [_eval_one(sn, instr) for sn, instr in flattened],
            max_concurrent=self.config.eval_concurrency,
        )

        suite_stats: dict[str, dict[str, int]] = {
            s.name: {"success": 0, "rollouts": 0} for s in self.eval_suites
        }
        for suite_name, res in results:
            suite_stats[suite_name]["success"] += res.success_count
            suite_stats[suite_name]["rollouts"] += res.total_count

        metrics_to_log: dict[str, float] = {}
        for suite_name, stats in suite_stats.items():
            success = stats["success"]
            rollouts = stats["rollouts"]
            success_ratio = success / rollouts if rollouts > 0 else 0.0
            metrics_to_log[f"eval_{suite_name}/success_ratio"] = success_ratio
            metrics_to_log[f"eval_{suite_name}/total_success"] = float(success)
            metrics_to_log[f"eval_{suite_name}/total_rollouts"] = float(rollouts)

        if metrics_to_log:
            self.ml_logger.log_metrics(metrics_to_log)

    async def _evaluate_single_task(
        self,
        snapshot: IndexedSamplingClient,
        version: int,
        instruction: str,
        source: str,
        rollouts: Optional[int] = None,
    ) -> EvalResult:
        model_input = self.renderer.build_generation_prompt(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=instruction),
            ]
        )

        num_samples = rollouts if rollouts is not None else self.config.eval_rollout
        sample_results = await snapshot.client.sample_async(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(include_logprobs=True),
        )

        success_count = 0
        for seq in sample_results.sequences:
            msg, ok = self.renderer.parse_response(seq.tokens)
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

            outcome = await self.executor.run_execution_and_verification(
                instruction, reasoning, answer_text
            )
            is_success = outcome.success
            if is_success:
                success_count += 1

            # Log to Prisma (source suite name persisted as knowledge_id/title for dashboard continuity)
            try:
                await self.prisma_client.simpletrajectory.create(
                    data={
                        "simple_train_id": self.config.simple_train_id,
                        "knowledge_id": source,
                        "knowledge_title": source,
                        "step": version,
                        "question": instruction,
                        "reasoning": reasoning,
                        "answer": answer_text,
                        "success": is_success,
                        "execution_output": outcome.execution_output,
                        "verification_output": outcome.verification_output,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to record evaluation trajectory: {e}")

        return EvalResult(success_count=success_count, total_count=num_samples)
