import asyncio
import logging
from typing import Any, Optional

import tinker
from oai_utils.async_utils import gather_with_semaphore
from prisma import Prisma
from tinker_cookbook.renderers import Message, Renderer
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .executor import InternalizeExecutor
from .types import EvalResult, PipelineConfig

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
        eval_tasks: list[tuple[Any, QRA]],  # (Knowledge, QRA)
        extra_eval_suites: dict[str, list[str]],
        trigger: asyncio.Event,
    ):
        self.config = config
        self.executor = executor
        self.ml_logger = ml_logger
        self.prisma_client = prisma_client
        self.shared_sampling_client = shared_sampling_client
        self.renderer = renderer
        self.eval_tasks = eval_tasks
        self.extra_eval_suites = extra_eval_suites
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

                await asyncio.gather(
                    self._run_evaluation(snapshot, version),
                    self._run_extra_evaluations(snapshot, version),
                )
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
        logger.info(f"Running standard evaluation on {len(self.eval_tasks)} tasks...")

        results = await gather_with_semaphore(
            [
                self._evaluate_single_task(snapshot, version, qra, k.id, k.title)
                for k, qra in self.eval_tasks
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
            }
        )

    async def _run_extra_evaluations(
        self, snapshot: IndexedSamplingClient, version: int
    ) -> None:
        if not self.extra_eval_suites:
            return

        all_eval_tasks = []
        for suite_name, instructions in self.extra_eval_suites.items():
            for instr in instructions:
                qra = QRA(question=instr, reasoning="", answer="")
                all_eval_tasks.append((suite_name, qra))

        logger.info(
            f"Running extra evaluation for {len(all_eval_tasks)} total tasks across {len(self.extra_eval_suites)} suites..."
        )

        async def _eval_with_suite(s_name: str, q: QRA) -> tuple[str, EvalResult]:
            res = await self._evaluate_single_task(
                snapshot,
                version,
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
            s: {"success": 0, "rollouts": 0} for s in self.extra_eval_suites.keys()
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
            self.ml_logger.log_metrics(metrics_to_log)

    async def _evaluate_single_task(
        self,
        snapshot: IndexedSamplingClient,
        version: int,
        qra: QRA,
        knowledge_id: str,
        knowledge_title: str,
        rollouts: Optional[int] = None,
    ) -> EvalResult:
        model_input = self.renderer.build_generation_prompt(
            [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=qra.question),
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
                qra.question, reasoning, answer_text
            )
            is_success = outcome.success
            if is_success:
                success_count += 1

            # Log to Prisma
            try:
                await self.prisma_client.simpletrajectory.create(
                    data={
                        "simple_train_id": self.config.simple_train_id,
                        "knowledge_id": knowledge_id,
                        "knowledge_title": knowledge_title,
                        "step": version,
                        "question": qra.question,
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
