import asyncio
import logging
import statistics

import tinker
from oai_utils.async_utils import gather_with_semaphore
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .rollout_engine import RolloutEngine
from .types import EvalResult, SeedSuite

logger = logging.getLogger(__name__)


class EvaluateWorker:
    def __init__(
        self,
        ml_logger: MLLogger,
        shared_sampling_client: SharedSamplingClient,
        eval_suites: list[SeedSuite],
        trigger: asyncio.Event,
        rollout_engine: RolloutEngine,
        eval_concurrency: int,
        eval_rollout: int,
        max_output_tokens: int,
    ):
        self.ml_logger = ml_logger
        self.shared_sampling_client = shared_sampling_client
        self.eval_suites = eval_suites
        self.trigger = trigger
        self.engine = rollout_engine
        self.eval_concurrency = eval_concurrency
        self.eval_rollout = eval_rollout
        self.max_output_tokens = max_output_tokens

    async def run_loop(self):
        logger.info("EvaluateWorker loop started.")
        while True:
            try:
                await self.trigger.wait()
                self.trigger.clear()
                await self.run_once()

            except asyncio.CancelledError:
                logger.info("EvaluateWorker loop cancelled.")
                break
            except Exception as e:
                logger.exception(f"EvaluateWorker encountered error: {e}")
                await asyncio.sleep(5)

    async def run_once(
        self, snapshot: IndexedSamplingClient | None = None
    ) -> None:
        """Run a single evaluation cycle synchronously (no trigger needed)."""
        if snapshot is None:
            snapshot = self.shared_sampling_client.get_client()
        version = snapshot.version
        logger.info(f"Starting evaluation cycle for model version {version}...")
        await self._run_evaluation(snapshot)
        logger.info(f"Evaluation cycle for version {version} completed.")

    async def _run_evaluation(self, snapshot: IndexedSamplingClient) -> None:
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
                snapshot, instruction
            )
            return suite_name, res

        results = await gather_with_semaphore(
            [_eval_one(sn, instr) for sn, instr in flattened],
            max_concurrent=self.eval_concurrency,
        )

        suite_stats: dict[str, dict[str, int]] = {
            s.name: {"success": 0, "rollouts": 0} for s in self.eval_suites
        }
        suite_lengths: dict[str, list[int]] = {s.name: [] for s in self.eval_suites}
        for suite_name, res in results:
            suite_stats[suite_name]["success"] += res.success_count
            suite_stats[suite_name]["rollouts"] += res.total_count
            suite_lengths[suite_name].extend(res.response_lengths)

        metrics_to_log: dict[str, float] = {}
        for suite_name, stats in suite_stats.items():
            success = stats["success"]
            rollouts = stats["rollouts"]
            success_ratio = success / rollouts if rollouts > 0 else 0.0
            metrics_to_log[f"eval/{suite_name}/success_ratio"] = success_ratio
            metrics_to_log[f"eval/{suite_name}/total_success"] = float(success)
            metrics_to_log[f"eval/{suite_name}/total_rollouts"] = float(rollouts)

            lengths = suite_lengths[suite_name]
            if lengths:
                mean_len = sum(lengths) / len(lengths)
                var_len = (
                    statistics.pvariance(lengths) if len(lengths) > 1 else 0.0
                )
                metrics_to_log[f"eval/{suite_name}/response_length_mean"] = float(
                    mean_len
                )
                metrics_to_log[f"eval/{suite_name}/response_length_variance"] = float(
                    var_len
                )
                metrics_to_log[f"eval/{suite_name}/response_length_max"] = float(
                    max(lengths)
                )

        if metrics_to_log:
            self.ml_logger.log_metrics(metrics_to_log)

    async def _evaluate_single_task(
        self,
        snapshot: IndexedSamplingClient,
        instruction: str,
        rollouts: int | None = None,
    ) -> EvalResult:
        num_samples = rollouts if rollouts is not None else self.eval_rollout
        batch = await self.engine.run(
            sampling_client=snapshot,
            instruction=instruction,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(
                include_logprobs=True,
                max_tokens=self.max_output_tokens,
            ),
        )
        success_count = sum(1 for o in batch.outcomes if o.success)
        return EvalResult(
            success_count=success_count,
            total_count=num_samples,
            response_lengths=[len(o.tokens) for o in batch.outcomes],
        )
