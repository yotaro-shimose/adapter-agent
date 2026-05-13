import asyncio
import logging
import statistics

import tinker
from oai_utils.async_utils import gather_with_semaphore
from prisma import Prisma
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .rollout_engine import RolloutBatch, RolloutEngine
from .rollout_persistence import persist_rl_groups, rollout_batch_to_rl_group
from .types import SeedSuite

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
        prisma_client: Prisma,
        simple_train_id: str,
    ):
        self.ml_logger = ml_logger
        self.shared_sampling_client = shared_sampling_client
        self.eval_suites = eval_suites
        self.trigger = trigger
        self.engine = rollout_engine
        self.eval_concurrency = eval_concurrency
        self.eval_rollout = eval_rollout
        self.max_output_tokens = max_output_tokens
        self.prisma_client = prisma_client
        self.simple_train_id = simple_train_id
        # Latest RL step the trainer has completed. Set by the pipeline via
        # `set_current_rl_step` immediately before firing `trigger`. Used as
        # the `rl_step` column for eval rollouts so they sort alongside the
        # train rollouts in graphvis.
        self._current_rl_step: int = 0

    def set_current_rl_step(self, rl_step: int) -> None:
        """Pipeline calls this right before `trigger.set()` so the next eval
        cycle persists rows tagged with the trainer's most recent step."""
        self._current_rl_step = rl_step

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

        # Each entry is one (suite_name, task_id, instruction) — keep all
        # three so the persistence layer can tag rows uniformly with the
        # train side.
        flattened: list[tuple[str, str, str]] = [
            (s.name, t.id, t.instruction) for s in self.eval_suites for t in s.tasks
        ]
        logger.info(
            f"Running evaluation for {len(flattened)} total tasks across {len(self.eval_suites)} suites..."
        )

        async def _eval_one(
            suite_name: str, task_id: str, instruction: str
        ) -> tuple[str, str, str, RolloutBatch]:
            batch = await self._evaluate_single_task(snapshot, instruction)
            return suite_name, task_id, instruction, batch

        results = await gather_with_semaphore(
            [_eval_one(sn, tid, instr) for sn, tid, instr in flattened],
            max_concurrent=self.eval_concurrency,
        )

        # --- Audit persistence (same path as train rollouts) ---
        groups: list[RLGroup] = [
            rollout_batch_to_rl_group(
                batch,
                suite_name=suite_name,
                task_id=task_id,
                instruction=instruction,
                sampling_client_version=snapshot.version,
            )
            for suite_name, task_id, instruction, batch in results
        ]
        if groups:
            await persist_rl_groups(
                self.prisma_client,
                simple_train_id=self.simple_train_id,
                rl_step=self._current_rl_step,
                groups=groups,
            )

        # --- Metrics aggregation ---
        # `success` / `rollouts` count individual rollouts (eval_rollout per
        # task). `tasks_any_success` / `tasks_total` count the task once if
        # ANY of its eval_rollout samples succeed — a less noisy signal of
        # whether the model can solve the task at all (pass@k).
        suite_stats: dict[str, dict[str, int]] = {
            s.name: {"success": 0, "rollouts": 0, "tasks_any_success": 0, "tasks_total": 0}
            for s in self.eval_suites
        }
        suite_lengths: dict[str, list[int]] = {s.name: [] for s in self.eval_suites}
        for suite_name, _task_id, _instr, batch in results:
            success = sum(1 for o in batch.outcomes if o.success)
            suite_stats[suite_name]["success"] += success
            suite_stats[suite_name]["rollouts"] += len(batch.outcomes)
            suite_stats[suite_name]["tasks_total"] += 1
            if success > 0:
                suite_stats[suite_name]["tasks_any_success"] += 1
            suite_lengths[suite_name].extend(len(o.tokens) for o in batch.outcomes)

        metrics_to_log: dict[str, float] = {}
        for suite_name, stats in suite_stats.items():
            success = stats["success"]
            rollouts = stats["rollouts"]
            tasks_total = stats["tasks_total"]
            tasks_any_success = stats["tasks_any_success"]
            success_ratio = success / rollouts if rollouts > 0 else 0.0
            any_success_ratio = (
                tasks_any_success / tasks_total if tasks_total > 0 else 0.0
            )
            metrics_to_log[f"eval/{suite_name}/success_ratio"] = success_ratio
            metrics_to_log[f"eval/{suite_name}/any_success_ratio"] = any_success_ratio
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
    ) -> RolloutBatch:
        num_samples = rollouts if rollouts is not None else self.eval_rollout
        return await self.engine.run(
            sampling_client=snapshot,
            instruction=instruction,
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(
                include_logprobs=True,
                max_tokens=self.max_output_tokens,
            ),
        )
