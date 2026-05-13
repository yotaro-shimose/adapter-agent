import asyncio
import logging
import random

import tinker

from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .rollout_engine import RolloutEngine
from .rollout_persistence import rollout_batch_to_rl_group
from .types import RLSource, SeedSuite

logger = logging.getLogger(__name__)


class RLWorkerPool:
    """RL rollout を並列実行するワーカープール。

    - `__aenter__` で seed_suites をタスクキューに seeding し、N ワーカーを stagger を
      挟んで起動する。
    - 各ワーカーは task を取り出して RolloutEngine を回し、RLGroup を結果キューへ push、
      同じ task を再エンキューする (100%成功時は requeue しない)。
    - `next_group()` は結果キューから 1 グループ取り出す。
    - `__aexit__` で spawner と全ワーカーを cancel し、gather で回収する。

    Two queue topologies depending on `suite_mix_weights`:
      - None → single shuffled queue; mix follows pool sizes (legacy).
      - dict → per-suite queues + weighted suite sampler; mix follows the
               configured weights (suite name → weight). Re-queues go
               back to the source suite, preserving the target ratio.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        shared_sampling_client: SharedSamplingClient,
        seed_suites: list[SeedSuite],
        num_workers: int,
        stagger_s: float,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        rl_seed: int = 42,
        suite_mix_weights: dict[str, float] | None = None,
    ) -> None:
        self._engine = rollout_engine
        self._shared_sampling_client = shared_sampling_client
        self._seed_suites = seed_suites
        self._num_workers = num_workers
        self._stagger_s = stagger_s
        self._num_samples = num_samples
        self._sampling_params = sampling_params
        self._rl_seed = rl_seed
        self._suite_mix_weights = suite_mix_weights

        # Single-queue mode (legacy / suite_mix_weights=None).
        self._tasks_queue: asyncio.Queue[tuple[Task, RLSource]] = asyncio.Queue()
        # Per-suite-queue mode (suite_mix_weights set).
        self._per_suite_queues: dict[str, asyncio.Queue[tuple[Task, RLSource]]] = {}
        self._suite_names: list[str] = []  # for weighted sampling
        self._suite_weights: list[float] = []

        self._results_queue: asyncio.Queue[RLGroup] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._spawner_task: asyncio.Task | None = None

    async def __aenter__(self) -> "RLWorkerPool":
        rl_suites = [s for s in self._seed_suites if s.for_rl]
        if self._suite_mix_weights is None:
            await self._seed_single_queue(rl_suites)
        else:
            await self._seed_per_suite_queues(rl_suites, self._suite_mix_weights)

        async def _staggered_spawner() -> None:
            for i in range(self._num_workers):
                self._worker_tasks.append(asyncio.create_task(self._worker()))
                if i < self._num_workers - 1:
                    await asyncio.sleep(self._stagger_s)

        self._spawner_task = asyncio.create_task(_staggered_spawner())
        return self

    async def _seed_single_queue(self, rl_suites: list[SeedSuite]) -> None:
        all_pairs: list[tuple[Task, RLSource]] = []
        for suite in rl_suites:
            source = RLSource(id=suite.name, title=suite.name)
            for task in suite.tasks:
                all_pairs.append((task, source))
        random.Random(self._rl_seed).shuffle(all_pairs)
        logger.info(
            f"Seeded {len(all_pairs)} RL tasks into single shuffled queue "
            f"(seed={self._rl_seed})."
        )
        for pair in all_pairs:
            await self._tasks_queue.put(pair)

    async def _seed_per_suite_queues(
        self, rl_suites: list[SeedSuite], weights: dict[str, float]
    ) -> None:
        unknown = set(weights.keys()) - {s.name for s in rl_suites}
        if unknown:
            raise ValueError(
                f"suite_mix_weights references unknown suite name(s): {unknown}. "
                f"Available RL suites: {sorted(s.name for s in rl_suites)}."
            )
        for suite in rl_suites:
            w = weights.get(suite.name, 0.0)
            if w <= 0:
                logger.info(
                    f"Skipping RL suite '{suite.name}' (weight={w} in mix)."
                )
                continue
            q: asyncio.Queue[tuple[Task, RLSource]] = asyncio.Queue()
            source = RLSource(id=suite.name, title=suite.name)
            pairs = [(task, source) for task in suite.tasks]
            random.Random(self._rl_seed).shuffle(pairs)
            for pair in pairs:
                await q.put(pair)
            self._per_suite_queues[suite.name] = q
            self._suite_names.append(suite.name)
            self._suite_weights.append(w)
            logger.info(
                f"Seeded suite '{suite.name}': {len(pairs)} tasks, weight={w}."
            )
        if not self._suite_names:
            raise ValueError("suite_mix_weights produced no usable suites.")
        total = sum(self._suite_weights)
        self._suite_weights = [w / total for w in self._suite_weights]
        logger.info(
            "RL mix (normalised): "
            + ", ".join(
                f"{n}={p:.3f}" for n, p in zip(self._suite_names, self._suite_weights)
            )
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._spawner_task:
            self._spawner_task.cancel()
        for t in self._worker_tasks:
            t.cancel()

        tasks_to_wait: list[asyncio.Task] = []
        if self._spawner_task:
            tasks_to_wait.append(self._spawner_task)
        tasks_to_wait.extend(self._worker_tasks)
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def next_group(self) -> RLGroup:
        return await self._results_queue.get()

    async def _worker(self) -> None:
        rng = random.Random()  # per-worker, used only in per-suite-queue mode
        while True:
            try:
                task, source = await self._next_task(rng)
                indexed_client = self._shared_sampling_client.get_client()

                group = await self._collect_rl_group(task, source, indexed_client)
                if group:
                    await self._results_queue.put(group)

                # Drop from the pool when the model solves the task perfectly
                # this round — uniform rewards yield zero PPO gradient (already
                # filtered by RLBatchState), so re-queueing only burns rollout
                # compute on already-mastered tasks.
                solved = (
                    group is not None
                    and bool(group.rewards)
                    and all(r >= 1.0 for r in group.rewards)
                )
                if solved:
                    logger.info(
                        f"Task '{task.id}' (suite={source.id}) achieved 100% "
                        f"success ({len(group.rewards)}/{len(group.rewards)}); "
                        "removing from RL pool."
                    )
                else:
                    await self._requeue(task, source)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"RL worker error: {e}")
                await asyncio.sleep(1)

    async def _next_task(
        self, rng: random.Random
    ) -> tuple[Task, RLSource]:
        """Pick the next task. Single-queue mode just dequeues; per-suite-
        queue mode samples a suite by weight (skipping empty queues) then
        dequeues from that suite. If all suites are momentarily empty
        (rare — other workers will re-queue shortly), waits briefly."""
        if self._suite_mix_weights is None:
            pair = await self._tasks_queue.get()
            self._tasks_queue.task_done()
            return pair

        while True:
            available_names: list[str] = []
            available_weights: list[float] = []
            for name, w in zip(self._suite_names, self._suite_weights):
                if not self._per_suite_queues[name].empty():
                    available_names.append(name)
                    available_weights.append(w)
            if not available_names:
                # All suite queues drained mid-flight; back off until a
                # peer worker requeues. Should be transient.
                await asyncio.sleep(0.1)
                continue
            picked = rng.choices(available_names, weights=available_weights)[0]
            q = self._per_suite_queues[picked]
            pair = await q.get()
            q.task_done()
            return pair

    async def _requeue(self, task: Task, source: RLSource) -> None:
        """Put a task back in its source queue (per-suite mode) or in the
        shared queue (single-queue mode)."""
        if self._suite_mix_weights is None:
            await self._tasks_queue.put((task, source))
        else:
            await self._per_suite_queues[source.id].put((task, source))

    async def _collect_rl_group(
        self,
        task: Task,
        source: RLSource,
        indexed_client: IndexedSamplingClient,
    ) -> RLGroup | None:
        batch = await self._engine.run(
            sampling_client=indexed_client,
            instruction=task.instruction,
            num_samples=self._num_samples,
            sampling_params=self._sampling_params,
        )
        return rollout_batch_to_rl_group(
            batch,
            suite_name=source.id,
            task_id=task.id,
            instruction=task.instruction,
            sampling_client_version=indexed_client.version,
        )
