import asyncio
import logging
import random

import tinker
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl import Trajectory
from tinker_cookbook.rl.types import Transition

from adapter_agent.hierarchical.state import RLGroup, RolloutSample
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)

from .rollout_engine import RolloutEngine
from .types import RLSource, SeedSuite

logger = logging.getLogger(__name__)


class RLWorkerPool:
    """RL rollout を並列実行するワーカープール。

    - `__aenter__` で seed_suites をタスクキューに seeding し、N ワーカーを stagger を
      挟んで起動する。
    - 各ワーカーは task を取り出して RolloutEngine を回し、RLGroup を結果キューへ push、
      同じ task を再エンキューする。
    - `next_group()` は結果キューから 1 グループ取り出す。
    - `__aexit__` で spawner と全ワーカーを cancel し、gather で回収する。
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
    ) -> None:
        self._engine = rollout_engine
        self._shared_sampling_client = shared_sampling_client
        self._seed_suites = seed_suites
        self._num_workers = num_workers
        self._stagger_s = stagger_s
        self._num_samples = num_samples
        self._sampling_params = sampling_params
        self._rl_seed = rl_seed

        self._tasks_queue: asyncio.Queue[tuple[Task, RLSource]] = asyncio.Queue()
        self._results_queue: asyncio.Queue[RLGroup] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._spawner_task: asyncio.Task | None = None

    async def __aenter__(self) -> "RLWorkerPool":
        all_pairs: list[tuple[Task, RLSource]] = []
        for suite in self._seed_suites:
            if not suite.for_rl:
                continue
            source = RLSource(id=suite.name, title=suite.name)
            for task in suite.tasks:
                all_pairs.append((task, source))
        random.Random(self._rl_seed).shuffle(all_pairs)
        logger.info(
            f"Seeded {len(all_pairs)} RL tasks into queue (shuffled with seed={self._rl_seed})."
        )
        for pair in all_pairs:
            await self._tasks_queue.put(pair)

        async def _staggered_spawner() -> None:
            for i in range(self._num_workers):
                self._worker_tasks.append(asyncio.create_task(self._worker()))
                if i < self._num_workers - 1:
                    await asyncio.sleep(self._stagger_s)

        self._spawner_task = asyncio.create_task(_staggered_spawner())
        return self

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
        while True:
            try:
                task, source = await self._tasks_queue.get()
                indexed_client = self._shared_sampling_client.get_client()

                group = await self._collect_rl_group(task, source, indexed_client)
                if group:
                    await self._results_queue.put(group)

                self._tasks_queue.task_done()
                await self._tasks_queue.put((task, source))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"RL worker error: {e}")
                await asyncio.sleep(1)

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

        trajectories: list[Trajectory] = []
        rewards: list[float] = []
        samples: list[RolloutSample] = []
        for o in batch.outcomes:
            ac = TokensWithLogprobs(tokens=o.tokens, maybe_logprobs=o.logprobs)
            trajectories.append(
                Trajectory(
                    transitions=[
                        Transition(
                            ob=batch.prompt, ac=ac, reward=0.0, episode_done=True
                        )
                    ],
                    final_ob=batch.prompt,
                )
            )
            rewards.append(1.0 if o.success else 0.0)
            samples.append(
                RolloutSample(
                    answer=o.answer,
                    reasoning=o.reasoning,
                    parsed=o.parsed,
                    success=o.success,
                    execution_output=o.execution_output,
                    verification_output=o.verification_output,
                )
            )

        return RLGroup(
            trajectories=trajectories,
            rewards=rewards,
            sampling_client_version=indexed_client.version,
            suite_name=source.id,
            task_id=task.id,
            instruction=task.instruction,
            samples=samples,
        )
