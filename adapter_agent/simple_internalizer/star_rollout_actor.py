"""STaR 用 Ray actor。

`RolloutEngine` を per-process で立ち上げ、入力 task queue から task を受け取って
`run()` を回し、結果 `(Task, RolloutBatch)` を出力 queue へ push する。

`RayRLWorkerPool` 側の `RolloutActor` を参考にしているが、RL 特有コンポーネント
(`RLWorkerPool`, `DistilledQRAManager`, Prisma 書き込み、RLGroup 変換、streaming な
task 再投入) は落としてある。STaR は「task 群を 1 パス回して全部取る」モデルなので、
actor 側は task を取って結果を返すだけ。
"""

from __future__ import annotations

import asyncio
import logging

import ray
import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from ray.util.queue import Queue as RayQueue

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rollout_engine import (
    RolloutEngine,
    build_solver_system_prompt,
)

logger = logging.getLogger(__name__)


_SHUTDOWN_SENTINEL = "__STAR_SHUTDOWN__"


@ray.remote(num_cpus=0.5)
class STaRRolloutActor:
    def __init__(
        self,
        *,
        actor_index: int,
        runtime_settings: RuntimeSettings,
        runtime_pool_size: int,
        verifier_model: AgentsSDKModel,
        library_name: str,
        model_name: str,
        num_workers: int,
        stagger_s: float,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        task_queue: RayQueue,
        result_queue: RayQueue,
    ) -> None:
        self._actor_index = actor_index
        self._runtime_settings = runtime_settings
        self._runtime_pool_size = runtime_pool_size
        self._verifier_model = verifier_model
        self._library_name = library_name
        self._model_name = model_name
        self._num_workers = num_workers
        self._stagger_s = stagger_s
        self._num_samples = num_samples
        self._sampling_params = sampling_params
        self._task_queue = task_queue
        self._result_queue = result_queue

        self._runtime_pool: RuntimePool | None = None
        self._shared: SharedSamplingClient | None = None
        self._engine: RolloutEngine | None = None
        self._worker_tasks: list[asyncio.Task] = []
        self._spawner_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self, sampling_client: tinker.SamplingClient) -> None:
        if self._initialized:
            return
        logger.info(
            f"[STaRRolloutActor#{self._actor_index}] initializing "
            f"(num_workers={self._num_workers}, runtime_pool_size={self._runtime_pool_size})"
        )

        self._shared = SharedSamplingClient(sampling_client)
        _, renderer = get_tokenizer_renderer(sampling_client, self._model_name)

        verifier = Verifier(model=self._verifier_model, library_name=self._library_name)
        self._runtime_pool = RuntimePool(
            self._runtime_settings, max_size=self._runtime_pool_size
        )
        executor = InternalizeExecutor(
            runtime_pool=self._runtime_pool, verifier=verifier
        )

        self._engine = RolloutEngine(
            renderer=renderer,
            executor=executor,
            system_prompt=build_solver_system_prompt(self._library_name),
        )

        async def _staggered_spawner() -> None:
            for i in range(self._num_workers):
                self._worker_tasks.append(asyncio.create_task(self._worker()))
                if i < self._num_workers - 1:
                    await asyncio.sleep(self._stagger_s)

        self._spawner_task = asyncio.create_task(_staggered_spawner())
        self._initialized = True
        logger.info(f"[STaRRolloutActor#{self._actor_index}] initialized.")

    async def _worker(self) -> None:
        assert self._engine is not None
        assert self._shared is not None
        while True:
            try:
                task = await self._task_queue.get_async()
                if task == _SHUTDOWN_SENTINEL:
                    # Return sentinel so other workers can also see it
                    await self._task_queue.put_async(_SHUTDOWN_SENTINEL)
                    return
                assert isinstance(task, Task), f"Unexpected task type: {type(task)}"
                indexed_client = self._shared.get_client()
                batch = await self._engine.run(
                    sampling_client=indexed_client,
                    instruction=task.instruction,
                    num_samples=self._num_samples,
                    sampling_params=self._sampling_params,
                )
                await self._result_queue.put_async((task, batch))
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(
                    f"[STaRRolloutActor#{self._actor_index}] worker error: {e}"
                )
                await asyncio.sleep(1)

    async def update_sampling_client(
        self, new_client: tinker.SamplingClient
    ) -> None:
        if self._shared is None:
            return
        self._shared.update_client(new_client)

    async def shutdown(self) -> None:
        if self._spawner_task is not None:
            self._spawner_task.cancel()
            try:
                await self._spawner_task
            except (asyncio.CancelledError, Exception):
                pass
            self._spawner_task = None

        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks = []

        if self._runtime_pool is not None:
            try:
                await self._runtime_pool.close_all()
            except Exception as e:
                logger.exception(
                    f"[STaRRolloutActor#{self._actor_index}] runtime pool close error: {e}"
                )
            self._runtime_pool = None

        self._initialized = False


SHUTDOWN_SENTINEL = _SHUTDOWN_SENTINEL
