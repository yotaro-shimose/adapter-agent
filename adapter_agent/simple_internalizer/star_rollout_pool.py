"""STaR 用の Ray rollout pool (driver facade)。

`stream_one_pass(tasks)` が本質: 全 task を fan-out し、結果 `(Task, RolloutBatch)`
を完了順にストリームで yield する (len(tasks) 件 yield すれば完了)。
`RayRLWorkerPool` と違い、内部で task を再投入し続ける continuous streaming は
しない (STaR は iteration 境界を持つ)。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import ray
import tinker
from oai_utils import AgentsSDKModel
from ray.util.queue import Queue as RayQueue

from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.rollout_engine import RolloutBatch
from adapter_agent.simple_internalizer.star_rollout_actor import (
    STaRRolloutActor,
    SHUTDOWN_SENTINEL,
)

logger = logging.getLogger(__name__)


class STaRRolloutPool:
    def __init__(
        self,
        *,
        num_processes: int,
        workers_per_process: int,
        runtime_pool_size_per_process: int,
        actor_stagger_s: float,
        worker_stagger_s: float,
        runtime_settings: RuntimeSettings,
        verifier_model: AgentsSDKModel,
        library_name: str,
        model_name: str,
        sampling_client: tinker.SamplingClient,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> None:
        self._num_processes = num_processes
        self._workers_per_process = workers_per_process
        self._runtime_pool_size_per_process = runtime_pool_size_per_process
        self._actor_stagger_s = actor_stagger_s
        self._worker_stagger_s = worker_stagger_s
        self._runtime_settings = runtime_settings
        self._verifier_model = verifier_model
        self._library_name = library_name
        self._model_name = model_name
        self._sampling_client = sampling_client
        self._num_samples = num_samples
        self._sampling_params = sampling_params

        self._actors: list = []
        self._task_queue: RayQueue | None = None
        self._result_queue: RayQueue | None = None

    async def __aenter__(self) -> "STaRRolloutPool":
        if not ray.is_initialized():
            logger.info(
                "Initializing Ray (local) without runtime_env "
                "(workers inherit driver os.environ)."
            )
            # Intentionally do NOT pass `runtime_env`. Ray 2.55+ treats any
            # `runtime_env` with a uv-managed project as a signal to re-run
            # `uv sync` inside each worker's venv, which fails for editable
            # deps outside the packaged working_dir (e.g. `../coder-mcp`).
            # For local (single-node) Ray, workers naturally inherit the driver
            # process's env vars, so explicit forwarding is unnecessary.
            ray.init(ignore_reinit_error=True)

        total_workers = self._num_processes * self._workers_per_process
        self._task_queue = RayQueue(maxsize=max(total_workers * 4, 64))
        self._result_queue = RayQueue(maxsize=max(total_workers * 4, 64))

        for i in range(self._num_processes):
            actor = STaRRolloutActor.remote(
                actor_index=i,
                runtime_settings=self._runtime_settings,
                runtime_pool_size=self._runtime_pool_size_per_process,
                verifier_model=self._verifier_model,
                library_name=self._library_name,
                model_name=self._model_name,
                num_workers=self._workers_per_process,
                stagger_s=self._worker_stagger_s,
                num_samples=self._num_samples,
                sampling_params=self._sampling_params,
                task_queue=self._task_queue,
                result_queue=self._result_queue,
            )
            self._actors.append(actor)

        sampling_client_ref = ray.put(self._sampling_client)
        logger.info(
            f"Spawning {self._num_processes} STaRRolloutActors "
            f"(stagger={self._actor_stagger_s}s, workers/proc={self._workers_per_process})..."
        )
        for i, actor in enumerate(self._actors):
            await actor.initialize.remote(sampling_client_ref)
            if i < self._num_processes - 1:
                await asyncio.sleep(self._actor_stagger_s)
        logger.info(f"All {self._num_processes} STaRRolloutActors initialized.")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Best-effort: tell workers to stop polling
        if self._task_queue is not None:
            try:
                await self._task_queue.put_async(SHUTDOWN_SENTINEL)
            except Exception:
                pass

        if self._actors:
            logger.info(f"Shutting down {len(self._actors)} STaRRolloutActors...")
            await asyncio.gather(
                *[a.shutdown.remote() for a in self._actors],
                return_exceptions=True,
            )
            for actor in self._actors:
                try:
                    ray.kill(actor)
                except Exception as e:
                    logger.warning(f"ray.kill failed (ignored): {e}")
            self._actors = []

        for q in (self._task_queue, self._result_queue):
            if q is not None:
                try:
                    q.shutdown(force=True)
                except Exception as e:
                    logger.warning(f"RayQueue shutdown failed (ignored): {e}")
        self._task_queue = None
        self._result_queue = None

    async def stream_one_pass(
        self, tasks: list[Task]
    ) -> AsyncIterator[tuple[Task, RolloutBatch]]:
        """全 task を投入し、結果が 1 件到着するたびに yield する。

        呼び出し側は `async for task, batch in pool.stream_one_pass(tasks):` で
        線形にストリーム処理できる (進捗ログ、インクリメンタルな buffer 追加、
        早期打ち切り等は呼び出し側の責務)。yield 総数は `len(tasks)` と一致する。
        """
        assert self._task_queue is not None, "STaRRolloutPool not entered"
        assert self._result_queue is not None
        if not tasks:
            return

        async def _producer() -> None:
            for t in tasks:
                await self._task_queue.put_async(t)

        producer_task = asyncio.create_task(_producer())
        try:
            for _ in range(len(tasks)):
                item = await self._result_queue.get_async()
                yield item
        finally:
            await producer_task

    async def broadcast_sampling_client(
        self, new_client: tinker.SamplingClient
    ) -> None:
        if not self._actors:
            return
        ref = ray.put(new_client)
        await asyncio.gather(
            *[a.update_sampling_client.remote(ref) for a in self._actors]
        )
