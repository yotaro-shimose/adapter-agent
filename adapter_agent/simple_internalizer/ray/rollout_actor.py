import asyncio
import logging

import ray
import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from ray.util.queue import Queue as RayQueue

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.simple_internalizer.distilled_qra_manager import DistilledQRAManager
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rl_worker_pool import RLWorkerPool
from adapter_agent.simple_internalizer.rollout_engine import (
    RolloutEngine,
    build_solver_system_prompt,
)
from adapter_agent.simple_internalizer.types import SeedSuite

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.5)
class RolloutActor:
    """Ray async actor that wraps an unmodified `RLWorkerPool` in its own process.

    各 actor は driver から渡された設定で per-process な依存
    (RuntimePool / Verifier / Prisma / RolloutEngine) を構築し、内部で
    `RLWorkerPool` を `__aenter__` する。生成された RLGroup は actor 内の
    drain task が受け取って共有 `RayQueue` へ push する。
    """

    def __init__(
        self,
        *,
        actor_index: int,
        runtime_settings: RuntimeSettings,
        runtime_pool_size: int,
        verifier_model: AgentsSDKModel,
        library_name: str,
        simple_train_id: str,
        model_name: str,
        num_workers: int,
        stagger_s: float,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        results_queue: RayQueue,
    ) -> None:
        self._actor_index = actor_index
        self._runtime_settings = runtime_settings
        self._runtime_pool_size = runtime_pool_size
        self._verifier_model = verifier_model
        self._library_name = library_name
        self._simple_train_id = simple_train_id
        self._model_name = model_name
        self._num_workers = num_workers
        self._stagger_s = stagger_s
        self._num_samples = num_samples
        self._sampling_params = sampling_params
        self._results_queue = results_queue

        self._db: PostgresDB | None = None
        self._runtime_pool: RuntimePool | None = None
        self._shared: SharedSamplingClient | None = None
        self._pool: RLWorkerPool | None = None
        self._drain_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(
        self,
        sampling_client: tinker.SamplingClient,
        seed_suites: list[SeedSuite],
    ) -> None:
        if self._initialized:
            return

        logger.info(
            f"[RolloutActor#{self._actor_index}] initializing "
            f"(num_workers={self._num_workers}, runtime_pool_size={self._runtime_pool_size})"
        )

        self._shared = SharedSamplingClient(sampling_client)

        _, renderer = get_tokenizer_renderer(sampling_client, self._model_name)

        self._db = PostgresDB()
        await self._db.connect()
        prisma_client = await self._db.get_client()
        await prisma_client.simpletrainrun.upsert(
            where={"id": self._simple_train_id},
            data={"create": {"id": self._simple_train_id}, "update": {}},
        )

        verifier = Verifier(model=self._verifier_model, library_name=self._library_name)
        self._runtime_pool = RuntimePool(
            self._runtime_settings, max_size=self._runtime_pool_size
        )
        executor = InternalizeExecutor(
            runtime_pool=self._runtime_pool, verifier=verifier
        )

        rollout_engine = RolloutEngine(
            renderer=renderer,
            executor=executor,
            system_prompt=build_solver_system_prompt(self._library_name),
        )

        # Study scope外: queue=None で on_all_fail は no-op
        distilled = DistilledQRAManager(
            qra_in_queue=None,
            study_task_queue=None,
        )

        self._pool = RLWorkerPool(
            rollout_engine=rollout_engine,
            shared_sampling_client=self._shared,
            seed_suites=seed_suites,
            num_workers=self._num_workers,
            stagger_s=self._stagger_s,
            num_samples=self._num_samples,
            sampling_params=self._sampling_params,
            distilled=distilled,
        )
        await self._pool.__aenter__()
        self._drain_task = asyncio.create_task(
            self._drain_loop(), name=f"rollout-actor-{self._actor_index}-drain"
        )
        self._initialized = True
        logger.info(f"[RolloutActor#{self._actor_index}] initialized.")

    async def _drain_loop(self) -> None:
        assert self._pool is not None
        while True:
            try:
                group = await self._pool.next_group()
                await self._results_queue.put_async(group)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    f"[RolloutActor#{self._actor_index}] drain loop error: {e}"
                )
                await asyncio.sleep(1)

    async def update_sampling_client(
        self, new_client: tinker.SamplingClient
    ) -> None:
        if self._shared is None:
            return
        self._shared.update_client(new_client)

    async def shutdown(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._drain_task = None

        if self._pool is not None:
            try:
                await self._pool.__aexit__(None, None, None)
            except Exception as e:
                logger.exception(
                    f"[RolloutActor#{self._actor_index}] pool teardown error: {e}"
                )
            self._pool = None

        if self._runtime_pool is not None:
            try:
                await self._runtime_pool.close_all()
            except Exception as e:
                logger.exception(
                    f"[RolloutActor#{self._actor_index}] runtime pool close error: {e}"
                )
            self._runtime_pool = None

        if self._db is not None:
            try:
                await self._db.close()
            except Exception as e:
                logger.exception(
                    f"[RolloutActor#{self._actor_index}] db close error: {e}"
                )
            self._db = None

        self._initialized = False
