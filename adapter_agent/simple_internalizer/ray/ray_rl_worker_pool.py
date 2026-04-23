import asyncio
import logging
import os

import ray
import tinker
from oai_utils import AgentsSDKModel
from ray.util.queue import Queue as RayQueue

from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.ray.rollout_actor import RolloutActor
from adapter_agent.simple_internalizer.types import SeedSuite

logger = logging.getLogger(__name__)

_ENV_KEYS_TO_FORWARD = (
    "TINKER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_AGENTS_DISABLE_TRACING",
    "DATABASE_URL",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "WANDB_API_KEY",
)


class RayRLWorkerPool:
    """`RLWorkerPool` 互換 facade。複数の `RolloutActor` プロセスを束ねる。

    interface:
      - `__aenter__`: ray.init → RayQueue 生成 → N actor を spawn → stagger 付き init
      - `next_group()`: 全 actor からの結果を RayQueue 経由で 1 つ取得
      - `broadcast_sampling_client()`: GRPO 後に各 actor の SharedSamplingClient を
        新しい `tinker.SamplingClient` で差し替える
      - `__aexit__`: 全 actor を shutdown → kill → RayQueue を close
    """

    def __init__(
        self,
        *,
        num_processes: int,
        workers_per_process: int,
        runtime_pool_size_per_process: int,
        actor_stagger_s: float,
        runtime_settings: RuntimeSettings,
        verifier_model: AgentsSDKModel,
        library_name: str,
        simple_train_id: str,
        model_name: str,
        sampling_client: tinker.SamplingClient,
        seed_suites: list[SeedSuite],
        worker_stagger_s: float,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> None:
        self._num_processes = num_processes
        self._workers_per_process = workers_per_process
        self._runtime_pool_size_per_process = runtime_pool_size_per_process
        self._actor_stagger_s = actor_stagger_s
        self._runtime_settings = runtime_settings
        self._verifier_model = verifier_model
        self._library_name = library_name
        self._simple_train_id = simple_train_id
        self._model_name = model_name
        self._sampling_client = sampling_client
        self._seed_suites = seed_suites
        self._worker_stagger_s = worker_stagger_s
        self._num_samples = num_samples
        self._sampling_params = sampling_params

        self._actors: list = []
        self._results_queue: RayQueue | None = None

    async def __aenter__(self) -> "RayRLWorkerPool":
        if not ray.is_initialized():
            env_vars = {
                k: os.environ[k] for k in _ENV_KEYS_TO_FORWARD if k in os.environ
            }
            logger.info(
                f"Initializing Ray (local) with {len(env_vars)} forwarded env vars."
            )
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"env_vars": env_vars},
            )

        queue_maxsize = max(
            self._num_processes * self._workers_per_process * 2, 16
        )
        self._results_queue = RayQueue(maxsize=queue_maxsize)

        for i in range(self._num_processes):
            actor = RolloutActor.remote(
                actor_index=i,
                runtime_settings=self._runtime_settings,
                runtime_pool_size=self._runtime_pool_size_per_process,
                verifier_model=self._verifier_model,
                library_name=self._library_name,
                simple_train_id=self._simple_train_id,
                model_name=self._model_name,
                num_workers=self._workers_per_process,
                stagger_s=self._worker_stagger_s,
                num_samples=self._num_samples,
                sampling_params=self._sampling_params,
                results_queue=self._results_queue,
            )
            self._actors.append(actor)

        sampling_client_ref = ray.put(self._sampling_client)
        seed_suites_ref = ray.put(self._seed_suites)

        logger.info(
            f"Spawning {self._num_processes} RolloutActors "
            f"(stagger={self._actor_stagger_s}s, workers/proc={self._workers_per_process})..."
        )
        for i, actor in enumerate(self._actors):
            await actor.initialize.remote(sampling_client_ref, seed_suites_ref)
            if i < self._num_processes - 1:
                await asyncio.sleep(self._actor_stagger_s)
        logger.info(f"All {self._num_processes} RolloutActors initialized.")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._actors:
            logger.info(f"Shutting down {len(self._actors)} RolloutActors...")
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

        if self._results_queue is not None:
            try:
                self._results_queue.shutdown(force=True)
            except Exception as e:
                logger.warning(f"RayQueue shutdown failed (ignored): {e}")
            self._results_queue = None

    async def next_group(self) -> RLGroup:
        assert self._results_queue is not None, "RayRLWorkerPool not entered"
        return await self._results_queue.get_async()

    async def broadcast_sampling_client(
        self, new_client: tinker.SamplingClient
    ) -> None:
        if not self._actors:
            return
        ref = ray.put(new_client)
        await asyncio.gather(
            *[a.update_sampling_client.remote(ref) for a in self._actors]
        )
