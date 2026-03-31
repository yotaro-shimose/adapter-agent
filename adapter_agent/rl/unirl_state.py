import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Mapping, Self, Sequence

import ray
from oai_utils.tinker import TinkerModel
from pydantic import BaseModel
from ray.actor import ActorHandle
from tinker import AdamParams, Datum
from tinker.types import LossFnType
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.rl.env.conclusion import conclusion_to_metrics
from adapter_agent.rl.env.session_result import (
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
)
from adapter_agent.rl.postgres_db import get_db
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.task_net import (
    StudyTaskCompleted,
    TaskContext,
    TaskNetwork,
    TaskNetworkCompleted,
    TaskNetworkTask,
)
from adapter_agent.rl.trajectory_db import TrajectoryDB
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)


@dataclass
class Counted[T]:
    count: int
    item: T


type BatchType = Literal["Study"]


class StudyRolloutParams(BaseModel):
    num_study_actor: int
    rollouts_per_actor: int


class UniRLTrainParams(BaseModel):
    update_freq: int
    save_freq: int
    adam_params: AdamParams
    max_sft_reuse: int
    min_sft_batch_size: int
    max_sft_batch_size: int


class UniRLConfig(BaseModel):
    experiment_setting: ExperimentSettings
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings
    study_rollout_params: StudyRolloutParams
    train_params: UniRLTrainParams
    log_freq: int
    vis_json_path: Path | None = None


class TinkerBatch(BaseModel):
    datum: List[Datum]
    loss_fn: LossFnType
    tasks: dict[str, Counted[Task]]
    batch_type: BatchType


@dataclass
class TrajectoryReplayBuffer:
    min_sft_batch_size: int
    max_sft_batch_size: int
    max_sft_reuse: int
    trajectory_db: TrajectoryDB
    renderer: Renderer

    async def get_batch(self) -> TinkerBatch | None:
        # Check available count before potentially consuming/deleting trajectories
        count = await self.trajectory_db.get_count(self.max_sft_reuse)
        if count < self.min_sft_batch_size:
            # Added for visibility: only log occasionally if it's repeatedly empty
            logger.info(
                f"Buffer check: {count}/{self.min_sft_batch_size} trajectories available. Training waiting..."
            )
            return None

        batch_size = self.max_sft_batch_size
        items = await self.trajectory_db.get_batch(batch_size, self.max_sft_reuse)
        if len(items) < self.min_sft_batch_size:
            return None

        logger.info(f"Training Batch Ready! size: {len(items)}")
        datum = [
            conversation_to_datum(
                item["messages"],
                self.renderer,
                max_length=None,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            for item in items
        ]

        logger.info(
            "Datum construction complete (conversation_to_datum finished). Returning batch..."
        )

        tasks = {
            item["task_id"]: Counted(
                item=Task.from_instruction(item["task_id"]),
                count=max(0, self.max_sft_reuse - item["usage_count"]),
            )
            for item in items
        }
        return TinkerBatch(
            datum=datum, loss_fn="cross_entropy", tasks=tasks, batch_type="Study"
        )

    async def get_metrics(self) -> dict[str, int]:
        count = await self.trajectory_db.get_count(self.max_sft_reuse)
        return {
            "study_buffer_size": count,
        }


@dataclass
class MetricManager:
    ml_logger: MLLogger
    study_metrics_list: List[dict[str, float]]
    log_freq: int

    @classmethod
    def create(cls, ml_logger: MLLogger, log_freq: int) -> Self:
        return cls(
            ml_logger=ml_logger,
            study_metrics_list=[],
            log_freq=log_freq,
        )

    def mean_metrics(self, metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
        if not metrics:
            return {}
        return {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}

    def with_prefix(
        self, metrics: dict[str, Any], prefix: str, separator: str = "/"
    ) -> dict[str, Any]:
        return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}

    def register_study_metrics(
        self,
        metrics: dict[str, float | int],
        non_averaging_metrics: dict[str, float | int],
    ):
        self.study_metrics_list.append(metrics)
        if self.should_study_log():
            mean_metrics = self.mean_metrics(self.study_metrics_list)
            self.ml_logger.log_metrics(self.with_prefix(mean_metrics, "study"))
            self.ml_logger.log_metrics(self.with_prefix(non_averaging_metrics, "study"))
            self.study_metrics_list = []

    def should_study_log(self) -> bool:
        return len(self.study_metrics_list) % self.log_freq == 0


@dataclass
class UniRLState:
    # Model
    study_task_queue: TaskNetwork
    buffer: TrajectoryReplayBuffer
    litellm_model_name: str
    renderer: Renderer
    sampling_client_manager: SharedSamplingClient
    task_verifier: TaskVerifier
    library_name: str
    cfg: UniRLConfig
    experiment_id: int
    metric_manager: MetricManager = field(init=False, default=None)  # type: ignore
    ml_logger: MLLogger = field(init=False, default=None)  # type: ignore

    def __post_init__(self):
        setup_base_loglevel()
        logger.setLevel(logging.DEBUG)
        TinkerModel.register_tinkerllm_to_litellm()
        log_root = self.cfg.experiment_setting.log_root()
        log_root.mkdir(parents=True, exist_ok=True)
        self.ml_logger = ml_log.setup_logging(
            log_dir=str(log_root),
            wandb_project=self.cfg.experiment_setting.wandb_project,
            config=self.cfg,
        )
        self.metric_manager = MetricManager.create(
            ml_logger=self.ml_logger,
            log_freq=self.cfg.log_freq,
        )

    @ray.method
    async def setup_db(self):
        # Initial save of graph to PostgreSQL for visualization
        db = await get_db()
        await db.update_graph_json(self.experiment_id, self.study_task_queue.to_dict())

        # Sync Knowledge from Postgres to ES (Postgres is SSOT)
        knowledges = await db.get_knowledges(self.experiment_id)
        if knowledges:
            logger.info(f"Syncing {len(knowledges)} knowledge items from Postgres to ES...")
            kdb = KnowledgeDB.for_experiment(self.experiment_id)
            await kdb.initialize()
            for k in knowledges:
                await kdb.add_knowledge(
                    id=k.id,
                    query=k.instruction,
                    title=k.title,
                    content=k.content
                )
            await kdb.close()
            logger.info("Knowledge sync complete.")

        # Also save to local data.json as backup
        log_root = self.cfg.experiment_setting.log_root()
        self.study_task_queue.save_json_sync(log_root / "data.json")

    @ray.method
    def get_next_task(self) -> TaskNetworkTask:
        task = self.study_task_queue.get_and_setup_next_task()
        return task

    @ray.method
    async def register_task_result(self, completed: TaskNetworkCompleted):
        await self.study_task_queue.task_teardown(completed)

        if not isinstance(completed, StudyTaskCompleted):
            return

        result = completed.result
        if not isinstance(result, RewireSessionResultNormal):
            return

        if result.reward > 0 or True:  # Save all trajectories for visualization SSOT
            # 1. Handle newly discovered knowledge (SSOT)
            knowledge_id = None
            if isinstance(result, RewireSessionResultSuccess) and result.knowledge:
                db = await get_db()
                # Save to Postgres first
                knowledge_id = await db.create_knowledge(
                    experiment_id=self.experiment_id,
                    task_id=result.task.id,
                    instruction=result.task.instruction,
                    title=result.knowledge.title,
                    content=result.knowledge.content,
                )
                # Then index in ES using the Postgres ID
                kdb = KnowledgeDB.for_experiment(self.experiment_id)
                await kdb.initialize()
                await kdb.add_knowledge(
                    id=knowledge_id,
                    query=result.task.instruction,
                    title=result.knowledge.title,
                    content=result.knowledge.content,
                )
                # Also update in-memory TaskNetwork knowledge ID for visualization mapping
                task_meta = self.study_task_queue.nodes[result.task.id]
                for k in task_meta.knowledges.values():
                    k.knowledge_id = str(knowledge_id)

                await kdb.close()

            # 2. Extract citations
            citations_data = [
                {
                    "knowledge_id": c.knowledge_id,
                    "turn_index": c.turn_index,
                    "content": c.content,
                    "title": c.title,
                }
                for c in result.citations
            ]
            
            # 3. Save trajectory with references to stable Knowledge IDs
            await self.buffer.trajectory_db.add_trajectory(
                task_id=result.task.id,
                instruction=result.task.instruction,
                conclusion=result.conclusion,
                reward=result.reward,
                trajectory=result.trials,
                final_knowledge=result.knowledge.content if result.knowledge else None,
                final_knowledge_title=result.knowledge.title if result.knowledge else None,
                citations=citations_data,
            )

        await self.handle_study_result(result)

        if self.metric_manager.should_study_log():
            # Save to PostgreSQL for visualization
            db = await get_db()
            await db.update_graph_json(
                self.experiment_id, self.study_task_queue.to_dict()
            )

            # Also save to local log_root as backup if vis_json_path is set
            if self.cfg.vis_json_path:
                await self.study_task_queue.save_json(self.cfg.vis_json_path)

    async def handle_study_result(self, result: RewireSessionResultNormal):
        metrics = conclusion_to_metrics(result.conclusion)
        buffer_metrics = await self.buffer.get_metrics()
        task_queue_metrics = {
            "study_queue_size": self.study_task_queue.node_count(),
        }
        non_averaging_metrics: dict[str, float | int] = {
            **buffer_metrics,
            **task_queue_metrics,
        }
        self.metric_manager.register_study_metrics(metrics, non_averaging_metrics)

    @ray.method
    def get_latest_model(self) -> TinkerModel:
        return self._get_latest_model()

    def is_finished(self) -> bool:
        return False

    @ray.method
    async def get_batch(self) -> TinkerBatch | None:
        return await self.buffer.get_batch()

    @ray.method
    def log_train_metrics(self, metrics: dict[str, Any]):
        # Prefix metrics with train/ for better W&B organization
        prefixed = {
            f"train/{k}" if not k.startswith("train/") else k: v
            for k, v in metrics.items()
        }
        logger.info(
            f"Logging train metrics: {prefixed}"
        )  # Ensure it shows up in logs.log
        self.ml_logger.log_metrics(prefixed)

    @ray.method
    def log_info(self, msg: str):
        """Allows other actors to write to the main experiment logs.log"""
        logger.info(msg)

    @ray.method
    async def update_sampling_client(self, new_client: Any):
        await self.sampling_client_manager.update_client(new_client)

    def _get_latest_model(self) -> TinkerModel:
        return TinkerModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
            renderer=self.renderer,
        )


async def task_manager_from_state_handle(
    state: ActorHandle[UniRLState],
) -> TaskContext:
    ret = await state.get_next_task.remote()
    register = state.register_task_result.remote
    return TaskContext(
        task=ret,
        register=register,
    )
