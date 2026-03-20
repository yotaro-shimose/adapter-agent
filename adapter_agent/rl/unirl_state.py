import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Self, Sequence

import ray
from oai_utils.tinker import TinkerModel
from pydantic import BaseModel
from ray.actor import ActorHandle
from tinker import AdamParams, Datum
from tinker.types import LossFnType
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
)
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    SolveVerifyTinkerSingleTurnResult,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.simplified_solver import conclusion_to_metrics
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.task_net import (
    StudyTask,
    StudyTaskCompleted,
    StudyTaskContext,
    TaskNetwork,
)
from adapter_agent.rl.trajectory import prepare_minibatch_simplified_sync
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.task_queue import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class Counted[T]:
    count: int
    item: T


@dataclass
class SFTSample:
    task: Task
    messages: list[TinkerMessage]


@dataclass
class RLSample:
    task: Task
    group: TrajectoryGroup

    def is_uniform_reward(self) -> bool:
        if len(self.group.final_rewards_G) == 0:
            return True
        return all(
            r == self.group.final_rewards_G[0] for r in self.group.final_rewards_G
        )


type BatchType = Literal["Study", "Practice"]


class StudyRolloutParams(BaseModel):
    num_study_actor: int
    rollouts_per_actor: int


class PracticeRolloutParams(BaseModel):
    num_practice_workers: int
    rollouts_per_task: int


class UniRLTrainParams(BaseModel):
    update_freq: int
    save_freq: int
    adam_params: AdamParams
    max_sft_reuse: int
    max_rl_reuse: int
    min_sft_batch_size: int
    max_sft_batch_size: int
    rl_group_size: int


class UniRLConfig(BaseModel):
    experiment_setting: ExperimentSettings
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings
    study_rollout_params: StudyRolloutParams
    practice_rollout_params: PracticeRolloutParams
    train_params: UniRLTrainParams
    log_freq: int


class TinkerBatch(BaseModel):
    datum: list[Datum]
    loss_fn: LossFnType
    tasks: dict[str, Counted[Task]]
    batch_type: BatchType


@dataclass
class CountDownStore[T]:
    items: dict[str, Counted[T]]
    init_count: int

    def get_batch(self, batch_size: int) -> list[Counted[T]]:
        batch_size = min(batch_size, len(self.items))
        if not self.items:
            return []
        keys = random.sample(list(self.items.keys()), k=batch_size)
        items: list[Counted[T]] = []
        for key in keys:
            item = self.items[key]
            items.append(item)
            item.count -= 1
            if item.count == 0:
                self.items.pop(key)
        return items

    def put(self, item: T):
        key = str(uuid.uuid4())
        self.items[key] = Counted(item=item, count=self.init_count)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class HybridReplayBuffer:
    min_sft_batch_size: int
    max_sft_batch_size: int
    rl_group_size: int
    sft_store: CountDownStore[SFTSample]
    rl_store: CountDownStore[RLSample]
    renderer: Renderer

    def get_batch(self) -> TinkerBatch | None:
        batch_type = self.choose_available_batch_type()
        if batch_type == "Study":
            batch_size = min(self.max_sft_batch_size, len(self.sft_store))
            items = self.sft_store.get_batch(batch_size)
            datum = [
                conversation_to_datum(
                    item.item.messages,
                    self.renderer,
                    max_length=None,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                for item in items
            ]
            tasks = {
                item.item.task.id: Counted(item=item.item.task, count=item.count)
                for item in items
            }
            return TinkerBatch(
                datum=datum, loss_fn="cross_entropy", tasks=tasks, batch_type="Study"
            )
        elif batch_type == "Practice":
            items = self.rl_store.get_batch(self.rl_group_size)
            traj_groups = [item.item.group for item in items]
            datum, _prepare_minibatch_metrics = prepare_minibatch_simplified_sync(
                traj_groups
            )
            datum_cleaned = [self._remove_mask(d) for d in datum]
            return TinkerBatch(
                datum=datum_cleaned,
                loss_fn="ppo",
                tasks={
                    item.item.task.id: Counted(item=item.item.task, count=item.count)
                    for item in items
                },
                batch_type="Practice",
            )
        else:
            return None

    def choose_available_batch_type(self) -> BatchType | None:
        availables: list[BatchType] = []

        if len(self.sft_store) >= self.min_sft_batch_size:
            availables.append("Study")
        if len(self.rl_store) >= self.rl_group_size:
            availables.append("Practice")

        if len(availables) == 0:
            return None

        item = random.choice(availables)
        return item

    def _remove_mask(self, datum: Datum) -> Datum:
        return Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"
            },
        )

    @classmethod
    def create(
        cls,
        min_sft_batch_size: int,
        max_sft_batch_size: int,
        rl_group_size: int,
        max_sft_reuse: int,
        max_rl_reuse: int,
        renderer: Renderer,
    ) -> Self:
        return cls(
            min_sft_batch_size=min_sft_batch_size,
            max_sft_batch_size=max_sft_batch_size,
            rl_group_size=rl_group_size,
            sft_store=CountDownStore({}, init_count=max_sft_reuse),
            rl_store=CountDownStore({}, init_count=max_rl_reuse),
            renderer=renderer,
        )

    def get_metrics(self) -> dict[str, int]:
        return {
            "study_buffer_size": len(self.sft_store),
            "practice_buffer_size": len(self.rl_store),
        }


@dataclass
class MetricManager:
    ml_logger: MLLogger
    study_metrics_list: list[dict[str, float]]
    practice_metrics_list: list[dict[str, float]]
    log_freq: int

    @classmethod
    def create(cls, ml_logger: MLLogger, log_freq: int) -> Self:
        return cls(
            ml_logger=ml_logger,
            study_metrics_list=[],
            practice_metrics_list=[],
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

    def register_practice_metrics(self, metrics: dict[str, float]):
        self.practice_metrics_list.append(metrics)
        if len(self.practice_metrics_list) % self.log_freq == 0:
            mean_metrics = self.mean_metrics(self.practice_metrics_list)
            self.ml_logger.log_metrics(self.with_prefix(mean_metrics, "practice"))
            self.practice_metrics_list = []

    def should_study_log(self) -> bool:
        return len(self.study_metrics_list) % self.log_freq == 0


@dataclass
class UniRLState:
    # Model
    study_task_queue: TaskNetwork
    practice_task_queue: TaskQueue[Task]
    buffer: HybridReplayBuffer
    litellm_model_name: str
    renderer: Renderer
    sampling_client_manager: SharedSamplingClient
    task_verifier: TaskVerifier
    library_name: str
    cfg: UniRLConfig
    metric_manager: MetricManager = field(init=False)
    ml_logger: MLLogger = field(init=False)

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
    def get_next_study_task(self) -> StudyTask:
        study_task = self.study_task_queue.get_and_setup_next_study_task()
        return study_task

    @ray.method
    def register_study_result(self, study_task: StudyTaskCompleted):
        self.study_task_queue.study_task_teardown(study_task)
        result = study_task.result
        if not isinstance(result, RewireSessionResultNormal):
            return

        self.handle_study_result(result)
        if isinstance(result, RewireSessionResultSuccess):
            self.buffer.sft_store.put(
                SFTSample(
                    task=study_task.task, messages=result.qra.to_tinker_messages()
                )
            )
        if self.metric_manager.should_study_log():
            self.study_task_queue.save_visualization(
                self.cfg.experiment_setting.log_root() / "task_net.html"
            )

    def handle_study_result(self, result: RewireSessionResultNormal):
        metrics = conclusion_to_metrics(result.conclusion)
        buffer_metrics = self.buffer.get_metrics()
        task_queue_metrics = {
            "study_queue_size": self.study_task_queue.node_count(),
            "practice_queue_size": self.practice_task_queue.qsize(),
        }
        non_averaging_metrics: dict[str, float | int] = {
            **buffer_metrics,
            **task_queue_metrics,
        }
        self.metric_manager.register_study_metrics(metrics, non_averaging_metrics)

    @ray.method
    async def get_next_practice_task(self) -> Task:
        while not self.is_finished():
            try:
                return await asyncio.wait_for(
                    self.practice_task_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
        raise IndexError("System finished")

    @ray.method
    async def add_to_practice_queue(self, task: Task):
        await self.practice_task_queue.put(task)

    @ray.method
    def get_latest_model(self) -> TinkerModel:
        return self._get_latest_model()

    def is_finished(self) -> bool:
        # TODO: implement actual finish condition
        return False

    @ray.method
    async def register_practice_group(
        self, worker_id: int, items: list[SolveVerifyTinkerSingleTurnResult]
    ):
        # TODO: fix this
        raise NotImplementedError
        # if not items:
        #     return

        # rewards = [item.reward for item in items]
        # task = items[0].env_state.task

        # for item in items:
        #     self.metric_manager.register_practice_metrics(
        #         conclusion_to_metrics(item.conclusion)
        #     )

        # success_counts = sum([int(ret.is_success()) for ret in items])

        # if success_counts == 0:
        #     await self.study_task_queue.put(
        #         TaskWithMeta.from_task(
        #             task, level=0
        #         )  # Practice tasks are likely top-level or level not easily tracked here
        #     )
        #     logger.info(
        #         f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, putting it to study queue."
        #     )
        #     return

        # elif success_counts == len(items):
        #     logger.info(
        #         f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, removing the task from the queue"
        #     )
        #     return

        # else:
        #     await self.practice_task_queue.put(task)
        #     logger.info(
        #         f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, putting it back to practice queue."
        #     )

        # if self.uniform_reward(rewards):
        #     return

        # rl_sample = RLSample(
        #     task=task,
        #     group=TrajectoryGroup(
        #         trajectories_G=[item.trajectory for item in items],
        #         final_rewards_G=rewards,
        #         metrics_G=[],
        #     ),
        # )
        # if not rl_sample.is_uniform_reward():
        #     self.buffer.rl_store.put(rl_sample)

    @ray.method
    def get_batch(self) -> TinkerBatch | None:
        return self.buffer.get_batch()

    @ray.method
    def log_train_metrics(self, metrics: dict[str, Any]):
        self.ml_logger.log_metrics(metrics)

    @ray.method
    async def update_sampling_client(self, new_client: Any):
        await self.sampling_client_manager.update_client(new_client)

    def _get_latest_model(self) -> TinkerModel:
        return TinkerModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
            renderer=self.renderer,
        )

    def uniform_reward(self, rewards: list[float]) -> bool:
        if not rewards:
            return True
        return all(r == rewards[0] for r in rewards)


async def study_task_manager_from_state_handle(
    state: ActorHandle[UniRLState],
) -> StudyTaskContext:
    ret = await state.get_next_study_task.remote()
    register = state.register_study_result.remote
    return StudyTaskContext(
        task=ret,
        register=register,
    )
