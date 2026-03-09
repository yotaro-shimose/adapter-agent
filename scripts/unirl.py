import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Self, Sequence

import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from pydantic import BaseModel
from tinker import AdamParams, Datum
from tinker.types import LossFnType
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QASFTDataset
from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
)
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    SolveVerifyTinkerSingleTurnResult,
    solve_verify_tinker_single_turn,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.simplified_solver import conclusion_to_metrics
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.trajectory import prepare_minibatch_simplified
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


class TinkerBatch(BaseModel):
    datum: list[Datum]
    loss_fn: LossFnType
    tasks: dict[str, Counted[Task]]


type BatchType = Literal["Study", "Practice"]


@dataclass
class CountDownStore[T]:
    items: dict[str, Counted[T]]
    init_count: int

    def get_batch(self, batch_size: int) -> list[Counted[T]]:
        batch_size = min(batch_size, len(self.items))
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
    min_batch_size: int
    max_batch_size: int
    sft_store: CountDownStore[SFTSample]
    rl_store: CountDownStore[RLSample]
    renderer: Renderer

    async def get_batch(self) -> TinkerBatch | None:
        batch_type = self.choose_available_batch_type()
        if batch_type == "Study":
            items = self.sft_store.get_batch(self.max_batch_size)
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
            return TinkerBatch(datum=datum, loss_fn="cross_entropy", tasks=tasks)
        elif batch_type == "Practice":
            items = self.rl_store.get_batch(self.max_batch_size)
            traj_groups = [item.item.group for item in items]
            datum, _prepare_minibatch_metrics = await prepare_minibatch_simplified(
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
            )
        else:
            return None

    def choose_available_batch_type(self) -> BatchType | None:
        availables: list[BatchType] = []

        if len(self.sft_store) >= self.min_batch_size:
            availables.append("Study")
        if len(self.rl_store) >= self.min_batch_size:
            availables.append("Practice")

        if len(availables) == 0:
            return None

        item = random.choice(availables)
        return item

    def _remove_mask(self, datum: tinker.Datum) -> tinker.Datum:
        return tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"
            },
        )

    @classmethod
    def create(
        cls,
        min_batch_size: int,
        max_batch_size: int,
        max_reuse: int,
        renderer: Renderer,
    ) -> Self:
        return cls(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            sft_store=CountDownStore({}, init_count=max_reuse),
            rl_store=CountDownStore({}, init_count=max_reuse),
            renderer=renderer,
        )


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
        return {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}

    def with_prefix(
        self, metrics: dict[str, float], prefix: str, separator: str = "/"
    ) -> dict[str, float]:
        return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}

    def register_study_metrics(self, metrics: dict[str, float]):
        self.study_metrics_list.append(metrics)
        if len(self.study_metrics_list) % self.log_freq == 0:
            mean_metrics = self.mean_metrics(self.study_metrics_list)
            self.ml_logger.log_metrics(self.with_prefix(mean_metrics, "study"))
            self.study_metrics_list = []

    def register_practice_metrics(self, metrics: dict[str, float]):
        self.practice_metrics_list.append(metrics)
        if len(self.practice_metrics_list) % self.log_freq == 0:
            mean_metrics = self.mean_metrics(self.practice_metrics_list)
            self.ml_logger.log_metrics(self.with_prefix(mean_metrics, "practice"))
            self.practice_metrics_list = []


@dataclass
class UniRLState:
    # Model
    study_task_queue: TaskQueue[Task]
    practice_task_queue: TaskQueue[Task]
    buffer: HybridReplayBuffer
    litellm_model_name: str
    renderer: Renderer
    sampling_client_manager: SharedSamplingClient
    metric_manager: MetricManager

    def get_latest_model(self) -> TinkerModel:
        current_model = TinkerModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
            renderer=self.renderer,
        )
        return current_model

    def is_finished(self) -> bool:
        return self.study_task_queue.is_done() and self.practice_task_queue.is_done()

    def register_study_rollout(self, item: RewireSessionResult):
        metrics = conclusion_to_metrics(item.conclusion)
        self.metric_manager.register_study_metrics(metrics)
        if item.rewired:
            self.buffer.sft_store.put(SFTSample(task=item.task, messages=item.rewired))

    def uniform_reward(self, rewards: list[float]) -> bool:
        return all(r == rewards[0] for r in rewards)

    def register_practice_group(self, items: list[SolveVerifyTinkerSingleTurnResult]):
        rewards = [item.reward for item in items]
        if self.uniform_reward(rewards):
            return
        for item in items:
            self.buffer.rl_store.put(
                RLSample(
                    task=item.env_state.task,
                    group=TrajectoryGroup(
                        trajectories_G=[item.trajectory for item in items],
                        final_rewards_G=rewards,
                        metrics_G=[],
                    ),
                )
            )
            self.metric_manager.register_practice_metrics(
                conclusion_to_metrics(item.conclusion)
            )

    async def get_batch(self) -> TinkerBatch | None:
        return await self.buffer.get_batch()


class StudyRolloutParams(BaseModel):
    qwen_no_think: bool


class PracticeRolloutParams(BaseModel):
    rollouts_per_task: int


async def study_rollout(
    worker_id: int,
    state: UniRLState,
    verifier: Verifier,
    study_rollout_params: StudyRolloutParams,
    env_params: EnvParams,
):
    logger.info(f"Study worker {worker_id} started.")
    latest_model = state.get_latest_model()
    analyzer = Analyzer(model=latest_model)

    while not state.is_finished():
        async with state.study_task_queue.get_item_manager() as task:
            logger.debug(
                f"Study worker {worker_id} got a task. Remaining study tasks: {state.study_task_queue.qsize()}"
            )
            ret = await ss_solve(
                solver_model=latest_model,
                verifier=verifier,
                rewirer_model=latest_model,
                task=task,
                max_turns=env_params.max_turns,
                qwen_no_think=study_rollout_params.qwen_no_think,
            )
            state.register_study_rollout(ret)
            is_successful = ret.rewired is not None
            if is_successful:
                logger.info(f"Study worker {worker_id} produced successful trajectory")
            else:
                await state.study_task_queue.put(item=task)
                subtask = None
                if len(ret.trials) > 0:
                    try:
                        subtask = await analyzer.analyze_trajectory(ret.trials)
                        await state.study_task_queue.put(item=subtask)
                    except Exception as e:
                        logger.debug(
                            f"Study worker {worker_id} failed to analyze trajectory: {e}"
                        )
                logger.info(
                    f"Study worker {worker_id} produced failed trajectory. Added retry task and subtask `{subtask.instruction[:100] if subtask else None}`"
                )


async def practice_rollout(
    worker_id: int,
    state: UniRLState,
    verifier: Verifier,
    practice_rollout_params: PracticeRolloutParams,
):
    logger.info(f"Practice worker {worker_id} started.")

    while not state.is_finished():
        async with state.practice_task_queue.get_item_manager() as task:
            logger.debug(
                f"Practice worker {worker_id} got a task. Remaining practice tasks: {state.practice_task_queue.qsize()}"
            )
            latest_model = state.get_latest_model()
            env_state = SingleTurnEnvState.numrs2(task=task)
            rets = await asyncio.gather(
                *[
                    solve_verify_tinker_single_turn(
                        solver_model=latest_model,
                        env_state=env_state,
                        verifier=verifier,
                    )
                    for _ in range(practice_rollout_params.rollouts_per_task)
                ]
            )
            state.register_practice_group(rets)
            success_counts = sum([int(ret.is_success()) for ret in rets])

            if success_counts == 0:
                await state.study_task_queue.put(task)
                logger.info(
                    f"Practice Worker generated {success_counts}/{len(rets)} successful samples, putting it to study queue."
                )
            elif success_counts == len(rets):
                logger.info(
                    f"Practice Worker generated {success_counts}/{len(rets)} successful samples, removing the task from the queue"
                )
            else:
                await state.practice_task_queue.put(task)
                logger.info(
                    f"Practice Worker generated {success_counts}/{len(rets)} successful samples, putting it to practice queue again."
                )


class UniRLTrainParams(BaseModel):
    update_freq: int
    adam_params: AdamParams


class UniRLConfig(BaseModel):
    experiment_setting: ExperimentSettings
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings
    study_rollout_params: StudyRolloutParams
    practice_rollout_params: PracticeRolloutParams
    train_params: UniRLTrainParams
    num_study_workers: int
    num_practice_workers: int
    min_batch_size: int
    max_batch_size: int
    max_reuse: int
    log_freq: int


async def train_worker(
    state: UniRLState,
    train_params: UniRLTrainParams,
    training_client: tinker.TrainingClient,
    ml_logger: MLLogger,
):
    logger.info("Train worker started.")
    step = 0
    tasks_to_practice: dict[str, Task] = dict()
    while not state.is_finished():
        batch = await state.get_batch()
        logger.info(
            f"Train worker step: {step}. Remaining practice tasks: {state.practice_task_queue.qsize()}"
        )
        if batch is None:
            await asyncio.sleep(1)
            continue
        fwd_bwd_future = await training_client.forward_backward_async(
            batch.datum,
            loss_fn=batch.loss_fn,
        )
        optim_future = await training_client.optim_step_async(train_params.adam_params)
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        exhausted_tasks = {
            counted_item.item.id: counted_item.item
            for counted_item in batch.tasks.values()
            if counted_item.count == 0
        }
        tasks_to_practice.update(exhausted_tasks)
        step += 1
        ml_logger.log_metrics(fwd_bwd_result.metrics)
        if step % train_params.update_freq == 0:
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            await state.sampling_client_manager.update_client(new_client)
            logger.info(
                f"Train Worker -> Latest sampling client updated. New ID: {id(await state.sampling_client_manager.get_client())}"
            )
            for task in tasks_to_practice.values():
                await state.practice_task_queue.put(task)
            tasks_to_practice.clear()


class _SuppressExtensionWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "train_on_what=ALL_ASSISTANT_MESSAGES" not in record.getMessage()


def setup_logging(cfg: UniRLConfig) -> MLLogger:
    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    setup_base_loglevel()
    ml_logger = ml_log.setup_logging(
        log_dir=str(log_root),
        wandb_project=cfg.experiment_setting.wandb_project,
        config=cfg,
    )
    logger.setLevel(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    logging.getLogger("tinker_cookbook.renderers.base").addFilter(
        _SuppressExtensionWarning()
    )

    logging.getLogger("adapter_agent.hierarchical.agent.rewirer").setLevel(
        logging.WARNING
    )
    logging.getLogger("adapter_agent.hierarchical.process.rewire_session").setLevel(
        logging.INFO
    )
    logging.getLogger("adapter_agent.hierarchical.process.rewire").setLevel(
        logging.INFO
    )
    logging.getLogger(
        "adapter_agent.hierarchical.process.rewire_session_single_turn"
    ).setLevel(logging.INFO)

    return ml_logger


def load_tasks(cfg: UniRLConfig) -> list[Task]:
    # Load questions
    logger.info(f"Loading questions from {cfg.env_params.dataset_path}...")
    qas_data_raw = QASFTDataset.model_validate_json(
        cfg.env_params.dataset_path.read_text()
    )
    # Parse into QA objects
    qas_data = qas_data_raw.shuffled()
    logger.info(f"Loaded {len(qas_data)} questions.")

    tasks = [Task.from_instruction(qa.question) for qa in qas_data]
    return tasks


def setup_agents(
    service_client: tinker.ServiceClient,
    model_loading_settings: ModelLoadingSettings,
    rust_doc_analyzer: RustDocAnalyzer,
):
    logger.info(
        f"Setting up model {model_loading_settings.model_name} from {model_loading_settings.resume_sampler_path}..."
    )
    model, tokenizer, renderer = setup_tinkermodel(
        service_client,
        model_loading_settings.model_name,
        model_loading_settings.resume_sampler_path,
    )
    verifier_model = get_gemini()
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    return model, verifier


async def setup_tinker_clients(
    model_loading_settings: ModelLoadingSettings,
) -> tuple[tinker.ServiceClient, tinker.TrainingClient]:
    service_client = tinker.ServiceClient()
    capa = await service_client.get_server_capabilities_async()
    logger.debug(f"Server capabilities: {capa}")
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_loading_settings.model_name,
        rank=model_loading_settings.lora_rank,
    )
    if model_loading_settings.resume_trainer_path:
        logger.info(
            f"Loading trainer from {model_loading_settings.resume_trainer_path}..."
        )
        await training_client.load_state_async(
            path=model_loading_settings.resume_trainer_path
        )
    else:
        logger.warning("No trainer path provided, starting from scratch.")

    return service_client, training_client


async def main():
    cfg = UniRLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent UniRL",
            experiment_name=f"Adapter Agent_UniRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        env_params=EnvParams(
            max_turns=2,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
            dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
        study_rollout_params=StudyRolloutParams(max_rewire=2, qwen_no_think=True),
        practice_rollout_params=PracticeRolloutParams(rollouts_per_task=4),
        train_params=UniRLTrainParams(
            update_freq=1,
            adam_params=AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
        ),
        num_study_workers=128,
        num_practice_workers=8,
        min_batch_size=128,
        max_batch_size=128,
        max_reuse=20,
        log_freq=50,
    )

    setup_base_loglevel()
    ml_logger = setup_logging(cfg)

    tasks = load_tasks(cfg)
    study_queue = TaskQueue.create(order="LIFO")
    practice_queue = TaskQueue.create(order="FIFO")

    for task in tasks:
        await study_queue.put(task)

    service_client, training_client = await setup_tinker_clients(
        cfg.model_loading_settings
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
    )

    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    buffer = HybridReplayBuffer(
        min_batch_size=cfg.min_batch_size,
        max_batch_size=cfg.max_batch_size,
        sft_store=CountDownStore({}, init_count=cfg.max_reuse),
        rl_store=CountDownStore({}, init_count=cfg.max_reuse),
        renderer=model.renderer,
    )

    state = UniRLState(
        study_task_queue=study_queue,
        practice_task_queue=practice_queue,
        buffer=buffer,
        litellm_model_name=model.model,
        renderer=model.renderer,
        sampling_client_manager=sampling_client_manager,
        metric_manager=MetricManager(ml_logger, [], [], log_freq=10),
    )

    async with litellm_concurrent_limit(
        cfg.num_study_workers
        + cfg.num_practice_workers * cfg.practice_rollout_params.rollouts_per_task
        + 10
    ):
        train_task = train_worker(
            state=state,
            train_params=cfg.train_params,
            training_client=training_client,
            ml_logger=ml_logger,
        )

        study_workers = [
            study_rollout(i, state, verifier, cfg.study_rollout_params, cfg.env_params)
            for i in range(cfg.num_study_workers)
        ]

        practice_workers = [
            practice_rollout(i, state, verifier, cfg.practice_rollout_params)
            for i in range(cfg.num_practice_workers)
        ]

        await asyncio.gather(
            train_task,
            *study_workers,
            *practice_workers,
        )

    await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(cfg.experiment_setting.log_root()),
        loop_state={},
        kind="both",
        ttl_seconds=cfg.experiment_setting.ttl_seconds,
    )


if __name__ == "__main__":
    asyncio.run(main())
