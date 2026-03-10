import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Self, Sequence

import numpy as np
import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from dotenv import load_dotenv
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
from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
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
from adapter_agent.util.exception import CodingEnvironmentError
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
    sft_batch_size: int
    rl_group_size: int
    sft_store: CountDownStore[SFTSample]
    rl_store: CountDownStore[RLSample]
    renderer: Renderer

    async def get_batch(self) -> TinkerBatch | None:
        batch_type = self.choose_available_batch_type()
        if batch_type == "Study":
            items = self.sft_store.get_batch(self.sft_batch_size)
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
                batch_type="Practice",
            )
        else:
            return None

    def choose_available_batch_type(self) -> BatchType | None:
        availables: list[BatchType] = []

        if len(self.sft_store) >= self.sft_batch_size:
            availables.append("Study")
        if len(self.rl_store) >= self.rl_group_size:
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
        sft_batch_size: int,
        rl_group_size: int,
        max_sft_reuse: int,
        max_rl_reuse: int,
        renderer: Renderer,
    ) -> Self:
        return cls(
            sft_batch_size=sft_batch_size,
            rl_group_size=rl_group_size,
            sft_store=CountDownStore({}, init_count=max_sft_reuse),
            rl_store=CountDownStore({}, init_count=max_rl_reuse),
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
    study_task_queue: TaskQueue[Counted[Task]]
    practice_task_queue: TaskQueue[Task]
    buffer: HybridReplayBuffer
    litellm_model_name: str
    renderer: Renderer
    sampling_client_manager: SharedSamplingClient
    metric_manager: MetricManager
    task_verifier: TaskVerifier
    library_name: str
    max_study_retry: int

    def get_latest_model(self) -> TinkerModel:
        current_model = TinkerModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
            renderer=self.renderer,
        )
        return current_model

    def is_finished(self) -> bool:
        return self.study_task_queue.is_done() and self.practice_task_queue.is_done()

    async def register_study_group(
        self,
        worker_id: int,
        items: list[RewireSessionResult],
        counted_task: Counted[Task],
    ):
        analyzer = Analyzer(model=self.get_latest_model())
        for item in items:
            metrics = conclusion_to_metrics(item.conclusion)
            self.metric_manager.register_study_metrics(metrics)
            if isinstance(item, RewireSessionResultSuccess):
                self.buffer.sft_store.put(
                    SFTSample(task=item.task, messages=item.rewired)
                )
                # self.buffer.sft_store.put(
                #     SFTSample(task=item.task, messages=item.trials)
                # )
        task = counted_task.item

        num_success = sum(
            1 for ret in items if isinstance(ret, RewireSessionResultSuccess)
        )

        if num_success > 0:
            logger.info(
                f"Study worker {worker_id} produced {num_success}/{len(items)} successful trajectory"
            )
        else:
            remaining_count = counted_task.count - 1
            if remaining_count == 0:
                logger.info(
                    f"Study worker {worker_id} produced {num_success}/{len(items)} successful trajectory. Task failed too many times, giving up without generating subtasks."
                )
                return

            await self.study_task_queue.put(
                item=Counted(count=remaining_count, item=task)
            )
            logger.info(
                f"Study worker {worker_id} produced {num_success}/{len(items)} successful trajectory. Added retry task. Remaining count: {remaining_count}"
            )

            trials_list = [
                ret.trials
                for ret in items
                if isinstance(ret, RewireSessionResultNormal)
            ]
            if len(trials_list) > 0:
                trials = trials_list[0]
                try:
                    if random.random() < self.newitem_probability("Study"):
                        subtask = await analyzer.analyze_trajectory(trials)
                        try:
                            verification_result = await self.task_verifier.verify_task(
                                task=subtask, library_name=self.library_name
                            )
                            if verification_result.output_type != "success":
                                logger.warning(
                                    f"Study worker {worker_id} discarded generated subtask because it failed verification: {verification_result.output_type}. Task: {subtask.instruction}"
                                )
                            else:
                                await self.study_task_queue.put(
                                    item=Counted(
                                        count=self.max_study_retry, item=subtask
                                    )
                                )
                                logger.info(
                                    f"Study worker {worker_id} produced failed trajectory. Added subtask `{subtask.instruction if subtask else None}`"
                                )
                        except Exception as ve:
                            logger.warning(
                                f"Study worker {worker_id} discarded generated subtask due to verification exception: {ve}"
                            )
                except Exception as e:
                    logger.debug(
                        f"Study worker {worker_id} failed to analyze trajectory: {e}."
                    )

    def uniform_reward(self, rewards: list[float]) -> bool:
        return all(r == rewards[0] for r in rewards)

    async def register_practice_group(
        self, worker_id: int, items: list[SolveVerifyTinkerSingleTurnResult]
    ):
        rewards = [item.reward for item in items]
        task = items[0].env_state.task

        for item in items:
            self.metric_manager.register_practice_metrics(
                conclusion_to_metrics(item.conclusion)
            )

        success_counts = sum([int(ret.is_success()) for ret in items])

        if success_counts == 0:
            await self.study_task_queue.put(
                Counted(count=self.max_study_retry, item=task)
            )
            logger.info(
                f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, putting it to study queue."
            )
            return

        elif success_counts == len(items):
            logger.info(
                f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, removing the task from the queue"
            )
            return

        else:
            await self.practice_task_queue.put(task)
            logger.info(
                f"Practice worker {worker_id} generated {success_counts}/{len(items)} successful samples, putting it back to practice queue."
            )

        if self.uniform_reward(rewards):
            return

        rl_sample = RLSample(
            task=task,
            group=TrajectoryGroup(
                trajectories_G=[item.trajectory for item in items],
                final_rewards_G=rewards,
                metrics_G=[],
            ),
        )
        if not rl_sample.is_uniform_reward():
            self.buffer.rl_store.put(rl_sample)

    async def get_batch(self) -> TinkerBatch | None:
        return await self.buffer.get_batch()

    def newitem_probability(self, batch_type: BatchType) -> float:
        if batch_type == "Study":
            queue = self.study_task_queue
        elif batch_type == "Practice":
            queue = self.practice_task_queue
        else:
            raise ValueError("Unexpected batch type")

        if queue.maxsize == 0:
            return 1.0
        qsize = queue.qsize()
        return (1 + np.cos(np.pi * qsize / queue.maxsize)) / 2


class StudyRolloutParams(BaseModel):
    qwen_no_think: bool
    rollouts_per_task: int
    max_study_retry: int


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
    initial_model = state.get_latest_model()

    while not state.is_finished():
        latest_model = state.get_latest_model()
        async with state.study_task_queue.get_item_manager() as counted_task:
            task = counted_task.item
            logger.debug(
                f"Study worker {worker_id} got a task. Remaining study tasks: {state.study_task_queue.qsize()}"
            )
            rets = await asyncio.gather(
                *[
                    ss_solve(
                        solver_model=latest_model,
                        verifier=verifier,
                        rewirer_model=initial_model,
                        task=task,
                        max_turns=env_params.max_turns,
                        qwen_no_think=study_rollout_params.qwen_no_think,
                    )
                    for _ in range(study_rollout_params.rollouts_per_task)
                ]
            )
            await state.register_study_group(
                worker_id=worker_id,
                items=rets,
                counted_task=counted_task,
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
            results = await asyncio.gather(
                *[
                    solve_verify_tinker_single_turn(
                        solver_model=latest_model,
                        env_state=env_state,
                        verifier=verifier,
                    )
                    for _ in range(practice_rollout_params.rollouts_per_task)
                ],
                return_exceptions=True,
            )
            Exception
            rets = []
            for r in results:
                if isinstance(r, CodingEnvironmentError):
                    logger.warning(
                        f"Practice worker {worker_id} encountered EnvironmentError: {r}. Excluding from results."
                    )
                elif isinstance(r, BaseException):
                    raise r
                else:
                    rets.append(r)
            if not rets:
                # If all rollouts failed with EnvironmentError, discard the task
                continue
            await state.register_practice_group(worker_id=worker_id, items=rets)


class UniRLTrainParams(BaseModel):
    update_freq: int
    save_freq: int
    adam_params: AdamParams
    max_sft_reuse: int
    max_rl_reuse: int


class UniRLConfig(BaseModel):
    experiment_setting: ExperimentSettings
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings
    study_rollout_params: StudyRolloutParams
    practice_rollout_params: PracticeRolloutParams
    train_params: UniRLTrainParams
    num_study_workers: int
    num_practice_workers: int
    sft_batch_size: int
    rl_group_size: int
    log_freq: int


async def train_worker(
    state: UniRLState,
    train_params: UniRLTrainParams,
    training_client: tinker.TrainingClient,
    ml_logger: MLLogger,
    experiment_setting: ExperimentSettings,
):
    logger.info("Train worker started.")
    step = 0
    tasks_to_practice: dict[str, Task] = dict()
    while not state.is_finished():
        batch = await state.get_batch()

        if batch is None:
            await asyncio.sleep(1)
            continue
        step += 1
        logger.info(
            f"Train worker step: {step}. Remaining practice tasks: {state.practice_task_queue.qsize()}"
        )
        fwd_bwd_future = await training_client.forward_backward_async(
            batch.datum,
            loss_fn=batch.loss_fn,
        )
        optim_future = await training_client.optim_step_async(train_params.adam_params)
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        if batch.batch_type == "Study":
            exhausted_tasks = {
                counted_item.item.id: counted_item.item
                for counted_item in batch.tasks.values()
                if counted_item.count == 0
            }
            tasks_to_practice.update(exhausted_tasks)
        ml_logger.log_metrics(
            {
                **fwd_bwd_result.metrics,
                "buffer_size/study": len(state.buffer.sft_store),
                "buffer_size/practice": len(state.buffer.rl_store),
            }
        )

        if step % train_params.save_freq == 0:
            logger.info(f"Saving checkpoint at train step {step}...")
            await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"step_{step}",
                log_path=str(experiment_setting.log_root()),
                loop_state={},
                kind="both",
                ttl_seconds=experiment_setting.ttl_seconds,
            )

        if step % train_params.update_freq == 0:
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            await state.sampling_client_manager.update_client(new_client)
            logger.info(
                f"Train Worker -> Latest sampling client updated. New ID: {id(await state.sampling_client_manager.get_client())}. "
                f"Buffer sizes - Study: {len(state.buffer.sft_store)}, Practice: {len(state.buffer.rl_store)}"
            )
            for task in tasks_to_practice.values():
                await state.practice_task_queue.put(task)
            tasks_to_practice.clear()


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
    logging.getLogger("adapter_agent.hierarchical.agent.rewirer").setLevel(
        logging.WARNING
    )
    logging.getLogger("adapter_agent.hierarchical.process.rewire_session").setLevel(
        logging.INFO
    )
    logging.getLogger("adapter_agent.hierarchical.process.rewire").setLevel(
        logging.WARNING
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
    load_dotenv()
    cfg = UniRLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent UniRL",
            experiment_name=f"Adapter Agent_UniRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        env_params=EnvParams(
            max_turns=10,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
            dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
        study_rollout_params=StudyRolloutParams(
            rollouts_per_task=8, qwen_no_think=True, max_study_retry=3
        ),
        practice_rollout_params=PracticeRolloutParams(rollouts_per_task=8),
        train_params=UniRLTrainParams(
            update_freq=1,
            save_freq=50,
            adam_params=AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            max_sft_reuse=20,
            max_rl_reuse=5,
        ),
        num_study_workers=32,
        num_practice_workers=32,
        sft_batch_size=128,
        rl_group_size=32,
        log_freq=50,
    )

    setup_base_loglevel()
    ml_logger = setup_logging(cfg)

    tasks = load_tasks(cfg)
    study_queue = TaskQueue.create(order="LIFO", maxsize=500)

    if study_queue.maxsize > 0 and len(tasks) > study_queue.maxsize:
        raise ValueError(
            f"Number of initial tasks ({len(tasks)}) exceeds the maximum "
            f"capacity of the study queue ({study_queue.maxsize})."
        )

    practice_queue = TaskQueue.create(order="FIFO", maxsize=0)

    for task in tasks:
        await study_queue.put(
            Counted(count=cfg.study_rollout_params.max_study_retry, item=task)
        )

    service_client, training_client = await setup_tinker_clients(
        cfg.model_loading_settings
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
    )
    task_verifier = TaskVerifier(model=get_gemini())

    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    buffer = HybridReplayBuffer.create(
        sft_batch_size=cfg.sft_batch_size,
        rl_group_size=cfg.rl_group_size,
        max_sft_reuse=cfg.train_params.max_sft_reuse,
        max_rl_reuse=cfg.train_params.max_rl_reuse,
        renderer=model.renderer,
    )

    state = UniRLState(
        study_task_queue=study_queue,
        practice_task_queue=practice_queue,
        buffer=buffer,
        litellm_model_name=model.model,
        renderer=model.renderer,
        sampling_client_manager=sampling_client_manager,
        metric_manager=MetricManager(ml_logger, [], [], log_freq=cfg.log_freq),
        task_verifier=task_verifier,
        library_name=cfg.env_params.library.name,
        max_study_retry=cfg.study_rollout_params.max_study_retry,
    )

    async with litellm_concurrent_limit(
        cfg.num_study_workers * cfg.study_rollout_params.rollouts_per_task
        + cfg.num_practice_workers * cfg.practice_rollout_params.rollouts_per_task
        + 10
    ):
        train_task = train_worker(
            state=state,
            train_params=cfg.train_params,
            training_client=training_client,
            ml_logger=ml_logger,
            experiment_setting=cfg.experiment_setting,
        )

        study_workers = [
            study_rollout(
                i,
                state,
                verifier,
                cfg.study_rollout_params,
                cfg.env_params,
            )
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
