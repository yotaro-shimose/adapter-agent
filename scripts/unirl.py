import asyncio
import collections
import logging
import random
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal, Self

import litellm
import numpy as np
import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from pydantic import BaseModel
from tinker import AdamParams, APIFuture, Datum, TrainingClient
from tinker.types import LossFnType
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import (
    QA,
    PydanticTinkerBaseMessage,
    QASFTDataset,
    TinkerMessagesDataset,
    TinkerMessageTrajectory,
)
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
)
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    SolveVerifyTinkerSingleTurnResult,
    rewire_session_single_turn,
    solve_verify_tinker_single_turn,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.trajectory import prepare_minibatch_simplified
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.task_queue import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class SFTSample:
    messages: list[TinkerMessage]


@dataclass
class RLSample:
    group: TrajectoryGroup


class TinkerBatch(BaseModel):
    datum: list[Datum]
    loss_fn: LossFnType


type BatchType = Literal["Study", "Practice"]


@dataclass
class Counted[T]:
    count: int
    item: T


@dataclass
class CountDownStore[T]:
    items: dict[str, Counted[T]]
    init_count: int

    def get_batch(self, batch_size: int) -> list[T]:
        batch_size = min(batch_size, len(self.items))
        keys = random.choices(list(self.items.keys()), k=batch_size)
        items: list[T] = []
        for key in keys:
            item = self.items[key]
            items.append(item.item)
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
    max_reuse: int

    async def get_batch(self) -> TinkerBatch | None:
        batch_type = self.choose_available_batch_type()
        if batch_type == "Study":
            items = self.sft_store.get_batch(self.max_batch_size)
            datum = [
                conversation_to_datum(
                    item.messages,
                    self.renderer,
                    max_length=None,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                for item in items
            ]
            return TinkerBatch(datum=datum, loss_fn="cross_entropy")
        elif batch_type == "Practice":
            items = self.rl_store.get_batch(self.max_batch_size)
            traj_groups = [item.group for item in items]
            datum, _prepare_minibatch_metrics = await prepare_minibatch_simplified(
                traj_groups
            )
            datum_cleaned = [self._remove_mask(d) for d in datum]
            return TinkerBatch(
                datum=datum_cleaned,
                loss_fn="ppo",
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


@dataclass
class UniRLState:
    # Model
    study_task_queue: TaskQueue[Task]
    practice_task_queue: TaskQueue[Task]
    buffer: HybridReplayBuffer
    litellm_model_name: str
    renderer: Renderer
    sampling_client_manager: SharedSamplingClient

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
        if item.rewired:
            self.buffer.sft_store.put(SFTSample(messages=item.rewired.messages))

    def uniform_reward(self, rewards: list[float]) -> bool:
        return all(r == rewards[0] for r in rewards)

    def register_practice_group(self, items: list[SolveVerifyTinkerSingleTurnResult]):
        rewards = [item.reward for item in items]
        if self.uniform_reward(rewards):
            return
        for item in items:
            self.buffer.rl_store.put(
                RLSample(
                    group=TrajectoryGroup(
                        trajectories_G=[item.trajectory for item in items],
                        final_rewards_G=rewards,
                        metrics_G=[item.metrics for item in items],
                    )
                )
            )

    async def get_batch(self) -> TinkerBatch | None:
        return await self.buffer.get_batch()


class StudyRolloutParams(BaseModel):
    max_rewire: int


class PracticeRolloutParams(BaseModel):
    rollouts_per_task: int


async def study_rollout(
    worker_id: int,
    state: UniRLState,
    verifier: Verifier,
    study_rollout_params: StudyRolloutParams,
):
    logger.info(f"Rollout worker {worker_id} started.")

    while not state.is_finished():
        async with state.study_task_queue.get_item_manager() as task:
            latest_model = state.get_latest_model()
            ret = await rewire_session_single_turn(
                solver_model=latest_model,
                verifier=verifier,
                rewirer_model=latest_model,
                task=task,
                max_rewire=study_rollout_params.max_rewire,
            )
            state.register_study_rollout(ret)
            is_successful = ret.rewired is not None
            if is_successful:
                logger.info(f"Study worker {worker_id} produced successful trajectory")
            else:
                logger.info(f"Study worker {worker_id} produced failed trajectory")
                # TODO: introduce decomposing logic


async def practice_rollout(
    worker_id: int,
    state: UniRLState,
    verifier: Verifier,
    practice_rollout_params: PracticeRolloutParams,
):
    logger.info(f"Rollout sorker {worker_id} started.")

    while not state.is_finished():
        async with state.practice_task_queue.get_item_manager() as task:
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
            logger.info(
                f"Practice Worker generated {success_counts}/{len(rets)} successful samples"
            )


class UniRLTrainParams(BaseModel):
    sleep_sec: int = 1
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


async def train_worker(
    state: UniRLState,
    train_params: UniRLTrainParams,
    training_client: tinker.TrainingClient,
    ml_logger: MLLogger,
):
    logger.info("Train worker started.")
    step = 0
    while not state.is_finished():
        batch = await state.get_batch()
        if batch is None:
            await asyncio.sleep(train_params.sleep_sec)
            continue
        fwd_bwd_future = await training_client.forward_backward_async(
            batch.datum,
            loss_fn=batch.loss_fn,
        )
        optim_future = await training_client.optim_step_async(train_params.adam_params)
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        step += 1
        ml_logger.log_metrics(fwd_bwd_result.metrics)
        if step % train_params.update_freq == 0:
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            await state.sampling_client_manager.update_client(new_client)
            logger.info(
                f"Train Worker -> Latest sampling client updated. New ID: {id(state.sampling_client_manager.get_client())}"
            )


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
    logging.basicConfig(level=logging.INFO)

    logging.getLogger("tinker_cookbook.renderers.base").addFilter(
        _SuppressExtensionWarning()
    )

    logging.getLogger("adapter_agent.hierarchical.agent.rewirer").setLevel(logging.INFO)
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
    verifier_model = litellm.get_gemini() if hasattr(litellm, "get_gemini") else None
    if verifier_model is None:
        from adapter_agent.model_helper import get_gemini

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
            dataset_path=Path(
                "logs/Adapter_Agent/Adapter Agent_20260307_010745/sft_trajectories.json"
            ),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
        study_rollout_params=StudyRolloutParams(max_rewire=2),
        practice_rollout_params=PracticeRolloutParams(rollouts_per_task=4),
        train_params=UniRLTrainParams(
            sleep_sec=1,
            update_freq=32,
            adam_params=AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
        ),
        num_study_workers=32,
        num_practice_workers=32,
        min_batch_size=64,
        max_batch_size=256,
        max_reuse=4,
    )

    setup_base_loglevel()
    ml_logger = setup_logging(cfg)

    service_client, training_client = await setup_tinker_clients(
        cfg.model_loading_settings
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
    )

    tasks = load_tasks(cfg)
    study_queue = TaskQueue[Task]()
    practice_queue = TaskQueue[Task]()

    # TODO: remove or adjust test logic
    for task in tasks * 2:
        await study_queue.put(task)

    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    buffer = HybridReplayBuffer(
        min_batch_size=cfg.min_batch_size,
        max_batch_size=cfg.max_batch_size,
        sft_store=CountDownStore({}, init_count=cfg.max_reuse),
        rl_store=CountDownStore({}, init_count=cfg.max_reuse),
        renderer=model.renderer,
        max_reuse=cfg.max_reuse,
    )

    state = UniRLState(
        study_task_queue=study_queue,
        practice_task_queue=practice_queue,
        buffer=buffer,
        litellm_model_name=model.model,
        renderer=model.renderer,
        sampling_client_manager=sampling_client_manager,
    )

    async with litellm_concurrent_limit(
        cfg.num_study_workers + cfg.num_practice_workers + 10
    ):
        train_task = train_worker(
            state=state,
            train_params=cfg.train_params,
            training_client=training_client,
            ml_logger=ml_logger,
        )

        study_workers = [
            study_rollout(i, state, verifier, cfg.study_rollout_params)
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
