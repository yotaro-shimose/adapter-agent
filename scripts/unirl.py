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
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.trajectory import prepare_minibatch_simplified
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.task_queue import TaskQueue

logger = logging.getLogger(__name__)


class SFTSample(BaseModel):
    messages: list[TinkerMessage]


class RLSample(BaseModel):
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

    def put(self, item: T, init_count: int):
        key = str(uuid.uuid4())
        self.items[key] = Counted(item=item, count=init_count)

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
        batch_type = self.choose_type()
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

    def choose_type(self) -> BatchType | None:
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

    def register_study_rollout(self, item: RewireSessionResult): ...

    def register_practice_group(
        self, items: list[SolveVerifyTinkerSingleTurnResult]
    ): ...

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


async def train_worker(
    state: UniRLState,
    train_params: UniRLTrainParams,
    training_client: tinker.TrainingClient,
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
        await fwd_bwd_future.result_async()
        await optim_future.result_async()
        step += 1
        if step % train_params.update_freq == 0:
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            await state.sampling_client_manager.update_client(new_client)
            logger.info(
                f"Train Worker -> Latest sampling client updated. New ID: {id(state.sampling_client_manager.get_client())}"
            )
