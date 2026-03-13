import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import cast

from oai_utils.tinker import TinkerModel

from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.unirl_state import CountedTask

# Silence Ray logs
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_COLOR_PREFIX"] = "0"
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
from asyncio import timeout
from dataclasses import dataclass, field

import ray
import tinker
import tinker_cookbook.checkpoint_utils
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import setup_tinkermodel
from ray.actor import ActorHandle
from tinker import AdamParams

from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    solve_verify_tinker_single_turn,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.unirl_state import (
    HybridReplayBuffer,
    PracticeRolloutParams,
    StudyRolloutParams,
    UniRLConfig,
    UniRLState,
    UniRLTrainParams,
)
from adapter_agent.util.exception import CodingEnvironmentError
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.task_queue import TaskQueue

logger = logging.getLogger(__name__)


@ray.remote
def study_rollout(
    worker_id: int,
    state: ActorHandle[UniRLState],
    verifier: Verifier,
    study_rollout_params: StudyRolloutParams,
    env_params: EnvParams,
):
    setup_rollout_logging()
    asyncio.run(
        _study_rollout(
            worker_id,
            state,
            verifier,
            study_rollout_params,
            env_params,
        )
    )


async def _study_rollout(
    worker_id: int,
    state: ActorHandle[UniRLState],
    verifier: Verifier,
    study_rollout_params: StudyRolloutParams,
    env_params: EnvParams,
):
    logger.info(f"Study worker {worker_id} started.")
    initial_model: TinkerModel = await state.get_latest_model.remote()
    initial_model.register_tinkerllm_to_litellm()

    while True:
        latest_model = await state.get_latest_model.remote()
        try:
            async with timeout(10.0):
                counted_task = await state.get_next_study_task.remote()
        except asyncio.TimeoutError as e:
            logger.warning(f"Study worker {worker_id} timed out getting a task: {e}")
            continue
        task = counted_task.item
        logger.debug(f"Study worker {worker_id} got a task. ")
        rets = await asyncio.gather(
            *[
                ss_solve(
                    solver_model=latest_model,
                    verifier=verifier,
                    rewirer_model=initial_model,
                    task=task,
                    max_turns=env_params.max_turns,
                    qwen_no_think=study_rollout_params.qwen_no_think,
                    runtime_settings=env_params.runtime_settings,
                )
                for _ in range(study_rollout_params.rollouts_per_task)
            ]
        )
        state.register_study_group.remote(
            worker_id,
            rets,
            counted_task,
        )


@ray.remote
def practice_rollout(
    worker_id: int,
    state: ActorHandle[UniRLState],
    verifier: Verifier,
    practice_rollout_params: PracticeRolloutParams,
    env_params: EnvParams,
):
    setup_rollout_logging()
    asyncio.run(
        _practice_rollout(
            worker_id,
            state,
            verifier,
            practice_rollout_params,
            env_params,
        )
    )


async def _practice_rollout(
    worker_id: int,
    state: ActorHandle[UniRLState],
    verifier: Verifier,
    practice_rollout_params: PracticeRolloutParams,
    env_params: EnvParams,
):
    logger.info(f"Practice worker {worker_id} started.")

    while True:
        task = await state.get_next_practice_task.remote()
        logger.debug(f"Practice worker {worker_id} got a task.")
        latest_model = await state.get_latest_model.remote()
        env_state = SingleTurnEnvState.numrs2(task=task)
        results = await asyncio.gather(
            *[
                solve_verify_tinker_single_turn(
                    solver_model=latest_model,
                    env_state=env_state,
                    verifier=verifier,
                    runtime_settings=env_params.runtime_settings,
                )
                for _ in range(practice_rollout_params.rollouts_per_task)
            ],
            return_exceptions=True,
        )
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

        await state.register_practice_group.remote(worker_id, rets)


@dataclass
class TrainWorker:
    state: ActorHandle[UniRLState]
    params: UniRLTrainParams
    experiment_setting: ExperimentSettings
    model_loading_settings: ModelLoadingSettings
    training_client: tinker.TrainingClient = field(init=False)

    def __post_init__(self):
        logger.info("Train worker init.")
        setup_base_loglevel()
        logging.basicConfig(level=logging.DEBUG)
        self.training_client = setup_training_client(self.model_loading_settings)

    async def run(self):
        logger.info("Train worker main loopstarted.")
        step = 0
        tasks_to_practice: dict[str, Task] = dict()
        pending_fwd_bwd_future = None
        pending_optim_future = None
        while not await self.state.is_finished.remote():
            # get_batch is now sync on the actor
            batch = await self.state.get_batch.remote()

            if batch is None:
                await asyncio.sleep(1)
                continue
            step += 1

            logger.info(f"Train worker step: {step}. ")

            fwd_bwd_future = await self.training_client.forward_backward_async(
                batch.datum,
                loss_fn=batch.loss_fn,
            )
            optim_future = await self.training_client.optim_step_async(
                self.params.adam_params
            )
            if pending_fwd_bwd_future is not None and pending_optim_future is not None:
                fwd_bwd_result = await pending_fwd_bwd_future.result_async()
                await pending_optim_future.result_async()
                await self.state.log_train_metrics.remote(
                    {
                        **fwd_bwd_result.metrics,
                    }
                )
            pending_fwd_bwd_future = fwd_bwd_future
            pending_optim_future = optim_future

            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()
            if batch.batch_type == "Study":
                exhausted_tasks = {
                    counted_item.item.id: counted_item.item
                    for counted_item in batch.tasks.values()
                    if counted_item.count == 0
                }
                tasks_to_practice.update(exhausted_tasks)

            # Save checkpoint
            if step % self.params.save_freq == 0:
                logger.info(f"Saving checkpoint at train step {step}...")
                await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
                    training_client=self.training_client,
                    name=f"step_{step}",
                    log_path=str(self.experiment_setting.log_root()),
                    loop_state={},
                    kind="both",
                    ttl_seconds=self.experiment_setting.ttl_seconds,
                )

            # Update latest model for sampling
            if step % self.params.update_freq == 0:
                new_client = await self.training_client.save_weights_and_get_sampling_client_async()
                await self.state.update_sampling_client.remote(new_client)

                logger.info("Train Worker -> Latest sampling client updated.")
                for task in tasks_to_practice.values():
                    await self.state.add_to_practice_queue.remote(task)
                tasks_to_practice.clear()


def setup_rollout_logging():
    logging.basicConfig(level=logging.INFO)
    setup_base_loglevel()
    logger.setLevel(level=logging.DEBUG)
    logging.getLogger("adapter_agent.hierarchical.process.rewire").setLevel(
        logging.WARNING
    )
    logging.getLogger(
        "adapter_agent.hierarchical.process.rewire_session_single_turn"
    ).setLevel(logging.INFO)


# def load_tasks(cfg: UniRLConfig) -> list[Task]:
#     # Load questions
#     logger.info(f"Loading questions from {cfg.env_params.dataset_path}...")
#     qas_data_raw = QASFTDataset.model_validate_json(
#         cfg.env_params.dataset_path.read_text()
#     )
#     # Parse into QA objects
#     qas_data = qas_data_raw.shuffled()
#     logger.info(f"Loaded {len(qas_data)} questions.")
#
#     tasks = [Task.from_instruction(qa.question) for qa in qas_data]
#     return tasks


def load_gh_archive() -> list[Task]:
    import polars as pl

    path = Path("data/easy_benchmark_verified.csv")
    logger.info(f"Loading verified tasks from {path}...")
    df = pl.read_csv(path)
    tasks = [
        Task.from_instruction(row["problem_statement"])
        for row in df.iter_rows(named=True)
    ]
    logger.info(f"Loaded {len(tasks)} verified tasks.")
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
        model_loading_settings.model_name,
        model_loading_settings.resume_sampler_path,
        service_client,
    )
    verifier_model = get_gemini()
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    return model, verifier


def setup_training_client(
    model_loading_settings: ModelLoadingSettings,
) -> tinker.TrainingClient:
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_loading_settings.model_name,
        rank=model_loading_settings.lora_rank,
    )
    if model_loading_settings.resume_trainer_path:
        logger.info(
            f"Loading trainer from {model_loading_settings.resume_trainer_path}..."
        )
        training_client.load_state(path=model_loading_settings.resume_trainer_path)
    else:
        logger.warning("No trainer path provided, starting from scratch.")

    return training_client


async def main():
    load_dotenv()
    # Suppress ray console output
    ray.init(
        configure_logging=True,
        logging_config=ray.LoggingConfig(
            log_level="INFO", additional_log_standard_attrs=["name"]
        ),
    )
    cfg = UniRLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent UniRL",
            experiment_name=f"Adapter Agent_UniRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        env_params=EnvParams(
            max_turns=10,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
            # runtime_settings=RuntimeSettings(
            #     type="docker",
            #     image_uri="coder-mcp-numrs2:latest",
            # ),
            runtime_settings=RuntimeSettings(
                type="cloudrun",
                image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2",
            ),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
        study_queue_size=500,
        study_rollout_params=StudyRolloutParams(
            rollouts_per_task=6, qwen_no_think=True, max_study_retry=5
        ),
        practice_rollout_params=PracticeRolloutParams(rollouts_per_task=6),
        train_params=UniRLTrainParams(
            update_freq=1,
            save_freq=50,
            adam_params=AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            max_sft_reuse=15,  # TODO: tune
            max_rl_reuse=5,
            min_sft_batch_size=32,  # TODO: tune
            max_sft_batch_size=128,
            rl_group_size=32,
        ),
        num_study_workers=20,
        num_practice_workers=8,
        log_freq=50,
    )

    setup_rollout_logging()

    tasks = load_gh_archive()
    study_queue = TaskQueue[CountedTask].create(
        order="LIFO", maxsize=cfg.study_queue_size
    )

    if study_queue.maxsize > 0 and len(tasks) > study_queue.maxsize:
        raise ValueError(
            f"Number of initial tasks ({len(tasks)}) exceeds the maximum "
            f"capacity of the study queue ({study_queue.maxsize})."
        )

    practice_queue = TaskQueue.create(order="FIFO", maxsize=0)

    for task in tasks:
        # TaskQueue.put_nowait is sync
        study_queue.put_nowait(
            CountedTask(count=None, item=task)  # Initial TaskはNone
        )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    service_client = tinker.ServiceClient()
    model, verifier = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
    )
    task_verifier = TaskVerifier(model=get_gemini())

    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    buffer = HybridReplayBuffer.create(
        min_sft_batch_size=cfg.train_params.min_sft_batch_size,
        max_sft_batch_size=cfg.train_params.max_sft_batch_size,
        rl_group_size=cfg.train_params.rl_group_size,
        max_sft_reuse=cfg.train_params.max_sft_reuse,
        max_rl_reuse=cfg.train_params.max_rl_reuse,
        renderer=model.renderer,
    )

    UniRLStateActor = ray.remote(UniRLState)
    state: ActorHandle[UniRLState] = cast(
        ActorHandle[UniRLState],
        UniRLStateActor.remote(
            study_task_queue=study_queue,
            practice_task_queue=practice_queue,
            buffer=buffer,
            litellm_model_name=model.model,
            renderer=model.renderer,
            sampling_client_manager=sampling_client_manager,
            task_verifier=task_verifier,
            library_name=cfg.env_params.library.name,
            max_study_retry=cfg.study_rollout_params.max_study_retry,
            cfg=cfg,
        ),
    )

    TrainWorkerActorClass = ray.remote(TrainWorker)
    train_worker_actor: ActorHandle[TrainWorker] = cast(
        ActorHandle[TrainWorker],
        TrainWorkerActorClass.remote(
            state,
            cfg.train_params,
            cfg.experiment_setting,
            cfg.model_loading_settings,
        ),
    )
    async with litellm_concurrent_limit(
        cfg.num_study_workers * cfg.study_rollout_params.rollouts_per_task
        + cfg.num_practice_workers * cfg.practice_rollout_params.rollouts_per_task
        + 10
    ):
        train_task = train_worker_actor.run.remote()
        # Study Worker の起動
        study_workers = []
        for i in range(cfg.num_study_workers):
            # ここで少し待つ (例: 1秒おきに1つずつ起動)ことによって初期起動スパイクをさける
            await asyncio.sleep(1.0)

            worker = study_rollout.remote(
                i,
                state,
                verifier,
                cfg.study_rollout_params,
                cfg.env_params,
            )
            study_workers.append(worker)
            logger.info(f"Launched Study Worker {i}...")

        # Practice Worker の起動
        practice_workers = []
        for i in range(cfg.num_practice_workers):
            await asyncio.sleep(1.0)

            worker = practice_rollout.remote(
                i, state, verifier, cfg.practice_rollout_params, cfg.env_params
            )
            practice_workers.append(worker)
            logger.info(f"Launched Practice Worker {i}...")

        ray.get([train_task, *study_workers, *practice_workers])


if __name__ == "__main__":
    asyncio.run(main())
