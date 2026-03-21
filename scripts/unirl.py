import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import ray
import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import setup_tinkermodel
from ray.actor import ActorHandle
from tinker import AdamParams
from tinker_cookbook.utils import ml_log

from adapter_agent.data import QASFTDataset
from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
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
    Counted,
    HybridReplayBuffer,
    MetricManager,
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
        try:
            counted_task = state.get_next_study_task()
        except IndexError:
            await asyncio.sleep(1)
            continue

        task = counted_task.item
        logger.debug(f"Study worker {worker_id} got a task. ")
        rets = await asyncio.gather(
            *[
                ss_solve_verify(
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
            worker_id,
            rets,
            counted_task,
        )


async def practice_rollout(
    worker_id: int,
    state: UniRLState,
    verifier: Verifier,
    practice_rollout_params: PracticeRolloutParams,
):
    logger.info(f"Practice worker {worker_id} started.")

    while not state.is_finished():
        try:
            task = state.get_next_practice_task()
        except IndexError:
            await asyncio.sleep(1)
            continue

        logger.debug(f"Practice worker {worker_id} got a task. ")
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

        state.register_practice_group(worker_id, rets)


async def train_worker(
    state: UniRLState,
    params: UniRLTrainParams,
    training_client: tinker.TrainingClient,
    ml_logger: ml_log.Logger,
    experiment_setting: ExperimentSettings,
):
    logger.info("Train worker started.")
    step = 0
    tasks_to_practice: dict[str, Task] = dict()
    pending_fwd_bwd_future = None
    pending_optim_future = None
    while not state.is_finished():
        batch = state.get_batch()

        if batch is None:
            await asyncio.sleep(1)
            continue
        step += 1

        logger.info(f"Train worker step: {step}. ")

        fwd_bwd_future = await training_client.forward_backward_async(
            batch.datum,
            loss_fn=batch.loss_fn,
        )
        optim_future = await training_client.optim_step_async(params.adam_params)
        if pending_fwd_bwd_future is not None and pending_optim_future is not None:
            fwd_bwd_result = await pending_fwd_bwd_future.result_async()
            await pending_optim_future.result_async()
            ml_logger.log_metrics(
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

        if step % params.save_freq == 0:
            logger.info(f"Saving checkpoint at train step {step}...")
            await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"step_{step}",
                log_path=str(experiment_setting.log_root()),
                loop_state={},
                kind="both",
                ttl_seconds=experiment_setting.ttl_seconds,
            )

        if step % params.update_freq == 0:
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            await state.update_sampling_client(new_client)

            logger.info("Train Worker -> Latest sampling client updated.")
            for task in tasks_to_practice.values():
                state.add_to_practice_queue(task)
            tasks_to_practice.clear()


def setup_logging(cfg: UniRLConfig):
    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    setup_base_loglevel()

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

    return service_client, training_client


async def main():
    load_dotenv()
    ray.init(logging_level=logging.ERROR, configure_logging=True)
    cfg = UniRLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent UniRL",
            experiment_name=f"Adapter Agent_UniRL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        env_params=EnvParams(
            max_turns=10,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2:latest",
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
        num_study_workers=16,
        num_practice_workers=16,
        sft_batch_size=128,
        rl_group_size=32,
        log_freq=50,
    )

    setup_base_loglevel()
    setup_logging(cfg)

    tasks = load_tasks(cfg)
    study_queue = TaskQueue.create(order="LIFO", maxsize=200)

    if study_queue.maxsize > 0 and len(tasks) > study_queue.maxsize:
        raise ValueError(
            f"Number of initial tasks ({len(tasks)}) exceeds the maximum "
            f"capacity of the study queue ({study_queue.maxsize})."
        )

    practice_queue = TaskQueue.create(order="FIFO", maxsize=0)

    for task in tasks:
        study_queue.put(
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
        task_verifier=task_verifier,
        library_name=cfg.env_params.library.name,
        max_study_retry=cfg.study_rollout_params.max_study_retry,
        cfg=cfg,
    )

    async with litellm_concurrent_limit(
        cfg.num_study_workers * cfg.study_rollout_params.rollouts_per_task
        + cfg.num_practice_workers * cfg.practice_rollout_params.rollouts_per_task
        + 10
    ):
        await asyncio.gather(
            train_worker(
                state=state,
                params=cfg.train_params,
                training_client=training_client,
                ml_logger=state.ml_logger,
                experiment_setting=cfg.experiment_setting,
            ),
            *[
                study_rollout(
                    i,
                    state,
                    verifier,
                    cfg.study_rollout_params,
                    cfg.env_params,
                )
                for i in range(cfg.num_study_workers)
            ],
            *[
                practice_rollout(i, state, verifier, cfg.practice_rollout_params)
                for i in range(cfg.num_practice_workers)
            ],
        )


if __name__ == "__main__":
    asyncio.run(main())
