import asyncio
import logging
import os

from oai_utils import AgentsSDKModel

from adapter_agent.rl.task_net import is_study
from adapter_agent.rl.unirl_state import StudyRolloutParams

# Silence Ray logs
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_COLOR_PREFIX"] = "0"
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import ray
import tinker
import tinker_cookbook.checkpoint_utils
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from ray.actor import ActorHandle
from tinker import AdamParams

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.knowledge_slicer import KnowledgeSlicer
from adapter_agent.hierarchical.agent.knowledge_summarizer import KnowledgeSummarizer
from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    solve_verify_tinker_single_turn,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    RewireSessionResultFailure,
)
from adapter_agent.rl.env.single_turn import SingleTurnEnvState
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.task_net import (
    TaskNetwork,
    is_slice,
)
from adapter_agent.rl.unirl_state import (
    HybridReplayBuffer,
    PracticeRolloutParams,
    UniRLConfig,
    UniRLState,
    UniRLTrainParams,
    task_manager_from_state_handle,
)
from adapter_agent.util.exception import CodingEnvironmentError
from adapter_agent.util.logger_util import setup_base_loglevel
from adapter_agent.util.task_queue import TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class StudyActor:
    process_id: int
    state: ActorHandle[UniRLState]
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: RustDocAnalyzer
    env_params: EnvParams
    num_workers: int

    def __post_init__(self):
        setup_base_loglevel()
        logging.basicConfig(level=logging.DEBUG)

    @ray.method
    async def run(self):
        await asyncio.gather(
            *[
                self.study_worker(f"{self.process_id}-{worker_id}")
                for worker_id in range(self.num_workers)
            ]
        )

    async def study_worker(
        self,
        worker_id: str,
    ):
        logger.info(f"Study worker {worker_id} started.")
        initial_model: TinkerModel = await self.state.get_latest_model.remote()
        initial_model.register_tinkerllm_to_litellm()
        knowledge_summarizer = KnowledgeSummarizer(model=get_gemini())
        knowledge_slicer = KnowledgeSlicer(model=get_gemini())

        while True:
            latest_model = await self.state.get_latest_model.remote()
            task_manager = await task_manager_from_state_handle(self.state)

            async with task_manager as current:
                if is_slice(current):
                    try:
                        qra = await knowledge_slicer.slice(
                            current.task.knowledge.knowledge
                        )
                        current.register_result(current.task.complete(qra))
                    except Exception as e:
                        logger.error(f"Slicing failed: {e}")
                        current.register_result(current.task.complete(None))
                elif is_study(current):
                    task = current.task
                    knowledges_str = None
                    if task.knowledges:
                        knowledges_str = await knowledge_summarizer.summarize(
                            task.knowledges, task_instruction=task.task.instruction
                        )

                    ret = await ss_solve_verify(
                        solver_model=latest_model,
                        verifier_model=self.verifier_model,
                        rust_doc_analyzer=self.rust_doc_analyzer,
                        task=task.task,
                        max_turns=self.env_params.max_turns,
                        qwen_no_think=self.env_params.qwen_no_think,
                        runtime_settings=self.env_params.runtime_settings,
                        knowledges=knowledges_str,
                    )

                    if not task.is_generation or not isinstance(
                        ret, RewireSessionResultFailure
                    ):
                        current.register_result(task.complete(ret))
                        continue

                    try:
                        analyzer = Analyzer(model=get_gemini())
                        subtask = await analyzer.analyze_trajectory(ret.trials)

                        current.register_result(task.complete(ret, new_task=subtask))
                        continue
                    except Exception as e:
                        logger.error(f"Subtask generation failed: {e}")
                        current.register_result(task.complete(ret, new_task=None))
                        continue
                else:
                    raise ValueError(f"Unknown task type: {current}")


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
        logger.info("Train worker main loop started.")
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
                    loop_state={"batch": step},
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
            qwen_no_think=True,
            runtime_settings=RuntimeSettings(
                type="cloudrun",
                image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2",
            ),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
        study_rollout_params=StudyRolloutParams(
            num_study_actor=10,
            rollouts_per_actor=8,
        ),
        practice_rollout_params=PracticeRolloutParams(
            rollouts_per_task=6,
            num_practice_workers=0,
        ),
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
        log_freq=20,
        vis_json_path=Path("graphvis/public/data.json"),
    )

    setup_rollout_logging()

    tasks = load_gh_archive()
    study_queue = TaskNetwork(tasks[:30])
    practice_queue = TaskQueue.create(order="FIFO", maxsize=0)

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    service_client = tinker.ServiceClient()
    model, tokenizer, renderer = setup_tinkermodel(
        cfg.model_loading_settings.model_name,
        cfg.model_loading_settings.resume_sampler_path,
        service_client,
    )
    verifier_model = get_gemini()
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
        cfg.study_rollout_params.num_study_actor
        * cfg.study_rollout_params.rollouts_per_actor
        + cfg.practice_rollout_params.num_practice_workers
        * cfg.practice_rollout_params.rollouts_per_task
        + 10
    ):
        train_task = train_worker_actor.run.remote()
        # Study Worker の起動
        study_workers = []
        StudyActorClass = ray.remote(StudyActor)
        for i in range(cfg.study_rollout_params.num_study_actor):
            # ここで少し待つ (例: 10秒おきに1つずつ起動)ことによって初期起動スパイクをさける
            await asyncio.sleep(10.0)

            worker = cast(
                ActorHandle[StudyActor],
                StudyActorClass.remote(
                    i,
                    state,
                    verifier_model,
                    rust_doc_analyzer,
                    cfg.env_params,
                    cfg.study_rollout_params.rollouts_per_actor,
                ),
            )

            worker_task_ref = worker.run.remote()
            study_workers.append(worker_task_ref)
            logger.info(f"Launched Study Worker {i}...")

        # Practice Worker の起動
        practice_workers = []
        # for i in range(cfg.practice_rollout_params.num_practice_workers):
        #     await asyncio.sleep(1.0)

        #     worker = practice_rollout.remote(
        #         i, state, verifier, cfg.practice_rollout_params, cfg.env_params
        #     )
        #     practice_workers.append(worker)
        #     logger.info(f"Launched Practice Worker {i}...")

        ray.get([train_task, *study_workers, *practice_workers])


if __name__ == "__main__":
    asyncio.run(main())
