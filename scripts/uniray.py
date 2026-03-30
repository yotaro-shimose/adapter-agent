import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import ray
import tinker
import tinker_cookbook.checkpoint_utils
from dotenv import load_dotenv
from oai_utils import AgentsSDKModel
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from ray.actor import ActorHandle
from tinker import AdamParams

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.task_verifier import TaskVerifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import EnvParams, ExperimentSettings, ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    RewireSessionResultFailure,
)
from adapter_agent.rl.postgres_db import get_db
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.task_net import (
    TaskNetwork,
    is_study,
)
from adapter_agent.rl.trajectory_db import create_trajectory_db
from adapter_agent.rl.unirl_state import (
    StudyRolloutParams,
    TrajectoryReplayBuffer,
    UniRLConfig,
    UniRLState,
    UniRLTrainParams,
    task_manager_from_state_handle,
)
from adapter_agent.util.logger_util import setup_base_loglevel

# Silence Ray logs
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_COLOR_PREFIX"] = "0"
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

logger = logging.getLogger(__name__)


@dataclass
class StudyActor:
    process_id: int
    state: ActorHandle[UniRLState]
    env_params: EnvParams
    num_workers: int
    experiment_id: int
    verifier_model: AgentsSDKModel = field(init=False, default=None)  # type: ignore
    rust_doc_analyzer: AsyncRustDocAnalyzer = field(init=False, default=None)  # type: ignore
    knowledge_db: KnowledgeDB = field(init=False, default=None)  # type: ignore

    def __post_init__(self):
        setup_base_loglevel()
        logging.basicConfig(level=logging.DEBUG)

    async def setup(self):
        logger.info(f"Study actor {self.process_id} setup started.")
        self.verifier_model = get_gemini()
        self.rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
            self.env_params.library.local_path
        )
        self.knowledge_db = KnowledgeDB.for_experiment(self.experiment_id)
        await self.knowledge_db.initialize()
        logger.info(f"Study actor {self.process_id} setup completed.")

    @ray.method
    async def run(self):
        await self.setup()
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

        while True:
            latest_model = await self.state.get_latest_model.remote()
            task_manager = await task_manager_from_state_handle(self.state)

            async with task_manager as current:
                if is_study(current):
                    task = current.task
                    ret = await ss_solve_verify(
                        solver_model=latest_model,
                        verifier_model=self.verifier_model,
                        rust_doc_analyzer=self.rust_doc_analyzer,
                        task=task.task,
                        max_turns=self.env_params.max_turns,
                        qwen_no_think=self.env_params.qwen_no_think,
                        runtime_settings=self.env_params.runtime_settings,
                        knowledge_db=self.knowledge_db,
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


@dataclass
class TrainWorker:
    state: ActorHandle[UniRLState]
    params: UniRLTrainParams
    experiment_setting: ExperimentSettings
    model_loading_settings: ModelLoadingSettings
    training_client: tinker.TrainingClient = field(init=False, default=None)  # type: ignore

    def __post_init__(self):
        logger.info("Train worker init.")
        setup_base_loglevel()
        logging.basicConfig(level=logging.DEBUG)
        self.training_client = setup_training_client(self.model_loading_settings)

    async def run(self):
        await self.state.log_info.remote("Train worker main loop started.")
        step = 0
        pending_fwd_bwd_future = None
        pending_optim_future = None
        while not await self.state.is_finished.remote():
            batch = await self.state.get_batch.remote()

            if batch is None:
                await asyncio.sleep(1)
                continue
            step += 1

            await self.state.log_info.remote(
                f"Train worker step: {step}. Starting Forward-Backward..."
            )

            fwd_bwd_future = await self.training_client.forward_backward_async(
                batch.datum,
                loss_fn=batch.loss_fn,
            )
            optim_future = await self.training_client.optim_step_async(
                self.params.adam_params
            )

            # Wait for previous step's results if pipelining
            if pending_fwd_bwd_future is not None and pending_optim_future is not None:
                await self.state.log_info.remote(
                    f"Awaiting results from step {step - 1}..."
                )
                fwd_bwd_result = await pending_fwd_bwd_future.result_async()
                await pending_optim_future.result_async()
                await self.state.log_train_metrics.remote(
                    {
                        **fwd_bwd_result.metrics,
                    }
                )
                await self.state.log_info.remote(
                    f"Results from step {step - 1} logged."
                )

            pending_fwd_bwd_future = fwd_bwd_future
            pending_optim_future = optim_future

            # Save checkpoint
            if step % self.params.save_freq == 0:
                await self.state.log_info.remote(
                    f"Saving checkpoint at train step {step}..."
                )
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
                await self.state.log_info.remote(
                    f"Updating sampling client weights (step {step})..."
                )
                new_client = await self.training_client.save_weights_and_get_sampling_client_async()
                await self.state.log_info.remote(
                    "Broadcasting new sampling client to UniRLState..."
                )
                await self.state.update_sampling_client.remote(new_client)
                await self.state.log_info.remote(
                    "Train Worker -> Latest sampling client updated."
                )


def setup_rollout_logging():
    logging.basicConfig(level=logging.INFO)
    setup_base_loglevel()
    logger.setLevel(level=logging.DEBUG)
    logging.getLogger("adapter_agent.hierarchical.process.rewire").setLevel(
        logging.WARNING
    )


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
        train_params=UniRLTrainParams(
            update_freq=1,
            save_freq=50,
            adam_params=AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            max_sft_reuse=15,
            min_sft_batch_size=32,
            max_sft_batch_size=128,
        ),
        log_freq=20,
        vis_json_path=Path("graphvis/public/data.json"),
    )

    setup_rollout_logging()

    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)

    tasks = load_gh_archive()
    study_queue = TaskNetwork(tasks[:30])

    service_client = tinker.ServiceClient()
    model, tokenizer, renderer = setup_tinkermodel(
        cfg.model_loading_settings.model_name,
        cfg.model_loading_settings.resume_sampler_path,
        service_client,
    )
    task_verifier = TaskVerifier(model=get_gemini())

    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    # Initialize Trajectory DB using PostgreSQL experiment registry
    db = await get_db()
    experiment_id = await db.register_experiment(cfg.experiment_setting.experiment_name)
    trajectory_db = await create_trajectory_db(experiment_id)
    buffer = TrajectoryReplayBuffer(
        min_sft_batch_size=cfg.train_params.min_sft_batch_size,
        max_sft_batch_size=cfg.train_params.max_sft_batch_size,
        max_sft_reuse=cfg.train_params.max_sft_reuse,
        trajectory_db=trajectory_db,
        renderer=model.renderer,
    )

    UniRLStateActor = ray.remote(UniRLState)
    state: ActorHandle[UniRLState] = cast(
        ActorHandle[UniRLState],
        UniRLStateActor.remote(
            study_task_queue=study_queue,
            buffer=buffer,
            litellm_model_name=model.model,
            renderer=model.renderer,
            sampling_client_manager=sampling_client_manager,
            task_verifier=task_verifier,
            library_name=cfg.env_params.library.name,
            cfg=cfg,
            experiment_id=experiment_id,
        ),
    )
    # Ensure initial graph is saved to DB for visualization
    await state.setup_db.remote()

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
        + 10
    ):
        train_task = train_worker_actor.run.remote()
        # Study Worker の起動
        study_workers = []
        StudyActorClass = ray.remote(StudyActor)
        for i in range(cfg.study_rollout_params.num_study_actor):
            # 初期起動スパイクをさける
            await asyncio.sleep(10.0)

            worker = cast(
                ActorHandle[StudyActor],
                StudyActorClass.remote(
                    i,
                    state,
                    cfg.env_params,
                    cfg.study_rollout_params.rollouts_per_actor,
                    experiment_id,
                ),
            )

            worker_task_ref = worker.run.remote()
            study_workers.append(worker_task_ref)
            logger.info(f"Launched Study Worker {i}...")

        ray.get([train_task, *study_workers])


if __name__ == "__main__":
    asyncio.run(main())
