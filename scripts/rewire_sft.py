import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import litellm
import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from tinker import TrainingClient
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.hierarchical.agent.rewirer import Rewirer
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire_session import (
    SolveVerifyTinkerResult,
    rewire_session,
)
from adapter_agent.hierarchical.state import SFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from adapter_agent.rl.config import (
    EnvParams,
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    RLConfig,
    RolloutParams,
)
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.util.logger_util import setup_base_loglevel

litellm.add_function_to_prompt

logger = logging.getLogger(__name__)


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


@dataclass
class RewireSFTState:
    renderer: Renderer
    queue_questions: asyncio.Queue[QA]
    queue_rollouts: asyncio.Queue[list[SolveVerifyTinkerResult]]
    sampling_client_manager: SharedSamplingClient
    litellm_model_name: str
    training_client: TrainingClient
    step_counter: int = 0

    def get_latest_model(self) -> TinkerModel:
        current_model = TinkerModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
            renderer=self.renderer,
        )
        return current_model


async def rollout_worker(
    worker_id: int,
    rewire_state: RewireSFTState,
    verifier: Verifier,
    rewirer: Rewirer,
    rollout_params: RolloutParams,
):
    logger.info(f"Rollout Worker {worker_id} started.")
    while True:
        try:
            qa_item = rewire_state.queue_questions.get_nowait()
        except asyncio.QueueEmpty:
            break

        question_text = qa_item.question

        # Create new model and solver for this run
        model = rewire_state.get_latest_model()
        task = Task.from_instruction(question_text)
        results = await gather_with_semaphore(
            [
                rewire_session(
                    solver_model=model, verifier=verifier, rewirer=rewirer, task=task
                )
                for _ in range(rollout_params.rollouts_per_question)
            ],
            max_concurrent=rollout_params.per_group_concurrency,
        )

        trajectories: list[Trajectory] = []
        final_rewards: list[float] = []

        # Update trajectory queue
        trajectories = [r.trajectory for r in results if r.trajectory]
        final_rewards = [r.reward for r in results if r.trajectory]

        # TODO

        # Update question queue
        if not trajectories:
            logger.info(f"Worker {worker_id} produced NO trajectories for question.")
            rewire_state.queue_questions.task_done()
            continue
        elif len(trajectories) == 1:
            logger.info("Only one trajectory produced. Skipping this question.")
            rewire_state.queue_questions.task_done()
            continue

        await rewire_state.queue_rollouts.put(results)
        rewire_state.queue_questions.task_done()
        logger.info(
            f"Worker {worker_id} produced TrajectoryGroup (mean reward: {sum(final_rewards) / len(final_rewards):.2f})"
        )

    logger.info(f"Rollout Worker {worker_id} finished.")


def setup_logging(cfg: RLConfig) -> MLLogger:
    # Setup logging
    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    setup_base_loglevel()
    ml_logger = ml_log.setup_logging(
        log_dir=str(log_root),
        wandb_project=cfg.experiment_setting.wandb_project,
        config=cfg,
    )
    logger.setLevel(logging.DEBUG)

    logging.getLogger("adapter_agent.hierarchical.agent.solver").setLevel(logging.DEBUG)
    logging.getLogger("adapter_agent.hierarchical.agent.simplified_solver").setLevel(
        logging.DEBUG
    )
    return ml_logger


def load_qas(cfg: RLConfig) -> list[QA]:
    # Load questions
    logger.info("Loading questions from generated_qas.json...")
    qas_data_raw = SFTDataset.model_validate_json(
        cfg.env_params.dataset_path.read_text()
    )
    # Parse into QA objects
    qas_data = qas_data_raw.shuffled()
    logger.info(f"Loaded {len(qas_data)} questions.")

    return qas_data


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

    # Verifier
    verifier_model = get_gemini()
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)

    # Rewirer
    rewirer = Rewirer(model=model, rust_doc_analyzer=rust_doc_analyzer)
    return model, verifier, rewirer


async def setup_tinker_clients(
    model_loading_settings: ModelLoadingSettings,
) -> tuple[tinker.ServiceClient, tinker.TrainingClient]:
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        model_loading_settings.model_name, rank=model_loading_settings.lora_rank
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
    # TODO: create RewireSFTConfig
    cfg = RLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent",
            experiment_name=f"Adapter Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        optimizer_params=OptimizerParams(
            adam_params=tinker.AdamParams(
                learning_rate=3e-5,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            loss_fn="ppo",
            advantage_regularizer="output_token",
            num_steps=1,
            kl_penalty_coef=0.0,
        ),
        rollout_params=RolloutParams(
            num_groups_per_batch=8,
            num_rollout_workers=1,
            rollouts_per_question=8,
            per_group_concurrency=1,
            temperature=0.7,
        ),
        env_params=EnvParams(
            max_turns=5,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
            dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            lora_rank=32,
        ),
    )

    # ml_logger = setup_logging(cfg)

    # Tinker Clients
    service_client, training_client = await setup_tinker_clients(
        cfg.model_loading_settings
    )

    # Setup agents
    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier, rewirer = setup_agents(
        service_client, cfg.model_loading_settings, rust_doc_analyzer
    )
    # Dataset
    qas_data = load_qas(cfg)

    sampling_client_manager = SharedSamplingClient(model.sampling_client)
    # Shared results list for CSV export
    rl_state = RewireSFTState(
        renderer=model.renderer,
        queue_questions=asyncio.Queue(),
        queue_rollouts=asyncio.Queue(),
        sampling_client_manager=sampling_client_manager,
        litellm_model_name=model.model,
        training_client=training_client,
    )

    # TODO: remove this
    for qa in qas_data * 3:
        rl_state.queue_questions.put_nowait(qa)

    # Start Workers
    # train_worker_task = asyncio.create_task(
    #     train_worker(
    #         worker_id=0,
    #         rl_state=rl_state,
    #         cfg=cfg,
    #         ml_logger=ml_logger,
    #     )
    # )
    rollout_workers_tasks = [
        asyncio.create_task(
            rollout_worker(
                worker_id=i,
                rewire_state=rl_state,
                verifier=verifier,
                rewirer=rewirer,
                rollout_params=cfg.rollout_params,
            )
        )
        for i in range(cfg.rollout_params.num_rollout_workers)
    ]

    # Wait for producers to finish, or trainer to crash
    rollouts_task = await asyncio.gather(*rollout_workers_tasks)
    # done, pending = await asyncio.wait(
    #     {rollouts_task, train_worker_task}, return_when=asyncio.FIRST_COMPLETED
    # )

    _ = await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(cfg.experiment_setting.log_root()),
        loop_state={"batch": rl_state.step_counter},
        kind="both",
        ttl_seconds=cfg.experiment_setting.ttl_seconds,
    )


if __name__ == "__main__":
    asyncio.run(main())
