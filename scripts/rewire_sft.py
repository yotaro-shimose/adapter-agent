import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import litellm
import numpy as np
import tinker
import tinker_cookbook.checkpoint_utils
import weave  # noqa: F401
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from tinker import TrainingClient
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import Message as TinkerMessage
from tinker_cookbook.renderers.base import (
    Renderer,
)
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QA, TinkerMessagesDataset, TinkerMessageTrajectory
from adapter_agent.hierarchical.agent.rewirer import Rewirer, SingleTurnRewirer
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
    rewire_session,
)
from adapter_agent.hierarchical.process.rewire_session_single_turn import (
    rewire_session_single_turn,
)
from adapter_agent.hierarchical.state import QASFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    EnvParams,
    ExperimentSettings,
    ModelLoadingSettings,
    RolloutParams,
    SFTConfig,
    SFTOptimizerParams,
)
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.util.logger_util import setup_base_loglevel

litellm.add_function_to_prompt

logger = logging.getLogger(__name__)


@dataclass
class RewireSFTState:
    renderer: Renderer
    queue_questions: asyncio.Queue[QA]
    queue_rollouts: asyncio.Queue[list[RewireSessionResult]]
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
    rewirer: Rewirer | SingleTurnRewirer,
    rollout_params: RolloutParams,
    env_params: EnvParams,
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
        if env_params.single_turn:
            assert isinstance(rewirer, SingleTurnRewirer)
            coros = [
                rewire_session_single_turn(
                    solver_model=model,
                    verifier=verifier,
                    rewirer=rewirer,
                    task=task,
                    max_rewire=2,
                )
                for _ in range(rollout_params.rollouts_per_question)
            ]
        else:
            assert isinstance(rewirer, Rewirer)
            coros = [
                rewire_session(
                    solver_model=model,
                    verifier=verifier,
                    rewirer=rewirer,
                    task=task,
                    max_turns=env_params.max_turns,
                    max_rewire=2,
                )
                for _ in range(rollout_params.rollouts_per_question)
            ]

        results = await gather_with_semaphore(
            coros,
            max_concurrent=rollout_params.per_group_concurrency,
        )

        await rewire_state.queue_rollouts.put(results)
        rewire_state.queue_questions.task_done()
        successful = len([ret for ret in results if ret.rewired is not None])
        logger.info(
            f"Worker {worker_id} produced {successful}/{len(results)} successful trajectories"
        )

    logger.info(f"Rollout Worker {worker_id} finished.")


@dataclass
class TrainingDataManager:
    trajectories: list[RewireSessionResult] = field(default_factory=list)
    all_trajectories: list[RewireSessionResult] = field(default_factory=list)

    def update(self, trajectories: list[RewireSessionResult]):
        self.trajectories.extend(trajectories)
        self.all_trajectories.extend(trajectories)

    def reset_temporal(self):
        self.trajectories = []

    def successful_trajectories(self) -> list[list[TinkerMessage]]:
        successful_messages = [
            ret.rewired.messages for ret in self.trajectories if ret.rewired is not None
        ]
        return successful_messages

    def all_successful_trajectories(self) -> list[list[TinkerMessage]]:
        successful_messages = [
            ret.rewired.messages
            for ret in self.all_trajectories
            if ret.rewired is not None
        ]
        return successful_messages

    def save_all_successful_trajectories(self, log_file_path: Path):
        if len(self.all_successful_trajectories()) > 0:
            self.as_sft_dataset().save(log_file_path)

    def temporal_success_ratio(self) -> float:
        return len(self.successful_trajectories()) / len(self.trajectories)

    def as_sft_dataset(self) -> TinkerMessagesDataset:
        return TinkerMessagesDataset(
            items=[
                TinkerMessageTrajectory(
                    messages=ret.rewired.messages,
                    task=ret.task,
                )
                for ret in self.all_trajectories
                if ret.rewired is not None
            ]
        )

    def get_current_metrics(self) -> dict[str, float]:
        metrics = {
            "success_ratio": self.temporal_success_ratio(),
            "num_unfiltered_samples": len(self.trajectories),
            "rewire_count": np.mean(
                [
                    ret.rewired.n_rewire
                    for ret in self.trajectories
                    if ret.rewired is not None
                ]
            ).item(),
            "success_without_rewire_ratio": np.mean(
                [
                    ret.rewired.n_rewire == 0
                    for ret in self.trajectories
                    if ret.rewired is not None
                ]
            ).item(),
        }
        session_metrics = defaultdict(list)
        for ret in self.trajectories:
            for key, value in ret.metrics.items():
                session_metrics[key].append(value)
        for key, value in session_metrics.items():
            metrics[key] = np.mean(value).item()
        return metrics


async def train_worker(
    training_client: TrainingClient,
    rewire_state: RewireSFTState,
    cfg: SFTConfig,
    kl_reference_client: tinker.SamplingClient | None,
    ml_logger: MLLogger,
):
    logger.info("Trainer Worker started.")
    log_file_path = cfg.experiment_setting.log_root() / "sft_trajectories.json"
    data_manager = TrainingDataManager()

    while True:
        results = await rewire_state.queue_rollouts.get()
        data_manager.update(results)
        successful_trajectories = data_manager.successful_trajectories()
        data_manager.save_all_successful_trajectories(log_file_path)

        rewire_state.queue_rollouts.task_done()
        if len(successful_trajectories) >= cfg.optimizer_params.batch_size:
            metrics = {}

            sft_samples = [
                conversation_to_datum(
                    messages,
                    rewire_state.renderer,
                    max_length=None,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                for messages in successful_trajectories
            ]
            fwd_bwd_future = await training_client.forward_backward_async(
                data=sft_samples, loss_fn="cross_entropy"
            )
            optim_future = await training_client.optim_step_async(
                cfg.optimizer_params.adam_params
            )
            for step in range(cfg.optimizer_params.num_epochs):
                # Enqueue next step before consuming current results
                if step + 1 < cfg.optimizer_params.num_epochs:
                    logger.info(
                        f"Starting SFT Step {step + 2}/{cfg.optimizer_params.num_epochs}"
                    )
                    next_fwd_bwd_future = await training_client.forward_backward_async(
                        data=sft_samples, loss_fn="cross_entropy"
                    )
                    next_optim_future = await training_client.optim_step_async(
                        cfg.optimizer_params.adam_params
                    )
                else:
                    next_fwd_bwd_future = None
                    next_optim_future = None

                # Consume current results
                fwd_bwd_result = await fwd_bwd_future.result_async()
                await optim_future.result_async()

                metrics = fwd_bwd_result.metrics
                # if step == 0 and kl_reference_client is not None:
                #     kl_div = await compute_kl_divergence(
                #         sft_samples, fwd_bwd_result, kl_reference_client
                #     )
                #     metrics["train/kl_div"] = kl_div

                logger.info(f"SFT Step {step + 1} Completed. Metrics: {metrics}")

                # Move to next iteration
                if next_fwd_bwd_future is not None and next_optim_future is not None:
                    fwd_bwd_future = next_fwd_bwd_future
                    optim_future = next_optim_future
            ml_logger.log_metrics(metrics | data_manager.get_current_metrics())
            data_manager.reset_temporal()

            # new_client = (
            #     await training_client.save_weights_and_get_sampling_client_async()
            # )
            # await rewire_state.sampling_client_manager.update_client(new_client)
            logger.info(
                f"[Train Worker]   -> Latest sampling client updated. New ID: {id(rewire_state.sampling_client_manager.get_client())}"
            )


class _SuppressExtensionWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "train_on_what=ALL_ASSISTANT_MESSAGES" not in record.getMessage()


def setup_logging(cfg: SFTConfig) -> MLLogger:
    # Setup logging
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

    logging.getLogger("adapter_agent.hierarchical.agent.rewirer").setLevel(
        # logging.DEBUG
        logging.INFO
    )
    logging.getLogger("adapter_agent.hierarchical.process.rewire_session").setLevel(
        logging.DEBUG
    )
    logging.getLogger(
        "adapter_agent.hierarchical.process.rewire_session_single_turn"
    ).setLevel(logging.DEBUG)

    return ml_logger


def load_qas(cfg: SFTConfig) -> list[QA]:
    # Load questions
    logger.info(f"Loading questions from {cfg.env_params.dataset_path}...")
    qas_data_raw = QASFTDataset.model_validate_json(
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
    single_turn: bool = False,
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
    rewirer_model = get_gemini()
    if single_turn:
        rewirer = SingleTurnRewirer(
            model=rewirer_model, rust_doc_analyzer=rust_doc_analyzer
        )
    else:
        rewirer = Rewirer(model=rewirer_model, rust_doc_analyzer=rust_doc_analyzer)
    return model, verifier, rewirer


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
    cfg = SFTConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent SFT",
            experiment_name=f"Adapter Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        optimizer_params=SFTOptimizerParams(
            adam_params=tinker.AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            num_epochs=1,
            batch_size=64,
        ),
        rollout_params=RolloutParams(
            num_rollout_workers=32,
            rollouts_per_question=2,
            per_group_concurrency=2,
            temperature=0.7,
        ),
        env_params=EnvParams(
            max_turns=2,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
            dataset_path=Path("data/sft/gen_20260218_182450/sft_dataset.json"),
            single_turn=True,
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            # model_name="Qwen/Qwen3-4B-Instruct-2507",
            lora_rank=32,
        ),
    )

    setup_base_loglevel()
    ml_logger = setup_logging(cfg)

    # Tinker Clients
    service_client, training_client = await setup_tinker_clients(
        cfg.model_loading_settings
    )

    # Setup agents
    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier, rewirer = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
        single_turn=cfg.env_params.single_turn,
    )
    # Dataset
    qas_data = load_qas(cfg)

    logger.info("Initializing KL reference client for monitoring...")
    kl_reference_client = service_client.create_sampling_client(
        base_model=cfg.model_loading_settings.model_name,
        model_path=cfg.model_loading_settings.resume_sampler_path,
    )

    sampling_client_manager = SharedSamplingClient(model.sampling_client)
    # Shared results list for CSV export
    rewire_state = RewireSFTState(
        renderer=model.renderer,
        queue_questions=asyncio.Queue(),
        queue_rollouts=asyncio.Queue(),
        sampling_client_manager=sampling_client_manager,
        litellm_model_name=model.model,
        training_client=training_client,
    )

    # TODO: remove this
    for qa in qas_data * 2:
        rewire_state.queue_questions.put_nowait(qa)

    # Start Workers
    train_worker_task = asyncio.create_task(
        train_worker(
            training_client=training_client,
            rewire_state=rewire_state,
            cfg=cfg,
            kl_reference_client=kl_reference_client,
            ml_logger=ml_logger,
        )
    )
    rollout_workers_tasks = [
        asyncio.create_task(
            rollout_worker(
                worker_id=i,
                rewire_state=rewire_state,
                verifier=verifier,
                rewirer=rewirer,
                rollout_params=cfg.rollout_params,
                env_params=cfg.env_params,
            )
        )
        for i in range(cfg.rollout_params.num_rollout_workers)
    ]

    # Wait for producers to finish, or trainer to crash
    rollouts_task = asyncio.gather(*rollout_workers_tasks)
    done, pending = await asyncio.wait(
        {rollouts_task, train_worker_task}, return_when=asyncio.FIRST_COMPLETED
    )

    _ = await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(cfg.experiment_setting.log_root()),
        loop_state={"batch": rewire_state.step_counter},
        kind="both",
        ttl_seconds=cfg.experiment_setting.ttl_seconds,
    )


if __name__ == "__main__":
    asyncio.run(main())
