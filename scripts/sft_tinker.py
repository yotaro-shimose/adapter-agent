import asyncio
import json
import logging
from datetime import datetime
from itertools import chain
from pathlib import Path

import tinker
import tinker.types as ttypes
from more_itertools import chunked
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from pydantic import BaseModel, Field
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QA
from adapter_agent.rl.config import (
    ExperimentSettings,
    ModelLoadingSettings,
    QADataConfig,
    SFTOptimizerParams,
    TrajectorySFTDataConfig,
)
from adapter_agent.util.logger_util import (
    ClockCycleFilteredLogger,
    setup_base_loglevel,
)

logger = logging.getLogger(__name__)


# --- Config ---


class SFTConfig(BaseModel):
    model_loading_settings: ModelLoadingSettings
    optimizer_params: SFTOptimizerParams
    log_path: str = "./logs/sft_tinker"
    wandb_project: str | None = "SFT Tinker"
    experiment_name: str = Field(
        default_factory=lambda: f"SFT_Tinker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    data_config: QADataConfig | TrajectorySFTDataConfig
    experiment_settings: ExperimentSettings
    save_every_epoch: int


def diagonify(conversation: list[Message]) -> list[list[Message]]:
    conversations = []
    for i, message in enumerate(conversation):
        if message["role"] == "assistant":
            conversations.append(conversation[: i + 1])
    return conversations


def qa2sample(qa: QA, renderer: Renderer) -> ttypes.Datum:
    return conversation_to_datum(
        conversation=[
            Message(
                role="user",
                content=qa.question,
            ),
            Message(
                role="assistant",
                content=qa.answer,
            ),
        ],
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        max_length=None,
    )


def load_qa_dataset(data_config: QADataConfig) -> tuple[list[QA], list[QA]]:
    path = data_config.data_path
    if not path.exists():
        raise FileNotFoundError(f"Data file {path} not found.")

    with path.open("r") as f:
        data = json.load(f)

    # SFTDataset stores items as a list of QA objects
    # Handle both direct list or SFTDataset format (dict with 'items')
    if isinstance(data, dict) and "items" in data:
        items_data = data["items"]
    elif isinstance(data, list):
        items_data = data
    else:
        raise ValueError("Invalid data format")

    qas = [QA(**item) for item in items_data]

    if len(qas) < 16:
        logger.warning(f"Data contains only {len(qas)} items, less than expected 16.")

    train_data = qas[: int(data_config.train_ratio * len(qas))]
    test_data = qas[int(data_config.train_ratio * len(qas)) :]
    return train_data, test_data


# --- Training Logic ---
async def run_training_loop(
    training_client: tinker.TrainingClient,
    training_data: list[tinker.Datum],
    num_epochs: int,
    batch_size: int,
    adam_params: ttypes.AdamParams,
    log_path: str,
    ttl_seconds: int,
    ml_logger: MLLogger,
    save_every_epoch: int,
):
    logger.info("Starting SFT Training...")

    # Enqueue first step
    fwd_bwd_future = await training_client.forward_backward_async(
        data=training_data, loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(adam_params)
    num_steps = 0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(chunked(training_data, batch_size)):
            num_steps += 1
            # Enqueue next step before consuming current results
            if step + 1 < num_epochs:
                next_fwd_bwd_future = await training_client.forward_backward_async(
                    data=batch, loss_fn="cross_entropy"
                )
                next_optim_future = await training_client.optim_step_async(adam_params)
            else:
                next_fwd_bwd_future = None
                next_optim_future = None

            # Consume current results
            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()

            metrics = fwd_bwd_result.metrics
            logger.info(f"Epoch {epoch + 1} Step {step + 1} Completed.")
            ml_logger.log_metrics(metrics)

            # Move to next iteration
            if next_fwd_bwd_future is not None and next_optim_future is not None:
                fwd_bwd_future = next_fwd_bwd_future
                optim_future = next_optim_future
        if (epoch + 1) % save_every_epoch == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{epoch + 1:06d}",
                log_path=log_path,
                loop_state={"batch": num_steps},
                kind="both",
                ttl_seconds=ttl_seconds,
            )
            logger.info(f"Saved checkpoint to {path_dict}")


def setup_logger(cfg: SFTConfig) -> MLLogger:
    setup_base_loglevel()

    # Setup Logging
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path, wandb_project=cfg.wandb_project, config=cfg
    )
    logging.basicConfig(level=logging.INFO)
    return ClockCycleFilteredLogger(ml_logger)


# --- Main Logic ---
async def main():
    cfg = SFTConfig(
        experiment_settings=ExperimentSettings.with_prefix("SFT_Tinker"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            resume_trainer_path=None,
            resume_sampler_path=None,
            lora_rank=32,
        ),
        optimizer_params=SFTOptimizerParams(
            adam_params=ttypes.AdamParams(
                learning_rate=1e-4,
            ),
            num_epochs=60,
            batch_size=280,  # Could be bigger than training set as a whole
        ),
        # data_config=QADataConfig(
        #     data_path=Path("data/sft/gen_20260218_182450/sft_dataset.json")
        # ),
        data_config=TrajectorySFTDataConfig(
            data_path=Path(
                # "logs/Adapter_Agent/Adapter Agent_20260225_054614/sft_trajectories.json",
                # "logs/Adapter_Agent/Adapter Agent_20260301_070513/sft_trajectories.json",
                "logs/Adapter_Agent/Adapter Agent_20260301_070513/sft_trajectories.json",
            ),
            train_ratio=0.9,
            test_ratio=0.1,
        ),
        save_every_epoch=15,
    )
    # Update log_path to include experiment_name
    cfg.log_path = f"{cfg.log_path}/{cfg.experiment_name}"
    ml_logger = setup_logger(cfg)

    logger.info("Starting sft_tinker.py")
    # Setup Clients
    service_client = tinker.ServiceClient()
    # Use LORA training client
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_loading_settings.model_name, rank=cfg.model_loading_settings.lora_rank
    )
    if cfg.model_loading_settings.resume_trainer_path is not None:
        await training_client.load_state_async(
            cfg.model_loading_settings.resume_trainer_path
        )
    tokenizer, renderer = get_tokenizer_renderer(
        training_client,
        cfg.model_loading_settings.model_name,
    )

    # Load Data
    if isinstance(cfg.data_config, QADataConfig):
        train_qas, test_qas = load_qa_dataset(cfg.data_config)
        train_data = [qa2sample(qa, renderer) for qa in train_qas]
    elif isinstance(cfg.data_config, TrajectorySFTDataConfig):
        train, test = cfg.data_config.train_test_split()
        diagonified_messages = list(
            chain.from_iterable(
                [diagonify(item.to_tinker_messages()) for item in train.items]
            )
        )
        train_data = [
            conversation_to_datum(
                messages,
                renderer=renderer,
                max_length=None,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            for messages in diagonified_messages
        ]

    else:
        raise ValueError("Invalid data config type")
    logger.info(f"Loaded {len(train_data)} training samples.")

    # --- SFT Training ---
    await run_training_loop(
        training_client=training_client,
        training_data=train_data,
        num_epochs=cfg.optimizer_params.num_epochs,
        batch_size=cfg.optimizer_params.batch_size,
        adam_params=cfg.optimizer_params.adam_params,
        log_path=cfg.log_path,
        ttl_seconds=cfg.experiment_settings.ttl_seconds,
        ml_logger=ml_logger,
        save_every_epoch=cfg.save_every_epoch,
    )


if __name__ == "__main__":
    asyncio.run(main())
