from oai_utils.tinker import setup_tinkermodel
from adapter_agent.hierarchical.process.evaluation import run_evaluation
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import tinker
import tinker.types as ttypes
from oai_utils.tinker import LogprobLitellmModel
from pydantic import BaseModel, Field
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)

# --- Config ---


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    def to_tinker_adam_params(self) -> ttypes.AdamParams:
        return ttypes.AdamParams(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
        )


class DataConfig(BaseModel):
    data_path: Path = Path("generated_qas.json")
    train_ratio: float = 0.9
    test_ratio: float = 0.1


class SFTConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    base_url: str | None = None
    lora_rank: int = 32
    workspace_template: Path = Path("templates/rust_template")
    library: Library = Library(name="numrs2", local_path=Path("repositories/numrs"))
    log_path: str = "./logs/sft_tinker"
    wandb_project: str | None = "SFT Tinker"
    experiment_name: str = Field(
        default_factory=lambda: f"SFT_Tinker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    data_config: DataConfig = Field(default_factory=DataConfig)
    num_train_steps: int = 30
    ttl_seconds: int = 604800  # 7 days


# --- Helper Functions (Borrowed from h_tinker.py) ---
def setup_rust_doc_analyzer(host_lib_dir: Path) -> RustDocAnalyzer:
    doc_path = host_lib_dir / "target" / "doc"
    pubapi_path = host_lib_dir / "pubapi.txt"
    json_path = None
    if doc_path.exists():
        if (doc_path / "numrs2.json").exists():
            json_path = doc_path / "numrs2.json"

    if json_path and json_path.exists():
        return RustDocAnalyzer.from_json(json_path, pubapi_path=pubapi_path)
    else:
        raise FileNotFoundError(f"Could not find rustdoc json in {doc_path}")


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


def update_solver_sampling_client(
    solver: Solver[LogprobLitellmModel],
    sampling_client: tinker.SamplingClient,
):
    solver.model.update_sampling_client(sampling_client)


def load_data(data_config: DataConfig) -> tuple[list[QA], list[QA]]:
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
    train_qas: list[QA],
    renderer: Renderer,
    cfg: SFTConfig,
):
    logger.info("Starting SFT Training...")

    # Convert QAs to Data
    train_data = [qa2sample(qa, renderer) for qa in train_qas]

    adam_params = cfg.optimizer_config.to_tinker_adam_params()

    # Enqueue first step
    logger.info(f"Starting SFT Step 1/{cfg.num_train_steps}")
    fwd_bwd_future = await training_client.forward_backward_async(
        data=train_data, loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for step in range(cfg.num_train_steps):
        # Enqueue next step before consuming current results
        if step + 1 < cfg.num_train_steps:
            logger.info(f"Starting SFT Step {step + 2}/{cfg.num_train_steps}")
            next_fwd_bwd_future = await training_client.forward_backward_async(
                data=train_data, loss_fn="cross_entropy"
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None

        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()

        metrics = fwd_bwd_result.metrics
        logger.info(f"SFT Step {step + 1} Completed. Metrics: {metrics}")

        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future
    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{cfg.num_train_steps:06d}",
        log_path=cfg.log_path,
        loop_state={"batch": cfg.num_train_steps},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )
    logger.info(f"Saved checkpoint to {path_dict}")


def setup_logger(cfg: SFTConfig):
    setup_base_loglevel()

    # Setup Logging
    ml_log.setup_logging(
        log_dir=cfg.log_path, wandb_project=cfg.wandb_project, config=cfg
    )
    logging.basicConfig(level=logging.INFO)


# --- Main Logic ---
async def main():
    cfg = SFTConfig(num_train_steps=30)
    # Update log_path to include experiment_name
    cfg.log_path = f"{cfg.log_path}/{cfg.experiment_name}"
    setup_logger(cfg)

    logger.info("Starting sft_tinker.py - V2 (12 Train / 4 Test) - Context Fix")
    # Load Data
    train_qas, test_qas = load_data(cfg.data_config)
    logger.info(
        f"Loaded {len(train_qas)} training samples and {len(test_qas)} test samples."
    )

    # Setup Clients
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    # Use LORA training client
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    model, _tokenizer, renderer = setup_tinkermodel(service_client, cfg.model_name)

    # --- SFT Training ---
    if cfg.num_train_steps > 0:
        await run_training_loop(training_client, train_qas, renderer, cfg)

    else:
        logger.info("SFT Training Skipped (num_train_steps == 0). Using Base Model.")

    # --- Evaluation ---
    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.library.local_path)
    # Agents for evaluation
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    model.update_sampling_client(sampling_client)
    solver = Solver(model=model, rust_doc_analyzer=rust_doc_analyzer, memory=None)

    # Use Gemini for verifier
    verifier_model = get_gemini()
    verifier = Verifier(
        model=verifier_model, rust_doc_analyzer=rust_doc_analyzer, memory=None
    )
    # await run_evaluation("Train", train_qas, solver, verifier, cfg.workspace_template)
    await run_evaluation("Test", test_qas, solver, verifier)


if __name__ == "__main__":
    asyncio.run(main())
