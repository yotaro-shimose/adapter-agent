import asyncio
import logging
import os
from pathlib import Path

import chz
import tinker
import tinker.types as ttypes
from oai_utils.litellm import litellm_concurrent_limit
from oai_utils.tinker import LogprobLitellmModel, setup_tinkermodel
from pydantic import BaseModel, Field
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.trace import scope

# Imports from hierarchical agent
from adapter_agent.hierarchical.agent.h_agent import Agents
from adapter_agent.hierarchical.process.runner import process_task
from adapter_agent.hierarchical.state import RLPool, SFTPool, TaskPool
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.qra import QA
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)


class KLReferenceConfig(BaseModel):
    base_model: str
    load_checkpoint_path: str | None = None


class AsyncConfig(BaseModel):
    max_steps_off_policy: int
    groups_per_batch: int


class ExperimentConfig(BaseModel):
    workspace_template_location: Path = Path("templates/rust_template")
    host_lib_dir: Path = Path("repositories/numrs")
    experiment_dir: Path = Path("experiments/hierarchical_tinker_run_260203")
    benchmark_path: Path = Path("experiments/gh/benchmark_dataset.csv")


class LoggingConfig(BaseModel):
    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = "./logs/hierarchical_tinker"
    enable_trace: bool = False


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-5
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


class Config(BaseModel):
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    temperature: float = 1.0
    lora_rank: int = 32
    sft_batch_size: int = 16
    group_size: int = 4

    compute_post_kl: bool = False
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    kl_reference_config: KLReferenceConfig | None = None

    num_rollout_workers: int = 30  # TODO: increase this

    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)
    base_url: str | None = None

    remove_constant_reward_groups: bool = False
    eval_every: int = 0
    save_every: int = 20
    ttl_seconds: int = 604800  # 7 days
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = chz.field(
        default_factory=lambda: AsyncConfig(
            max_steps_off_policy=1000, groups_per_batch=1
        )
    )
    num_groups_to_log: int = 4

    # Hierarchical specific paths
    experiment_config: ExperimentConfig = Field(default_factory=ExperimentConfig)


async def load_training_client(cfg: Config, service_client: tinker.ServiceClient):
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.logging_config.log_path)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
    return training_client


def setup_rust_doc_analyzer(host_lib_dir: Path) -> RustDocAnalyzer:
    # Init RustDocAnalyzer
    doc_path = host_lib_dir / "target" / "doc"
    pubapi_path = host_lib_dir / "pubapi.txt"

    # Try to find the json file
    json_path = None
    if doc_path.exists():
        # Heuristic: find numrs2.json or {repo_name}.json
        # Check specific known case first
        if (doc_path / "numrs2.json").exists():
            json_path = doc_path / "numrs2.json"
        else:
            raise ValueError(f"Could not find rustdoc json in {doc_path}")

    if json_path and json_path.exists():
        logger.info(f"Loading RustDocAnalyzer from {json_path}")
        rust_doc_analyzer = RustDocAnalyzer.from_json(
            json_path, pubapi_path=pubapi_path
        )
    else:
        logger.warning(
            f"Could not find rustdoc json in {doc_path}. Using dummy analyzer path (will fail on use)."
        )
        # Initialize with a non-existent path or fail?
        # Agents.from_model needs it. Failing is better than hidden bugs.
        raise FileNotFoundError(f"Could not find rustdoc json in {doc_path}")
    return rust_doc_analyzer


async def initialization(cfg: Config):
    # Setup
    exp_dir = Path(cfg.experiment_config.experiment_dir).absolute()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Init TaskPool
    logger.warning(
        f"Loading only 1 tasks from {cfg.experiment_config.benchmark_path} for debugging."
    )
    task_pool = TaskPool.from_benchmark_csv(cfg.experiment_config.benchmark_path, 1)
    sft_pool = SFTPool.new()

    logger.info(f"Loaded {len(task_pool.tasks)} tasks.")

    # Init Tinker Clients
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.logging_config.log_path,
        wandb_project=cfg.logging_config.wandb_project,
        config=cfg,
        wandb_name=cfg.logging_config.wandb_name,
    )
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await load_training_client(cfg, service_client)
    model, tokenizer, _renderer = setup_tinkermodel(service_client, cfg.model_name)
    rust_doc_analyzer = setup_rust_doc_analyzer(cfg.experiment_config.host_lib_dir)
    agents = Agents.from_model(model, rust_doc_analyzer)
    rl_pool = RLPool.new()
    return agents, training_client, ml_logger, task_pool, sft_pool, rl_pool, tokenizer


def qa2sample(qa: QA, tokenizer: Tokenizer) -> ttypes.Datum:
    return ttypes.Datum(
        model_input=ttypes.ModelInput.from_ints(tokenizer.encode(qa.question)),
        loss_fn_inputs={
            "target_tokens": ttypes.ModelInput.from_ints(tokenizer.encode(qa.answer))
        },
    )


def update_agents_sampling_client(
    agents: Agents, sampling_client: tinker.SamplingClient
):
    assert isinstance(agents.solver.model, LogprobLitellmModel)
    assert isinstance(agents.verifier.model, LogprobLitellmModel)
    assert isinstance(agents.analyzer.model, LogprobLitellmModel)
    assert isinstance(agents.decomposer.model, LogprobLitellmModel)
    agents.solver.model.update_sampling_client(sampling_client)
    agents.verifier.model.update_sampling_client(sampling_client)
    agents.analyzer.model.update_sampling_client(sampling_client)
    agents.decomposer.model.update_sampling_client(sampling_client)


@scope
async def main(cfg: Config):
    (
        agents,
        training_client,
        ml_logger,
        task_pool,
        sft_pool,
        rl_pool,
        tokenizer,
    ) = await initialization(cfg)
    setup_base_loglevel()

    async def rollout_worker(worker_id: str):
        logger.info(f"Worker {worker_id} started.")
        while True:
            task = await task_pool.pop_task()
            if task is None:
                logger.info(f"Worker {worker_id} stopping (shutdown signal received).")
                break

            try:
                await process_task(
                    agents=agents,
                    task=task,
                    task_pool=task_pool,
                    sft_pool=sft_pool,
                    workspace_template_location=cfg.experiment_config.workspace_template_location,
                    host_lib_dir=cfg.experiment_config.host_lib_dir,
                    experiment_dir=cfg.experiment_config.experiment_dir,
                    rl_pool=rl_pool,
                    group_size=cfg.group_size,
                )
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} encountered an error processing task {task.id}: {e}"
                )
            finally:
                # Mark task as finished regardless of success or failure
                await task_pool.finish_task(task)

    async def train_worker(worker_id: str):
        logger.info(f"Worker {worker_id} started.")
        while True:
            # Check RL pool first
            if (
                rl_pool.queue.qsize() >= cfg.sft_batch_size
            ):  # Using sft_batch_size as batch size for now
                # This might need to be cfg.async_config.groups_per_batch
                groups = rl_pool.get_batch(cfg.sft_batch_size)

                # Compute GRPO loss and backward
                from adapter_agent.hierarchical.grpo import compute_grpo_loss

                loss = await compute_grpo_loss(groups, training_client)
                # TODO: reorder task registering and consumption of its future
                optim_future = await training_client.optim_step_async(
                    cfg.optimizer_config.to_tinker_adam_params()
                )
                optim_result = await optim_future.result_async()

                new_client = (
                    await training_client.save_weights_and_get_sampling_client_async()
                )
                update_agents_sampling_client(agents, new_client)
                logger.info(f"Worker {worker_id} finished GRPO step. Loss: {loss}")
                continue

            if not sft_pool.queue.qsize() >= cfg.sft_batch_size:
                await asyncio.sleep(0.1)
                continue

            # SFT Fallback (or mixed training)
            samples = sft_pool.get_batch(cfg.sft_batch_size)
            data = [qa2sample(sample, tokenizer) for sample in samples]
            fwd_bwd_future = await training_client.forward_backward_async(
                data=data, loss_fn="cross_entropy"
            )
            optim_future = await training_client.optim_step_async(
                cfg.optimizer_config.to_tinker_adam_params()
            )
            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_future.result_async()
            new_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            update_agents_sampling_client(agents, new_client)
            logger.info(f"Worker {worker_id} finished SFT step.")
            # TODO: add metrics

    async with litellm_concurrent_limit(cfg.num_rollout_workers):
        workers = [
            asyncio.create_task(rollout_worker(f"rollout-{i}"))
            for i in range(cfg.num_rollout_workers)
        ]
        workers.append(asyncio.create_task(train_worker("train-0")))
        await asyncio.gather(*workers)


if __name__ == "__main__":
    config = Config(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    log_path = os.path.expanduser(config.logging_config.log_path)
    if os.path.exists(log_path):
        logger.warning(f"Log directory {log_path} already exists.")
    os.makedirs(log_path, exist_ok=True)

    asyncio.run(main(config))
