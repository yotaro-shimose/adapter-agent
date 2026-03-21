import asyncio
import logging
from datetime import datetime
from pathlib import Path

import tinker
import weave  # noqa: F401
from oai_utils.async_utils import gather_with_semaphore

from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.rl.config import (
    EnvParams,
    ExperimentSettings,
    ModelLoadingSettings,
    RewireSFTConfig,
    RolloutParams,
    SFTOptimizerParams,
)
from adapter_agent.util.logger_util import setup_base_loglevel
from scripts.rewire_sft import load_qas, setup_agents

logger = logging.getLogger(__name__)


class _SuppressExtensionWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "train_on_what=ALL_ASSISTANT_MESSAGES" not in record.getMessage()


def setup_logging(cfg: RewireSFTConfig):
    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    setup_base_loglevel()

    logging.basicConfig(level=logging.INFO)

    logging.getLogger("tinker_cookbook.renderers.base").addFilter(
        _SuppressExtensionWarning()
    )
    logging.getLogger(
        "adapter_agent.hierarchical.process.rewire_session_single_turn"
    ).setLevel(logging.DEBUG)

    logging.getLogger("adapter_agent").setLevel(logging.DEBUG)
    logging.getLogger("scripts").setLevel(logging.DEBUG)


async def process_task(task, solver_model, verifier, worker_id):
    logger.info(f"[Worker {worker_id}] Evaluating task: {task.instruction}")
    try:
        result = await ss_solve_verify(
            solver_model=solver_model,
            verifier=verifier,
            rewirer_model=solver_model,
            task=task,
            max_turns=5,
        )
        is_success = result.rewired is not None
        logger.info(
            f"[Worker {worker_id}] Task result: {'Success' if is_success else 'Failed'}"
        )
        return result
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error evaluating task: {e}")
        return None


async def main():
    cfg = RewireSFTConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent Eval",
            experiment_name=f"Adapter_Agent_Eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
            num_rollout_workers=2,
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
            lora_rank=32,
        ),
    )

    setup_logging(cfg)

    # Tinker Clients (No training client)
    logger.info("Setting up Tinker Service Client...")
    service_client = tinker.ServiceClient()

    # Setup agents
    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    model, verifier = setup_agents(
        service_client,
        cfg.model_loading_settings,
        rust_doc_analyzer,
    )

    # Dataset
    qas_data = load_qas(cfg)
    tasks = [Task.from_instruction(qa.question) for qa in qas_data]

    logger.info(f"Starting evaluation of {len(tasks)} tasks...")

    coros = [
        process_task(task, model, verifier, i % cfg.rollout_params.num_rollout_workers)
        for i, task in enumerate(tasks)
    ]

    results = await gather_with_semaphore(
        coros, max_concurrent=cfg.rollout_params.per_group_concurrency
    )

    successful = len([r for r in results if r and r.rewired is not None])
    logger.info(f"Total successful tasks: {successful} / {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
