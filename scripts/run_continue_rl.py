"""Continue RL training from an existing checkpoint against gh_archive tasks.

No SFT, no knowledge-based task generation: only a single RL seed suite
(first 40 gh_archive tasks) and a disjoint eval suite (tasks 40:60).
Each invocation allocates a fresh `simple_train_id` so multiple continuation
runs from the same base checkpoint remain separable in the DB.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import tinker
from dotenv import load_dotenv
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class ClockCycleFilteredLogger(MLLogger):
    """MLLogger wrapper that drops metric keys containing 'clock_cycle'."""

    def __init__(self, base: MLLogger) -> None:
        self._base = base

    def log_hparams(self, config: Any) -> None:
        self._base.log_hparams(config)

    def log_metrics(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        filtered = {k: v for k, v in metrics.items() if "clock_cycle" not in k}
        self._base.log_metrics(filtered, step)

    def log_long_text(self, key: str, text: str) -> None:
        self._base.log_long_text(key, text)

    def close(self) -> None:
        self._base.close()


async def main() -> None:
    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up continuation RL pipeline...")

    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    simple_train_id = f"continue_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    study_rl_suite = load_gh_archive_suite(
        name="gh_archive_study_suite",
        task_slice=slice(0, 40),
        for_rl=True,
        for_eval=False,
    )
    additional_rl_suite = load_gh_archive_suite(
        name="gh_archive_study_suite",
        task_slice=slice(60, 300),
        for_rl=True,
        for_eval=False,
    )
    eval_suite = load_gh_archive_suite(
        name="gh_archive_eval",
        task_slice=slice(40, 60),
        for_rl=False,
        for_eval=True,
    )
    logger.info(
        f"RL suite: {len(study_rl_suite.tasks)} tasks, eval suite: {len(eval_suite.tasks)} tasks."
    )

    config = PipelineConfig(
        runtime_pool_size=50,
        rl_worker_count=50,
        eval_concurrency=48,
        generation_concurrency=400,
        simple_train_id=simple_train_id,
        library_name="numrs2",
        runtime_settings=runtime_settings,
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            resume_trainer_path="tinker://e2a354f7-905b-5e4f-a2f5-c4cb4217c56c:train:0/weights/rl_0050",
            resume_sampler_path="tinker://e2a354f7-905b-5e4f-a2f5-c4cb4217c56c:train:0/sampler_weights/rl_0050",
            lora_rank=32,
        ),
        sft=None,
        eval_rollout=4,
        eval_interval=5,
        rl_rollout=16,
        max_iterations=200,
        rl_checkpoint_interval=10,
        rl_batch_size=96,
        rl_update_epochs=1,
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="cispo",
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
    )

    pipeline = await SimplePipeline.create(
        config=config,
        rust_doc_analyzer=analyzer,
        knowledge_list=[],
        seed_suites=[study_rl_suite, additional_rl_suite, eval_suite],
    )

    filtered_logger = ClockCycleFilteredLogger(pipeline.ml_logger)
    pipeline.ml_logger = filtered_logger
    pipeline.evaluate_worker.ml_logger = filtered_logger

    try:
        async with analyzer:
            logger.info("Starting continuation RL pipeline execution.")
            await pipeline.run()
            logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
