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

import tinker
from dotenv import load_dotenv

from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.types import (
    CheckpointSettings,
    EvalSettings,
    RolloutSettings,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


async def main() -> None:
    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up continuation RL pipeline...")

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
        simple_train_id=simple_train_id,
        library_name="numrs2",
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            # resume_trainer_path="tinker://e2a354f7-905b-5e4f-a2f5-c4cb4217c56c:train:0/weights/rl_0050",
            # resume_sampler_path="tinker://e2a354f7-905b-5e4f-a2f5-c4cb4217c56c:train:0/sampler_weights/rl_0050",
            resume_trainer_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/weights/rl_0020",
            resume_sampler_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020",
            lora_rank=32,
        ),
        rollout=RolloutSettings(
            runtime_settings=runtime_settings,
            num_samples=8,
            runtime_pool_size=50,
            worker_count=50,
        ),
        eval=EvalSettings(
            eval_rollout=4,
            eval_interval=5,
            eval_concurrency=48,
        ),
        checkpoint=CheckpointSettings(checkpoint_interval=10),
        sft=None,
        max_iterations=200,
        generation_concurrency=400,
        rl_batch_size=48,
        rl_update_epochs=1,
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="cispo",
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
    )

    pipeline = await SimplePipeline.create(
        config=config,
        knowledge_list=[],
        seed_suites=[study_rl_suite, additional_rl_suite, eval_suite],
    )

    try:
        logger.info("Starting continuation RL pipeline execution.")
        await pipeline.run()
        logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
