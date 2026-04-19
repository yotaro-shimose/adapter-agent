import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv

from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Simplified Internalization Pipeline")
    parser.add_argument(
        "--granular-id",
        type=str,
        default="granular_prep_20260419_024056",
        help="SimpleTrainRun ID for granular knowledge (produced by prepare_granular_knowledge.py)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="study_20260418_233708",
        help="study.py experiment name; its solved root tasks will be added to the RL pool",
    )
    parser.add_argument(
        "--cpt",
        action="store_true",
        help="Use Continual Pre-training mode (single trainable user message)",
    )
    args = parser.parse_args()

    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up simplified internalization pipeline...")

    # 1. Analyzer
    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)

    # 2. Runtime Settings
    # Use standard cloudrun settings for the runtime environment.
    runtime_settings = RuntimeSettings.cloudrun_numrs2()

    # 3. Pipeline Configuration
    simple_train_id = f"simple_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = PipelineConfig(
        runtime_pool_size=100,
        rl_worker_count=48,
        eval_concurrency=48,
        generation_concurrency=400,
        simple_train_id=simple_train_id,
        granular_id=args.granular_id,
        study_experiment_id=args.experiment_id,
        library_name="numrs2",
        runtime_settings=runtime_settings,
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            resume_trainer_path=None,
            resume_sampler_path=None,
            lora_rank=32,
        ),
        k_sft=8,
        k_eval=1,
        k_rl=8,
        sft_epochs=8,
        eval_rollout=4,
        rl_rollout=8,
        sft_batch_size=256,
        max_iterations=50,
        rl_checkpoint_interval=10,
        adam_params=tinker.AdamParams(learning_rate=1e-3),
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="cispo",
        stop_grpo=False,
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
        cpt=args.cpt,
    )

    pipeline = await SimplePipeline.create(config=config, rust_doc_analyzer=analyzer)

    try:
        async with analyzer:
            logger.info("Starting simple pipeline execution.")
            await pipeline.run()
            logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
