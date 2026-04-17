import asyncio
import logging
from pathlib import Path

import tinker

from adapter_agent.hierarchical.pipeline.internalization_pipeline import (
    InternalizationPipeline,
    PipelineConfig,
)
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    SFTOptimizerParams,
)
from adapter_agent.rl.env.runtime_settings import RuntimeSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
# Only enable DEBUG for our own modules to avoid httpcore/hpack noise
logging.getLogger("adapter_agent").setLevel(logging.DEBUG)
# Also grpo.py uses the module logger
logging.getLogger("adapter_agent.hierarchical.grpo").setLevel(logging.DEBUG)
# Suppress noisy coder_mcp logs
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def main():
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up internalization pipeline with professional analyzer...")

    # 1. Analyzer (RustDoc JSON + Elasticsearch)
    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)

    # 2. Model (Gemini)
    try:
        model = get_gemini()
    except ValueError as e:
        logger.error(e)
        return

    # 3. Runtime Settings
    runtime_settings = RuntimeSettings(
        type="docker", image_uri="coder-mcp-numrs2:latest"
    )
    # runtime_settings = RuntimeSettings(
    #     type="cloudrun",
    #     image_uri="europe-north1-docker.pkg.dev/dsat2-405406/shimose-repo/coder-mcp-numrs2",
    # )
    # 4. Pipeline Config
    db = KnowledgeDB.for_experiment("unirl_20260406_102313")
    async with db:
        knowledge_list = await db.list_knowledge(limit=4)

    if not knowledge_list:
        logger.warning("No knowledge found in DB for the specified experiment.")
        return

    logger.info(f"Loaded {len(knowledge_list)} knowledge items from DB.")

    config = PipelineConfig(
        knowledge_list=knowledge_list,
        runtime_settings=runtime_settings,
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B", lora_rank=32
        ),
        sft_optimizer_params=SFTOptimizerParams(
            adam_params=tinker.AdamParams(learning_rate=1e-3),
            batch_size=32,
            num_epochs=1,
        ),
        rl_optimizer_params=OptimizerParams(
            adam_params=tinker.AdamParams(learning_rate=1e-3),
            num_steps=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            loss_fn="ppo",
        ),
        experiment_settings=ExperimentSettings.with_prefix(
            "Primitive_SFT_Internalization"
        ),
        k_sft=4,
        k_init_sft=16,
        init_sft_epochs=6,
        k_rl=4,
        k_rollout=8,
        concurrency=32,
        max_iterations=20,
        max_sft_knowledge=8,
        task_gen_concurrency=64,
        stop_at_100=False,
    )

    # 5. Pipeline
    pipeline = InternalizationPipeline(
        config=config,
        generator_model=model,
        verifier_model=model,  # Using Gemini for both generation and verification
        rust_doc_analyzer=analyzer,  # type: ignore
        library_name="numrs2",
    )

    try:
        async with analyzer:
            print("\n" + "=" * 50)
            print("PRIMITIVE SFT INTERNALIZATION TEST")
            print("=" * 50 + "\n")

            await pipeline.setup()
            await pipeline.run()

            print("\n" + "=" * 50)
            print("TEST COMPLETE")
            print("=" * 50 + "\n")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
