import asyncio
import logging
from pathlib import Path

import tinker

from adapter_agent.hierarchical.pipeline.internalization_pipeline import (
    InternalizationPipeline,
    PipelineConfig,
)
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import (
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    SFTOptimizerParams,
)
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import Knowledge

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

    # 4. Pipeline Config
    k1 = Knowledge(
        title="numrs2", content=Path("numrs2.md").read_text()
    )
    k2 = Knowledge(
        title="numrs2_exp", content=Path("numrs2_exp.md").read_text()
    )
    k3 = Knowledge(
        title="numrs2_cos", content=Path("numrs2_cos.md").read_text()
    )
    k4 = Knowledge(
        title="numrs2_arithmetic",
        content=Path("numrs2_arithmetic.md").read_text(),
    )
    config = PipelineConfig(
        # knowledge_list=[k1, k2, k3, k4],
        knowledge_list=[k1],
        runtime_settings=runtime_settings,
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B", lora_rank=32
        ),
        sft_optimizer_params=SFTOptimizerParams(
            adam_params=tinker.AdamParams(learning_rate=1e-3),
            batch_size=32,
            num_epochs=5,
        ),
        rl_optimizer_params=OptimizerParams(
            adam_params=tinker.AdamParams(learning_rate=1e-4),
            num_steps=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            loss_fn="importance_sampling",
        ),
        experiment_settings=ExperimentSettings.with_prefix(
            "Primitive_SFT_Internalization"
        ),
        k_sft=8,
        k_rl=4,
        k_rollout=8,
        concurrency=32,
        max_iterations=10,
        max_sft_knowledge=8,
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
