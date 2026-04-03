import asyncio
import logging
from pathlib import Path

import tinker

from adapter_agent.hierarchical.pipeline.internalization_pipeline import (
    InternalizationPipeline,
    PipelineConfig,
)
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

logger = logging.getLogger(__name__)


class MarkdownAnalyzer:
    """
    Minimal analyzer that returns the content of a Markdown file as the overview.
    Used for primitive-only internalization tests.
    """

    def __init__(self, path: Path):
        self.path = path

    def get_overview(self) -> str:
        if not self.path.exists():
            return f"Error: {self.path} not found."
        return self.path.read_text(encoding="utf-8")

    def get_modules(self) -> list[str]:
        """Return a list of modules for topic hinting. For this test, just one."""
        return ["numrs2::array"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def main():
    api_doc_path = Path("numrs2.md")

    if not api_doc_path.exists():
        logger.error(f"API documentation not found at {api_doc_path}")
        return

    logger.info("Setting up primitive-only internalization pipeline...")

    # 1. Analyzer (Minimal Markdown)
    analyzer = MarkdownAnalyzer(api_doc_path)

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
    config = PipelineConfig(
        api_doc_path=api_doc_path,
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
    )

    # 5. Pipeline
    pipeline = InternalizationPipeline(
        config=config,
        generator_model=model,
        verifier_model=model,  # Using Gemini for both generation and verification
        rust_doc_analyzer=analyzer,  # type: ignore
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
