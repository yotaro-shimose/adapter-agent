import asyncio
import logging
from pathlib import Path

import ray
import tinker
from oai_utils.tinker import setup_tinkermodel

from adapter_agent.hierarchical.gh import Library
from adapter_agent.internalize.internalizer import Internalizer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
# Only enable DEBUG for our own modules
logging.getLogger("adapter_agent").setLevel(logging.DEBUG)
# Suppress noisy logs
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def main():
    # 0. Initialize Ray
    logger.info("Initializing Ray...")
    ray.init(
        configure_logging=True,
        logging_config=ray.LoggingConfig(
            log_level="INFO", additional_log_standard_attrs=["name"]
        ),
        runtime_env={"env_vars": {"RAY_DEBUG": "1"}},
    )

    # 1. Configuration
    model_name = "Qwen/Qwen3-8B"
    library_name = "numrs2"
    json_path = Path("repositories/numrs/target/doc/numrs2.json")
    # Experiment ID to pull knowledge from
    source_experiment_id = "unirl_20260406_102313"

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    # 2. Setup Library
    logger.info(f"Setting up library for {library_name}...")
    library = Library(name=library_name, local_path=Path("repositories/numrs"))

    # 3. Setup Knowledge Source
    logger.info(f"Fetching knowledge from DB (Exp ID: {source_experiment_id})...")
    db = KnowledgeDB.for_experiment(source_experiment_id)
    async with db:
        knowledges = await db.list_knowledge(limit=10)

    if not knowledges:
        logger.error("No knowledge items found in the database. Aborting.")
        return
    logger.info(f"Loaded {len(knowledges)} knowledge items.")

    # 4. Setup Tinker & SharedSamplingClient
    logger.info("Setting up Tinker clients...")
    # Note: resume_sampler_path/resume_trainer_path could be added here if needed
    tinker_model, _, _ = setup_tinkermodel(model_name=model_name)
    sampling_client = SharedSamplingClient(tinker_model.sampling_client)

    # 6. Runtime Settings
    runtime_settings = RuntimeSettings(
        type="docker", image_uri="coder-mcp-numrs2:latest"
    )

    # 7. Start Internalizer
    logger.info("Starting Internalizer and Ray actors...")
    internalizer = await Internalizer.start(
        knowledges=knowledges,
        sampling_client=sampling_client,
        library=library,
        model_name=model_name,
        runtime_settings=runtime_settings,
        sft_batch_size=64,
        sft_adam_params=tinker.AdamParams(learning_rate=1e-3),
        rl_adam_params=tinker.AdamParams(learning_rate=3e-4),
        rl_loss_fn="ppo",
        min_sft_batch_size=64,
        min_rl_batch_size=8,
        k_rollout=8,
        studying_threshold=0.2,
        success_threshold=0.5,
        overgen_factor=1.5,
        k_sft=8,
        k_rl=4,
        num_workers=4,
        max_iterations=50,
    )

    # 8. Run the Internalization Experiment
    try:
        await internalizer.run()
    except Exception as e:
        logger.exception(f"Experiment execution failed: {e}")
    finally:
        logger.info("Cleaning up...")
    # Add any necessary cleanup here (e.g. closing analyzer if it has open resources)
    # ray.shutdown() # Internalizer start doesn't imply it owns the ray context


if __name__ == "__main__":
    asyncio.run(main())
