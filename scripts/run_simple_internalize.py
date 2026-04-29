import asyncio
import logging
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv

from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import (
    build_knowledge_suites,
    load_granular_knowledge,
    load_study_solved_suite,
)
from adapter_agent.simple_internalizer.types import (
    CheckpointSettings,
    EvalSettings,
    RolloutSettings,
    SFTConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

GRANULAR_ID = "granular_prep_20260419_024056"
STUDY_EXPERIMENT_ID = "study_20260418_233708"
K_RL_PER_KNOWLEDGE = 8
K_EVAL_PER_KNOWLEDGE = 1
GENERATION_CONCURRENCY = 400
CACHE_DIR = Path(".cache/simple_internalizer")


async def main() -> None:
    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")

    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    logger.info("Setting up simplified internalization pipeline...")

    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    simple_train_id = f"simple_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    db_manager = PostgresDB()
    await db_manager.connect()
    prisma_client = await db_manager.get_client()

    knowledge_list = await load_granular_knowledge(prisma_client, GRANULAR_ID)
    logger.info(f"Loaded {len(knowledge_list)} granular knowledge items.")

    generator = GeneratorAgent(model=get_gemini(), rust_doc_analyzer=analyzer)

    knowledge_rl_suites = await build_knowledge_suites(
        generator=generator,
        knowledge_list=knowledge_list,
        k_per_knowledge=K_RL_PER_KNOWLEDGE,
        cache_dir=CACHE_DIR,
        name_prefix="knowledge_rl",
        for_rl=True,
        for_eval=False,
        generation_concurrency=GENERATION_CONCURRENCY,
    )
    knowledge_eval_suites = await build_knowledge_suites(
        generator=generator,
        knowledge_list=knowledge_list,
        k_per_knowledge=K_EVAL_PER_KNOWLEDGE,
        cache_dir=CACHE_DIR,
        name_prefix="knowledge_eval",
        for_rl=False,
        for_eval=True,
        generation_concurrency=GENERATION_CONCURRENCY,
    )

    study_solved_suite = await load_study_solved_suite(
        prisma_client, STUDY_EXPERIMENT_ID
    )
    logger.info(
        f"Loaded study_solved suite with {len(study_solved_suite.tasks)} tasks."
    )

    seed_suites = [
        *knowledge_rl_suites,
        *knowledge_eval_suites,
        study_solved_suite,
    ]

    config = PipelineConfig(
        simple_train_id=simple_train_id,
        library_name="numrs2",
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-8B",
            resume_trainer_path=None,
            resume_sampler_path=None,
            lora_rank=32,
        ),
        rollout=RolloutSettings(
            runtime_settings=runtime_settings,
            num_samples=8,
            runtime_pool_size=100,
            worker_count=48,
        ),
        eval=EvalSettings(
            eval_rollout=4,
            eval_concurrency=48,
        ),
        checkpoint=CheckpointSettings(checkpoint_interval=10),
        sft=SFTConfig(
            k_sft=8,
            epochs=8,
            batch_size=256,
            adam_params=tinker.AdamParams(learning_rate=1e-3),
            cpt=False,
        ),
        cache_dir=CACHE_DIR,
        max_iterations=50,
        generation_concurrency=GENERATION_CONCURRENCY,
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="cispo",
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
    )

    pipeline = await SimplePipeline.create(
        config=config,
        rust_doc_analyzer=analyzer,
        knowledge_list=knowledge_list,
        seed_suites=seed_suites,
    )

    try:
        async with analyzer:
            logger.info("Starting simple pipeline execution.")
            await pipeline.run()
            logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
