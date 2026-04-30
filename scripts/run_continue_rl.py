"""SFT/RL pipeline driver.

Three named recipes; pick one via `CONFIG = ...` near the bottom.

  - SFT_RECIPE:     fresh LoRA, run initial SFT (granular knowledge × k_sft +
                    study trajectory distilled QRAs), then a single eval on
                    gh_archive[150:200]. No RL.
  - RL_RECIPE:      resume from an SFT checkpoint, skip SFT, build knowledge-
                    derived RL + eval suites (`build_knowledge_suites`), run
                    the GRPO loop.
  - TASK_RL_RECIPE: resume from an SFT checkpoint, skip SFT, RL directly on
                    gh_archive[0:150] (the real task), eval on
                    gh_archive[150:200].
"""

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from prisma import Prisma

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import (
    build_knowledge_suites,
    load_gh_archive_suite,
    load_granular_knowledge,
    load_study_root_qras_cached,
)
from adapter_agent.simple_internalizer.types import (
    CheckpointSettings,
    EvalSettings,
    RLConfig,
    RolloutSettings,
    SeedSuite,
    SFTConfig,
)
from adapter_agent.study.qra_distiller import QRADistiller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

GRANULAR_ID = "granular_prep_20260430_055104"

# SFT-completed checkpoint (output of SFT_RECIPE).
_SFT_CHECKPOINT_BASE_8B = "tinker://b8d2d31a-ed4d-511f-bd4b-956eaccdc204:train:0"
_SFT_CHECKPOINT_BASE_32B = "tinker://c3ce4acc-191b-5e0b-8a98-27995fac5384:train:0"


# --- Shared sub-configs (re-used across recipes) ---
_RUNTIME = RuntimeSettings.cloudrun_numrs2()

_ROLLOUT = RolloutSettings(
    runtime_settings=_RUNTIME,
    num_samples=8,
    runtime_pool_size=50,
    worker_count=50,
)

_EVAL = EvalSettings(
    eval_rollout=4,
    eval_interval=5,
    eval_concurrency=48,
)

_CHECKPOINT = CheckpointSettings(checkpoint_interval=10)


@dataclass(frozen=True)
class RunRecipe:
    """End-to-end pipeline recipe. simple_train_id gets timestamped at run time."""

    simple_train_id_prefix: str
    granular_id: str
    pipeline_config: PipelineConfig  # simple_train_id="" placeholder; filled in main()

    # Suite wiring (knowledge-derived suites built via build_knowledge_suites).
    rl_k_per_knowledge: int  # 0 disables knowledge-derived RL suite
    eval_k_per_knowledge: int  # 0 disables knowledge-derived eval suite
    gh_archive_rl_slice: slice | None  # gh_archive RL suite slice; None disables
    gh_archive_eval_slice: slice | None  # additional gh_archive eval slice; None disables

    # SFT extras (only consumed when pipeline_config.sft is set).
    sft_study_experiment_name: str | None
    sft_traj_qra_id: str | None


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

# SFT-only run: fresh LoRA → granular SFT (k_sft=32) + study trajectory QRAs
# → single gh_archive[150:200] eval → exit.
SFT_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft",
    granular_id=GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",  # filled at run time
        library_name="numrs2",
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
            k_sft=32,
            epochs=2,
            batch_size=128,
            sft_seed=42,
            save_checkpoint=True,
        ),
        rl=None,  # SFT-only run; pipeline runs final eval and exits.
        generation_concurrency=400,
    ),
    rl_k_per_knowledge=0,
    eval_k_per_knowledge=0,
    gh_archive_rl_slice=None,
    gh_archive_eval_slice=slice(150, 200),
    sft_study_experiment_name="study_20260430_024306",
    sft_traj_qra_id="study_20260430_024306_v1",
)


# RL run: resume SFT_RECIPE's checkpoint → skip SFT → knowledge-derived RL +
# eval suites (build_knowledge_suites) → GRPO loop.
RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl",
    granular_id=GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_SFT_CHECKPOINT_BASE_32B}/weights/init_sft",
            resume_sampler_path=f"{_SFT_CHECKPOINT_BASE_32B}/sampler_weights/init_sft",
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,  # SFT already done; checkpoint loaded.
        rl=RLConfig(
            # ~1 pass over the dataset: tasks (= rl_k_per_knowledge × |knowledge|
            # = 32 × 70 = 2,240) divided by batch_size (48) → ceil(2240/48) = 47.
            max_iterations=47,
            adam_params=tinker.AdamParams(learning_rate=7e-5),
            loss_fn="ppo",
            batch_size=48,
            update_epochs=1,
            max_version_lag=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            skip_update=False,
        ),
        generation_concurrency=400,
    ),
    rl_k_per_knowledge=32,
    eval_k_per_knowledge=1,
    gh_archive_rl_slice=None,
    gh_archive_eval_slice=None,
    sft_study_experiment_name=None,
    sft_traj_qra_id=None,
)


# Task RL run: resume SFT_RECIPE's checkpoint → skip SFT → RL directly on
# gh_archive[0:150] (the real task) → eval on gh_archive[150:200].
_TASK_RL_CHECKPOINT_BASE = "tinker://4d530b2a-8dff-5335-a4fc-eb5e78fa797b:train:0"
_TASK_RL_ROLLOUT = dataclasses.replace(_ROLLOUT, num_samples=16)

TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task",
    granular_id=GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_TASK_RL_CHECKPOINT_BASE}/weights/rl_0040",
            resume_sampler_path=f"{_TASK_RL_CHECKPOINT_BASE}/sampler_weights/rl_0040",
            lora_rank=32,
        ),
        rollout=_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,  # SFT already done; checkpoint loaded.
        rl=RLConfig(
            # 150 tasks / batch_size 48 → ceil(150/48) = 4 iterations per pass.
            max_iterations=10,
            adam_params=tinker.AdamParams(learning_rate=7e-5),
            loss_fn="ppo",
            batch_size=48,
            update_epochs=1,
            max_version_lag=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            skip_update=False,
        ),
        generation_concurrency=400,
    ),
    rl_k_per_knowledge=0,
    eval_k_per_knowledge=0,
    gh_archive_rl_slice=slice(0, 150),
    gh_archive_eval_slice=slice(150, 200),
    sft_study_experiment_name=None,
    sft_traj_qra_id=None,
)
# これによって出来上がったチェックポイント
# "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010"

CONFIG = TASK_RL_RECIPE


# ---------------------------------------------------------------------------


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    json_path = Path("repositories/numrs/target/doc/numrs2.json")
    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    simple_train_id = (
        f"{cfg.simple_train_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    pipeline_config = dataclasses.replace(
        cfg.pipeline_config, simple_train_id=simple_train_id
    )
    logger.info(
        f"Recipe: {cfg.simple_train_id_prefix} (sft="
        f"{'on' if pipeline_config.sft is not None else 'off'}, "
        f"resume={'yes' if pipeline_config.model_loading_settings.resume_trainer_path else 'no'})"
    )

    prisma = Prisma()
    await prisma.connect()
    try:
        knowledge_list = await load_granular_knowledge(prisma, cfg.granular_id)
        logger.info(
            f"Loaded {len(knowledge_list)} granular knowledge rows from '{cfg.granular_id}'."
        )

        seed_suites: list[SeedSuite] = []
        if cfg.rl_k_per_knowledge > 0 or cfg.eval_k_per_knowledge > 0:
            generator = GeneratorAgent(model=get_gemini())

            if cfg.rl_k_per_knowledge > 0:
                rl_suites = await build_knowledge_suites(
                    generator=generator,
                    knowledge_list=knowledge_list,
                    k_per_knowledge=cfg.rl_k_per_knowledge,
                    cache_dir=pipeline_config.cache_dir,
                    name_prefix="knowledge_rl",
                    for_rl=True,
                    for_eval=False,
                    generation_concurrency=pipeline_config.generation_concurrency,
                )
                logger.info(
                    f"Built {len(rl_suites)} knowledge-RL suites "
                    f"(k_per_knowledge={cfg.rl_k_per_knowledge})."
                )
                seed_suites.extend(rl_suites)

            if cfg.eval_k_per_knowledge > 0:
                eval_suites = await build_knowledge_suites(
                    generator=generator,
                    knowledge_list=knowledge_list,
                    k_per_knowledge=cfg.eval_k_per_knowledge,
                    cache_dir=pipeline_config.cache_dir,
                    name_prefix="knowledge_eval",
                    for_rl=False,
                    for_eval=True,
                    generation_concurrency=pipeline_config.generation_concurrency,
                )
                logger.info(
                    f"Built {len(eval_suites)} knowledge-eval suites "
                    f"(k_per_knowledge={cfg.eval_k_per_knowledge})."
                )
                seed_suites.extend(eval_suites)

        if cfg.gh_archive_rl_slice is not None:
            sl = cfg.gh_archive_rl_slice
            gh_rl_suite = load_gh_archive_suite(
                name="gh_archive_rl",
                task_slice=sl,
                for_rl=True,
                for_eval=False,
            )
            step_repr = f":{sl.step}" if sl.step is not None else ""
            logger.info(
                f"gh_archive RL suite: {len(gh_rl_suite.tasks)} tasks "
                f"(slice=[{sl.start}:{sl.stop}{step_repr}])."
            )
            seed_suites.append(gh_rl_suite)

        if cfg.gh_archive_eval_slice is not None:
            sl = cfg.gh_archive_eval_slice
            gh_eval_suite = load_gh_archive_suite(
                name="gh_archive_eval",
                task_slice=sl,
                for_rl=False,
                for_eval=True,
            )
            step_repr = f":{sl.step}" if sl.step is not None else ""
            logger.info(
                f"gh_archive eval suite: {len(gh_eval_suite.tasks)} tasks "
                f"(slice=[{sl.start}:{sl.stop}{step_repr}])."
            )
            seed_suites.append(gh_eval_suite)

        extra_sft_qras: list[QRA] = []
        if (
            pipeline_config.sft is not None
            and cfg.sft_study_experiment_name
            and cfg.sft_traj_qra_id
        ):
            distiller = QRADistiller(model=get_gemini())
            extra_sft_qras = await load_study_root_qras_cached(
                prisma_client=prisma,
                experiment_name=cfg.sft_study_experiment_name,
                traj_qra_id=cfg.sft_traj_qra_id,
                distiller=distiller,
                cache_dir=pipeline_config.cache_dir,
            )
            logger.info(
                f"Loaded {len(extra_sft_qras)} study root QRAs from "
                f"'{cfg.sft_study_experiment_name}'."
            )
    finally:
        await prisma.disconnect()

    pipeline = await SimplePipeline.create(
        config=pipeline_config,
        knowledge_list=knowledge_list,
        seed_suites=seed_suites,
        extra_sft_qras=extra_sft_qras,
    )

    try:
        logger.info("Starting pipeline execution.")
        await pipeline.run()
        logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
