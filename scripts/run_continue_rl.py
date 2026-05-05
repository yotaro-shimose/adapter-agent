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
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import tinker
from agents import set_tracing_disabled
from dotenv import load_dotenv
from prisma import Prisma

from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import (
    build_knowledge_suites,
    load_gh_archive_suite,
    load_granular_knowledge,
)
from adapter_agent.simple_internalizer.sft_qra_loaders import (
    SftLoaderContext,
    SftSuiteFactory,
    load_granular_sft_suite,
    load_sft_cache_suite,
    load_study_root_sft_suite,
)
from adapter_agent.simple_internalizer.types import (
    CheckpointSettings,
    EvalSettings,
    RLConfig,
    RolloutSettings,
    SeedSuite,
    SFTConfig,
    SftSuite,
    default_cache_dir,
)
from adapter_agent.study.qra_distiller import QRADistiller

# Disable OpenAI Agents SDK telemetry posts. With generation_concurrency=400,
# the SDK's tracing client floods OpenAI's /v1/traces endpoint and gets 429'd
# even though the underlying Gemini calls are semaphore-throttled.
set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

GRANULAR_ID = "granular_prep_20260430_055104"

# Hisab study lineage. Output of `prepare_granular_knowledge.py` running
# HISAB_STUDY_PREP against study_20260504_070444 with no path prefix
# (8 api/ + 15 concepts/ + 1 MOC = 24 articles total).
HISAB_GRANULAR_ID = "granular_prep_hisab_20260504_073544"

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

# hisab variants — same shape as the numrs2 ones, only the runtime image
# (and downstream the library identity) differs.
_RUNTIME_HISAB = LibrarySpec.hisab().cloudrun_runtime()
_ROLLOUT_HISAB = dataclasses.replace(_ROLLOUT, runtime_settings=_RUNTIME_HISAB)

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
    gh_archive_eval_slice: (
        slice | None
    )  # additional gh_archive eval slice; None disables

    # SFT data sources (only consumed when pipeline_config.sft is set).
    # Each entry is a `partial`-bound loader that, given the per-run
    # SftLoaderContext, returns one `SftSuite`. The pipeline flattens all
    # returned suites into the SFT pool. Empty list = no SFT data.
    sft_sources: list[SftSuiteFactory] = field(default_factory=list)

    # Library identity. Drives rustdoc JSON path (startup sanity check) and
    # the gh_archive benchmark CSV that build_*/load_gh_archive_suite reads.
    # Default keeps numrs2-era recipes unchanged; hisab recipes override.
    library_spec: LibrarySpec = field(default_factory=LibrarySpec.numrs2)

    @property
    def json_path(self) -> Path:
        return (
            self.library_spec.libdir / "target/doc" / f"{self.library_spec.name}.json"
        )


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
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
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
    sft_sources=[
        partial(
            load_granular_sft_suite,
            name="granular_numrs2",
            cache_id="sft__numrs2",
            k_per_knowledge=32,
        ),
        partial(
            load_study_root_sft_suite,
            name="study_root_numrs2",
            experiment_name="study_20260430_024306",
            traj_qra_id="study_20260430_024306_v1",
        ),
    ],
)


# RL run: resume SFT_RECIPE's checkpoint → skip SFT → knowledge-derived RL +
# eval suites (build_knowledge_suites) → GRPO loop.
RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl",
    granular_id=GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
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
            # Short probe run (10 updates) under the new stub-rejecting verifier
            # + anti-stub solver prompt — to check whether the stub strategy
            # actually erodes within a handful of GRPO updates.
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
    rl_k_per_knowledge=32,
    eval_k_per_knowledge=1,
    gh_archive_rl_slice=None,
    gh_archive_eval_slice=None,
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
        cache_dir=default_cache_dir("numrs2"),
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
)
# これによって出来上がったチェックポイント
# "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010"


# ---------------------------------------------------------------------------
# Hisab recipes
# ---------------------------------------------------------------------------

# Hisab equivalent of SFT_RECIPE: fresh LoRA → granular SFT (k_sft=32 over
# hisab granular knowledge from study_20260504_070444) + study trajectory
# QRAs distilled from the same experiment → exit (SFT-only).
#
# `gh_archive_eval_slice=None` because `load_gh_archive_suite` still pulls
# the numrs2 benchmark CSV — flip this to `slice(150, 200)` once the suite
# loader is plumbed for hisab's `data/benchmarks/hisab_2026-05-04/` source.
HISAB_SFT_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_hisab",
    granular_id=HISAB_GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            lora_rank=32,
        ),
        rollout=_ROLLOUT_HISAB,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
            epochs=2,
            batch_size=128,
            sft_seed=42,
            save_checkpoint=True,
        ),
        rl=None,
        generation_concurrency=400,
    ),
    rl_k_per_knowledge=0,
    eval_k_per_knowledge=0,
    gh_archive_rl_slice=None,
    gh_archive_eval_slice=slice(150, 200),
    sft_sources=[
        partial(
            load_granular_sft_suite,
            name="granular_hisab",
            cache_id="sft__hisab",
            k_per_knowledge=32,
        ),
        partial(
            load_study_root_sft_suite,
            name="study_root_hisab",
            experiment_name="study_20260504_070444",
            traj_qra_id="study_20260504_070444_v1",
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# Hisab SFT, but using only the augmentation-pipeline output (no granular,
# no study-root). The QRAs in `pipeline_v1_qra` are verified Q/R/A triples
# from `study2_pipeline.py`; this recipe trains exclusively on those.
HISAB_SFT_FROM_PIPELINE_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_hisab_pipeline",
    granular_id=GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            lora_rank=32,
        ),
        rollout=_ROLLOUT_HISAB,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
            epochs=2,
            batch_size=128,
            sft_seed=42,
            save_checkpoint=True,
        ),
        rl=None,
        generation_concurrency=400,
    ),
    rl_k_per_knowledge=0,
    eval_k_per_knowledge=0,
    gh_archive_rl_slice=None,
    gh_archive_eval_slice=slice(150, 200),
    sft_sources=[
        partial(
            load_sft_cache_suite,
            name="pipeline_v1_qra",
            cache_id="pipeline_v1_qra",
            verified_only=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# Output of HISAB_SFT_RECIPE (see /tmp/hisab-sft.log).
_HISAB_SFT_CHECKPOINT_BASE = "tinker://25175663-6abf-5703-90ad-0a92081da02e:train:0"


# Hisab equivalent of RL_RECIPE: resume from HISAB_SFT_RECIPE's checkpoint →
# skip SFT → knowledge-derived RL + eval suites (build_knowledge_suites) over
# the 101 hisab granular knowledge entries → GRPO loop.
HISAB_KNOWLEDGE_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_hisab",
    granular_id=HISAB_GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_SFT_CHECKPOINT_BASE}/weights/init_sft",
            resume_sampler_path=f"{_HISAB_SFT_CHECKPOINT_BASE}/sampler_weights/init_sft",
            lora_rank=32,
        ),
        rollout=_ROLLOUT_HISAB,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            num_passes=1,
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
    library_spec=LibrarySpec.hisab(),
)


# Hisab equivalent of TASK_RL_RECIPE: RL directly on hisab gh_archive[0:150],
# eval on [150:200]. Resumes from HISAB_KNOWLEDGE_RL_RECIPE's rl_0060 (1-pass
# run terminated at iter 60 because num_passes=1 stopped before iter 68 and
# the old code didn't save a final checkpoint — see /tmp/hisab-knowledge-rl.log).
_HISAB_TASK_RL_CHECKPOINT_BASE = "tinker://a7e97833-7934-558e-842d-a29f8a2bd48f:train:0"
_HISAB_TASK_RL_CHECKPOINT_NAME = "rl_0060"
_HISAB_TASK_RL_ROLLOUT = dataclasses.replace(_ROLLOUT_HISAB, num_samples=16)

HISAB_TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab",
    granular_id=HISAB_GRANULAR_ID,
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_TASK_RL_CHECKPOINT_BASE}/weights/{_HISAB_TASK_RL_CHECKPOINT_NAME}",
            resume_sampler_path=f"{_HISAB_TASK_RL_CHECKPOINT_BASE}/sampler_weights/{_HISAB_TASK_RL_CHECKPOINT_NAME}",
            lora_rank=32,
        ),
        rollout=_HISAB_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            num_passes=20,
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
    library_spec=LibrarySpec.hisab(),
)


CONFIG = HISAB_KNOWLEDGE_RL_RECIPE


# ---------------------------------------------------------------------------


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    if not cfg.json_path.exists():
        logger.error(f"RustDoc JSON not found at {cfg.json_path}")
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
        # Granular knowledge is only consumed by `build_knowledge_suites`
        # and the granular SFT loader. Other SFT loaders (sft_cache,
        # study-root) don't need it. Skip the DB query when nothing wants it.
        needs_granular_sft_loader = (
            pipeline_config.sft is not None
            and any(
                getattr(s, "func", None) is load_granular_sft_suite
                for s in cfg.sft_sources
            )
        )
        needs_knowledge = (
            needs_granular_sft_loader
            or cfg.rl_k_per_knowledge > 0
            or cfg.eval_k_per_knowledge > 0
        )
        if needs_knowledge:
            knowledge_list = await load_granular_knowledge(prisma, cfg.granular_id)
            logger.info(
                f"Loaded {len(knowledge_list)} granular knowledge rows from '{cfg.granular_id}'."
            )
        else:
            knowledge_list = []
            logger.info(
                "Skipping granular knowledge load (no SFT and no knowledge-derived suites)."
            )

        seed_suites: list[SeedSuite] = []
        if cfg.rl_k_per_knowledge > 0 or cfg.eval_k_per_knowledge > 0:
            generator = GeneratorAgent(model=get_gemini())

            if cfg.rl_k_per_knowledge > 0:
                rl_suites = await build_knowledge_suites(
                    generator=generator,
                    knowledge_list=knowledge_list,
                    k_per_knowledge=cfg.rl_k_per_knowledge,
                    cache_id=f"knowledge_rl__{cfg.library_spec.name}",
                    prisma_client=prisma,
                    name_prefix="knowledge_rl",
                    granular_id=cfg.granular_id,
                    library_name=cfg.library_spec.name,
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
                    cache_id=f"knowledge_eval__{cfg.library_spec.name}",
                    prisma_client=prisma,
                    name_prefix="knowledge_eval",
                    granular_id=cfg.granular_id,
                    library_name=cfg.library_spec.name,
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
                csv_path=cfg.library_spec.benchmark_csv,
                difficulty=cfg.library_spec.default_difficulty,
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
                csv_path=cfg.library_spec.benchmark_csv,
                difficulty=cfg.library_spec.default_difficulty,
            )
            step_repr = f":{sl.step}" if sl.step is not None else ""
            logger.info(
                f"gh_archive eval suite: {len(gh_eval_suite.tasks)} tasks "
                f"(slice=[{sl.start}:{sl.stop}{step_repr}])."
            )
            seed_suites.append(gh_eval_suite)

        sft_suites: list[SftSuite] = []
        if pipeline_config.sft is not None and cfg.sft_sources:
            # Build deps eagerly — they're cheap (model client wrappers) and
            # individual loaders pick what they need from `ctx`.
            sft_ctx = SftLoaderContext(
                prisma=prisma,
                library_name=cfg.library_spec.name,
                cache_dir=pipeline_config.cache_dir,
                generation_concurrency=pipeline_config.generation_concurrency,
                knowledge_list=knowledge_list,
                generator=GeneratorAgent(model=get_gemini()),
                distiller=QRADistiller(model=get_gemini()),
            )
            sft_suites = [await src(sft_ctx) for src in cfg.sft_sources]
            for s in sft_suites:
                logger.info(f"SFT source loaded: {s.name} -> {len(s.qras)} QRAs")
    finally:
        await prisma.disconnect()

    pipeline = await SimplePipeline.create(
        config=pipeline_config,
        knowledge_list=knowledge_list,
        seed_suites=seed_suites,
        sft_suites=sft_suites,
    )

    try:
        logger.info("Starting pipeline execution.")
        await pipeline.run()
        logger.info("Pipeline executed successfully.")
    except Exception as e:
        logger.exception(f"Pipeline encountered an error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
