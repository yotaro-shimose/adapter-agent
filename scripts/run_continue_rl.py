"""SFT/RL pipeline driver.

Active recipes (pick one via `CONFIG = ...` near the bottom):

  Hisab (post-decomposition lineage):
    - HISAB_SFT_FROM_PIPELINE_V2_RECIPE:    fresh LoRA → SFT on
                                            `pipeline_v2_qra` → eval.
    - HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE: resume QRA-SFT → RL on
                                             `pipeline_v2_qra` seeds.
    - HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE:  resume QRA-RL → Task RL on
                                             `pipeline_v3_decomposed_qra_train`.

  Numrs2 (analogues of the hisab pipeline-v2 recipes):
    - NUMRS2_KNOWLEDGE_SFT_RECIPE:  fresh LoRA → SFT on
                                    `pipeline_v2_qra_numrs2` → eval.
    - NUMRS2_KNOWLEDGE_RL_RECIPE:   resume QRA-SFT → RL on
                                    `pipeline_v2_qra_numrs2` seeds.
    - NUMRS2_TASK_RL_RECIPE:        resume Knowledge-RL → Task RL on
                                    numrs2 gh_archive[0:150].

  Numrs2 RESTUDY lineage (branches off NUMRS2_TASK_RL_RECIPE output —
  fills gaps surfaced by `passatk_restudy.py` routing):
    - NUMRS2_RESTUDY_KSFT_RECIPE:     resume Task-RL → 1-epoch SFT on
                                      `restudy_numrs2_qra`.
    - NUMRS2_RESTUDY_KRL_RECIPE:      resume restudy KSFT → RL on the
                                      SFT+RL bucket subset of
                                      `restudy_numrs2_qra`.
    - NUMRS2_RESTUDY_TASK_RL_RECIPE:  resume restudy KRL → Task RL on
                                      numrs2 gh_archive[0:150], eval on
                                      gh_archive[150:200].

  Hisab RESTUDY lineage (mirror of the numrs2 restudy lineage —
  branches off HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE output):
    - HISAB_RESTUDY_KSFT_RECIPE:      resume hisab Task-RL → 1-epoch SFT
                                      on `restudy_hisab_qra`.
    - HISAB_RESTUDY_KRL_RECIPE:       resume restudy KSFT → RL on SFT+RL
                                      bucket of `restudy_hisab_qra` mixed
                                      50/50 with `pipeline_v2_qra` replay.
    - HISAB_RESTUDY_TASK_RL_RECIPE:   resume restudy KRL → Task RL on
                                      hisab gh_archive[0:150], eval on
                                      gh_archive[150:200].

Retired / legacy recipes (granular-knowledge SFT, knowledge RL, the v1
augmentation pipeline, pre-decomposed Task RL, etc.) live in
`scripts/legacy/train_recipes.py`. They share this module's `RunRecipe`
type and shared sub-configs via import.
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
    load_granular_knowledge,
)
from adapter_agent.simple_internalizer.seed_loaders import (
    SeedLoaderContext,
    SeedSuiteFactory,
    load_gh_archive_seed_suite,
    load_sft_cache_seed_suite_factory,
    load_sft_cache_seed_suite_factory_bucket_filtered,
)
from adapter_agent.simple_internalizer.sft_qra_loaders import (
    SftLoaderContext,
    SftSuiteFactory,
    load_rl_rollout_replay_suite,
    load_sft_cache_suite,
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


# --- Shared sub-configs (re-used across recipes; legacy recipes import these) ---
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

_CHECKPOINT = CheckpointSettings(checkpoint_interval=5)


@dataclass(frozen=True)
class RunRecipe:
    """End-to-end pipeline recipe. simple_train_id gets timestamped at run time."""

    simple_train_id_prefix: str
    pipeline_config: PipelineConfig  # simple_train_id="" placeholder; filled in main()

    # Granular-knowledge ID. Loaded into ctx.knowledge_list iff non-None,
    # so SFT/seed factories that need knowledge can pull it. Recipes that
    # don't touch granular knowledge can leave this as None.
    granular_id: str | None = None

    # SFT data sources (only consumed when pipeline_config.sft is set).
    # Each entry is a `partial`-bound loader that, given the per-run
    # SftLoaderContext, returns one `SftSuite`. The pipeline flattens all
    # returned suites into the SFT pool. Empty list = no SFT data.
    sft_sources: list[SftSuiteFactory] = field(default_factory=list)

    # RL / eval seed-suite sources. Each entry is a `partial`-bound async
    # callable that, given the per-run SeedLoaderContext, returns a list
    # of `SeedSuite`s (`for_rl` / `for_eval` already baked in). The pipeline
    # flattens everything into a single seed pool.
    seed_sources: list[SeedSuiteFactory] = field(default_factory=list)

    # Library identity. Drives rustdoc JSON path (startup sanity check) and
    # the gh_archive benchmark CSV that the seed/SFT loaders read.
    library_spec: LibrarySpec = field(default_factory=LibrarySpec.numrs2)

    @property
    def json_path(self) -> Path:
        return (
            self.library_spec.libdir / "target/doc" / f"{self.library_spec.name}.json"
        )


# ---------------------------------------------------------------------------
# Hisab active recipes (post-decomposition lineage)
# ---------------------------------------------------------------------------

# Hisab SFT, training exclusively on the v2 augmentation cache produced by
# study2_pipeline.py FULL_PIPELINE_V2 (150 tasks × 5 items × 8 variants).
HISAB_SFT_FROM_PIPELINE_V2_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_hisab_pipeline_v2",
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
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    sft_sources=[
        partial(
            load_sft_cache_suite,
            name="pipeline_v2_qra",
            cache_id="pipeline_v2_qra",
            verified_only=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# RL on the v2 QRA cache. Resumes from the v2 QRA-SFT checkpoint (output of
# HISAB_SFT_FROM_PIPELINE_V2_RECIPE).
_HISAB_QRA_SFT_V2_CHECKPOINT_BASE = (
    "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0"
)

HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_hisab_from_qra_v2",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_QRA_SFT_V2_CHECKPOINT_BASE}/weights/init_sft",
            resume_sampler_path=f"{_HISAB_QRA_SFT_V2_CHECKPOINT_BASE}/sampler_weights/init_sft",
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
    seed_sources=[
        partial(
            load_sft_cache_seed_suite_factory,
            name="pipeline_v2_qra_rl",
            cache_id="pipeline_v2_qra",
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# Task RL on the DECOMPOSED mid-level pool (verified rows of
# `pipeline_v3_decomposed_qra_train` — produced by scripts/study2_decompose.py
# from gh_archive[0:150]). Resumes from the v2 QRA-RL checkpoint, evals on
# gh_archive[150:200] for direct comparison with the other hisab Task-RL
# variants. The RL seed pool is the only thing that changes — instead of
# training on the 150 raw gh_archive tasks we train on the ~233 mid-level
# sub-tasks decomposed from them.
_HISAB_QRA_RL_V2_CHECKPOINT_BASE = (
    "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0"
)
_HISAB_QRA_RL_V2_CHECKPOINT_NAME = "rl_0054"
_HISAB_TASK_RL_ROLLOUT = dataclasses.replace(_ROLLOUT_HISAB, num_samples=16)

HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab_from_decomposed",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_QRA_RL_V2_CHECKPOINT_BASE}/weights/{_HISAB_QRA_RL_V2_CHECKPOINT_NAME}",
            resume_sampler_path=f"{_HISAB_QRA_RL_V2_CHECKPOINT_BASE}/sampler_weights/{_HISAB_QRA_RL_V2_CHECKPOINT_NAME}",
            lora_rank=32,
        ),
        rollout=_HISAB_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            num_passes=10,
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
    seed_sources=[
        partial(
            load_sft_cache_seed_suite_factory,
            name="decomposed_train_rl",
            cache_id="pipeline_v3_decomposed_qra_train",
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# ---------------------------------------------------------------------------
# Numrs2 active recipes (analogues of the hisab pipeline-v2 lineage)
# ---------------------------------------------------------------------------

# Numrs2 Knowledge SFT — analogue of HISAB_SFT_FROM_PIPELINE_V2_RECIPE.
# Trains a fresh LoRA on `pipeline_v2_qra_numrs2` (2208 verified QRAs from
# study2_pipeline.py FULL_PIPELINE_V2_NUMRS2). Eval on numrs2
# gh_archive[150:200] (Easy filter applied via spec.default_difficulty).
NUMRS2_KNOWLEDGE_SFT_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_numrs2_pipeline_v2",
    pipeline_config=PipelineConfig(
        simple_train_id="",
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
            # 2208 QRAs / batch_size 128 → ~17 batches/epoch × 2 = ~34 updates.
            # Mirrors HISAB_SFT_FROM_PIPELINE_V2_RECIPE's shape.
            epochs=2,
            batch_size=128,
            sft_seed=42,
            save_checkpoint=True,
        ),
        rl=None,
        generation_concurrency=400,
    ),
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    sft_sources=[
        partial(
            load_sft_cache_suite,
            name="pipeline_v2_qra_numrs2",
            cache_id="pipeline_v2_qra_numrs2",
            verified_only=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# Numrs2 Knowledge RL — analogue of HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE.
# Resumes from NUMRS2_KNOWLEDGE_SFT_RECIPE's checkpoint, skips SFT, RLs
# directly on `pipeline_v2_qra_numrs2` questions. Eval on numrs2
# gh_archive[150:200].
#
# Output of NUMRS2_KNOWLEDGE_SFT_RECIPE.
_NUMRS2_KNOWLEDGE_SFT_CHECKPOINT_BASE = (
    "tinker://4c6bb913-cf76-53c6-9d04-c6e1097e0cb0:train:0"
)

NUMRS2_KNOWLEDGE_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_numrs2_from_qra",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_NUMRS2_KNOWLEDGE_SFT_CHECKPOINT_BASE}/weights/init_sft",
            resume_sampler_path=f"{_NUMRS2_KNOWLEDGE_SFT_CHECKPOINT_BASE}/sampler_weights/init_sft",
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
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
    seed_sources=[
        partial(
            load_sft_cache_seed_suite_factory,
            name="pipeline_v2_qra_numrs2_rl",
            cache_id="pipeline_v2_qra_numrs2",
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# Numrs2 Task RL — analogue of the legacy HISAB_TASK_RL_RECIPE shape (direct
# gh_archive[0:150] training, no decomposition prerequisite). Resumes from
# NUMRS2_KNOWLEDGE_RL_RECIPE rl_0040 and RLs on the 150 numrs2 train tasks.
# Eval stays on gh_archive[150:200] (Easy filter via spec.default_difficulty)
# so success rate is comparable across the numrs2 RL lineage.
_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_BASE = (
    "tinker://caaa9922-b354-5e58-80f7-54262e4ca496:train:0"
)
_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_NAME = "rl_0040"
_NUMRS2_TASK_RL_ROLLOUT = dataclasses.replace(_ROLLOUT, num_samples=16)

NUMRS2_TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_numrs2",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_BASE}/weights/{_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_NAME}",
            resume_sampler_path=f"{_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_BASE}/sampler_weights/{_NUMRS2_KNOWLEDGE_RL_CHECKPOINT_NAME}",
            lora_rank=32,
        ),
        rollout=_NUMRS2_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            num_passes=10,
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
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_rl",
            task_slice=slice(0, 150),
            for_rl=True,
            for_eval=False,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# ---------------------------------------------------------------------------
# Numrs2 RESTUDY lineage (branch off NUMRS2_TASK_RL_RECIPE)
# ---------------------------------------------------------------------------
# Flow (run them in order, fill in the checkpoint URI of each previous stage
# into the next recipe's resume paths before running):
#
#   1. NUMRS2_RESTUDY_KSFT_RECIPE  — 1-epoch SFT on `restudy_numrs2_qra`.
#      Resumes from the NUMRS2_TASK_RL_RECIPE checkpoint (the same one
#      passatk_restudy.py evaluates). Output: KSFT checkpoint.
#
#   2. NUMRS2_RESTUDY_KRL_RECIPE   — RL on the SFT+RL bucket subset of
#      `restudy_numrs2_qra` (drops Proficient tasks via the routing CSV).
#      Resumes from the KSFT checkpoint. Output: KRL checkpoint.
#
#   3. NUMRS2_RESTUDY_TASK_RL_RECIPE — Task RL on numrs2 gh_archive[0:150],
#      eval on gh_archive[150:200]. Resumes from the KRL checkpoint.
#
# Checkpoint base for stage 1's resume = current Task RL output
# (= the model passatk_restudy.py evaluated).
_NUMRS2_TASK_RL_CHECKPOINT_BASE = (
    "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0"
)
_NUMRS2_TASK_RL_CHECKPOINT_NAME = "rl_0040"

NUMRS2_RESTUDY_KSFT_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_numrs2_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_NUMRS2_TASK_RL_CHECKPOINT_BASE}/weights/"
                f"{_NUMRS2_TASK_RL_CHECKPOINT_NAME}"
            ),
            resume_sampler_path=(
                f"{_NUMRS2_TASK_RL_CHECKPOINT_BASE}/sampler_weights/"
                f"{_NUMRS2_TASK_RL_CHECKPOINT_NAME}"
            ),
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
            # restudy QRA pool is small (~89 verified rows); shrink batch to
            # get ~3 updates/epoch × 1 epoch = ~3 updates total. Combined with
            # lower lr below to keep drift from the resumed RL weights minimal.
            epochs=1,
            batch_size=32,
            sft_seed=42,
            save_checkpoint=True,
            # Matches the RL stage lr (7e-5) so the SFT+RL chain keeps a
            # consistent step size. Previous run at SFT default 1e-4 + 6
            # updates caused catastrophic forgetting (gh_archive_eval
            # success 4.5%, was ~30-40% pre-KSFT).
            adam_params=tinker.AdamParams(learning_rate=7e-5),
        ),
        rl=None,
        generation_concurrency=400,
    ),
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    sft_sources=[
        partial(
            load_sft_cache_suite,
            name="restudy_numrs2_qra",
            cache_id="restudy_numrs2_qra",
            verified_only=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# After NUMRS2_RESTUDY_KSFT_RECIPE completes, fill in its tinker:// URI here.
_NUMRS2_RESTUDY_KSFT_CHECKPOINT_BASE = (
    "tinker://38894579-40a1-547b-a466-ea6babae4ef6:train:0"
)

NUMRS2_RESTUDY_KRL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_krl_numrs2_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_NUMRS2_RESTUDY_KSFT_CHECKPOINT_BASE}/weights/init_sft"
            ),
            resume_sampler_path=(
                f"{_NUMRS2_RESTUDY_KSFT_CHECKPOINT_BASE}/sampler_weights/init_sft"
            ),
            lora_rank=32,
        ),
        rollout=_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            # Step-based budget — chosen over num_passes because the
            # mixed pool (53 restudy + 2208 replay) makes "passes over
            # rl_task_count" meaningless under suite_mix_weights.
            # 30 iters × batch_size=16 = 480 group rollouts → at 50/50
            # ≈ 4.5 restudy epochs + ~10% replay epoch (replay is a
            # forgetting brake, not the main training data).
            max_iterations=30,
            adam_params=tinker.AdamParams(learning_rate=7e-5),
            loss_fn="ppo",
            batch_size=16,
            update_epochs=1,
            max_version_lag=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            skip_update=False,
            # 50/50 mix between new restudy tasks and replay (broad-knowledge
            # QRA pool from the original NUMRS2 KRL stage). Replay is the
            # forgetting brake — keeps the model practicing on the wide
            # distribution while the new tasks drive learning.
            suite_mix_weights={
                "restudy_numrs2_qra_rl": 1.0,
                "pipeline_v2_qra_numrs2_replay": 1.0,
            },
        ),
        generation_concurrency=400,
    ),
    seed_sources=[
        # Restudy QRA cache filtered by routing CSV — keep only SFT (c==0)
        # and RL (1<=c<9) bucket tasks; drop Proficient.
        partial(
            load_sft_cache_seed_suite_factory_bucket_filtered,
            name="restudy_numrs2_qra_rl",
            cache_id="restudy_numrs2_qra",
            routing_csv_path="logs/passatk/restudy_self_learnability.csv",
            include_buckets=("SFT", "RL"),
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        # Replay seed pool — broad-knowledge QRAs from the canonical KRL
        # stage. Mixed with the restudy seeds via RLConfig.suite_mix_weights
        # to slow catastrophic forgetting on gh_archive_eval.
        partial(
            load_sft_cache_seed_suite_factory,
            name="pipeline_v2_qra_numrs2_replay",
            cache_id="pipeline_v2_qra_numrs2",
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# After NUMRS2_RESTUDY_KRL_RECIPE completes, fill in its tinker:// URI here.
_NUMRS2_RESTUDY_KRL_CHECKPOINT_BASE = (
    "tinker://51ee406e-f614-5831-b780-d718c7698f29:train:0"
)
_NUMRS2_RESTUDY_KRL_CHECKPOINT_NAME = "rl_0030"

NUMRS2_RESTUDY_TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_numrs2_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="numrs2",
        cache_dir=default_cache_dir("numrs2"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_NUMRS2_RESTUDY_KRL_CHECKPOINT_BASE}/weights/"
                f"{_NUMRS2_RESTUDY_KRL_CHECKPOINT_NAME}"
            ),
            resume_sampler_path=(
                f"{_NUMRS2_RESTUDY_KRL_CHECKPOINT_BASE}/sampler_weights/"
                f"{_NUMRS2_RESTUDY_KRL_CHECKPOINT_NAME}"
            ),
            lora_rank=32,
        ),
        rollout=_NUMRS2_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            # Fixed step budget — matches KRL's 30-iter sizing so the
            # restudy-lineage stages share a consistent total update count.
            max_iterations=30,
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
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_rl",
            task_slice=slice(0, 150),
            for_rl=True,
            for_eval=False,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.numrs2(),
)


# ---------------------------------------------------------------------------
# Hisab RESTUDY lineage (mirror of the numrs2 restudy lineage)
# ---------------------------------------------------------------------------
# Same 3-stage flow as the numrs2 restudy block above; only the library,
# resume base, replay pool, and cache ids differ.
#
# Resume base for stage 1 = the canonical hisab TaskRL ckpt that
# passatk_restudy.py evaluates (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE output,
# ca15e826/rl_0030). Failures were mined from the most recent hisab TaskRL
# run with rollouts persisted (from_qra_v2_20260507_120638) — that run is
# from a different lineage but mining failures from the canonical one is
# infeasible because its rollouts were not persisted to DB.
_HISAB_TASK_RL_CHECKPOINT_BASE = (
    "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0"
)
_HISAB_TASK_RL_CHECKPOINT_NAME = "rl_0030"

HISAB_RESTUDY_KSFT_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_sft_hisab_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_HISAB_TASK_RL_CHECKPOINT_BASE}/weights/"
                f"{_HISAB_TASK_RL_CHECKPOINT_NAME}"
            ),
            resume_sampler_path=(
                f"{_HISAB_TASK_RL_CHECKPOINT_BASE}/sampler_weights/"
                f"{_HISAB_TASK_RL_CHECKPOINT_NAME}"
            ),
            lora_rank=32,
        ),
        rollout=_ROLLOUT_HISAB,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=SFTConfig(
            # Mirrors NUMRS2_RESTUDY_KSFT_RECIPE: small batch + 1 epoch keeps
            # drift from the resumed RL weights minimal. lr=7e-5 matches the
            # RL stage to avoid catastrophic forgetting (default 1e-4 tanked
            # gh_archive_eval on the numrs2 side).
            epochs=1,
            batch_size=32,
            sft_seed=42,
            save_checkpoint=True,
            adam_params=tinker.AdamParams(learning_rate=7e-5),
        ),
        rl=None,
        generation_concurrency=400,
    ),
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    sft_sources=[
        # Primary signal: the gap-filling QRAs we want the model to learn.
        partial(
            load_sft_cache_suite,
            name="restudy_hisab_qra",
            cache_id="restudy_hisab_qra",
            verified_only=True,
        ),
        # On-policy forgetting brake: sample the model's OWN past successful
        # rollouts to anchor the SFT update toward distributions the model
        # already produces. Out-of-band SFT data (Gemini QRAs) shifts the
        # model toward the source's style — even at lr=7e-5 we saw
        # gh_archive_eval collapse to 1.5% without replay. Using on-policy
        # replay sidesteps that drift.
        #
        # Stratified: take all 115 TaskRL task-latest successes (small pool,
        # closely matches gh_archive distribution) + 1341 KRL task-latest
        # successes (large pool, covers knowledge breadth). 115 + 1341 = 1456
        # ≈ 7× the 208 restudy QRAs → ~12.5%/87.5% restudy/replay per batch.
        # Bumped from 3× after observing post-KSFT gh_archive_eval=7% under
        # the 3× mix; heavier anchor pulls the post-KSFT eval closer to
        # the pre-KSFT baseline.
        partial(
            load_rl_rollout_replay_suite,
            name="hisab_taskrl_replay",
            simple_train_ids=[
                "continue_rl_task_hisab_from_qra_v2_20260507_120638",
            ],
            take_n=115,
        ),
        partial(
            load_rl_rollout_replay_suite,
            name="hisab_krl_replay",
            simple_train_ids=[
                "continue_rl_hisab_from_qra_v2_20260506_235537",
            ],
            take_n=1341,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# HISAB_RESTUDY_KSFT_RECIPE v3 output (1:7 on-policy replay mix,
# post-KSFT gh_archive_eval = 8.0%).
_HISAB_RESTUDY_KSFT_CHECKPOINT_BASE = (
    "tinker://7fd46c37-1169-5d9b-a2b2-def75ca3c354:train:0"
)

HISAB_RESTUDY_KRL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_krl_hisab_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_HISAB_RESTUDY_KSFT_CHECKPOINT_BASE}/weights/init_sft"
            ),
            resume_sampler_path=(
                f"{_HISAB_RESTUDY_KSFT_CHECKPOINT_BASE}/sampler_weights/init_sft"
            ),
            lora_rank=32,
        ),
        rollout=_ROLLOUT_HISAB,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            # Step-based budget — see NUMRS2_RESTUDY_KRL_RECIPE for rationale.
            max_iterations=30,
            adam_params=tinker.AdamParams(learning_rate=7e-5),
            loss_fn="ppo",
            batch_size=16,
            update_epochs=1,
            max_version_lag=1,
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            skip_update=False,
            # 50/50 mix between new hisab restudy tasks and the canonical
            # hisab KRL replay pool (pipeline_v2_qra = study2_pipeline V2
            # QRAs). Replay acts as the forgetting brake.
            suite_mix_weights={
                "restudy_hisab_qra_rl": 1.0,
                "pipeline_v2_qra_hisab_replay": 1.0,
            },
        ),
        generation_concurrency=400,
    ),
    seed_sources=[
        partial(
            load_sft_cache_seed_suite_factory_bucket_filtered,
            name="restudy_hisab_qra_rl",
            cache_id="restudy_hisab_qra",
            routing_csv_path="logs/passatk/restudy_hisab_self_learnability.csv",
            include_buckets=("SFT", "RL"),
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_sft_cache_seed_suite_factory,
            name="pipeline_v2_qra_hisab_replay",
            cache_id="pipeline_v2_qra",
            for_rl=True,
            for_eval=False,
            verified_only=True,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# HISAB_RESTUDY_KRL_RECIPE output (30-iter replay-mix RL,
# post-KRL gh_archive_eval = 18.0% from post-KSFT 8.0%).
_HISAB_RESTUDY_KRL_CHECKPOINT_BASE = (
    "tinker://1e72a865-5d46-5ffe-9499-d029e02ff6be:train:0"
)
_HISAB_RESTUDY_KRL_CHECKPOINT_NAME = "rl_0030"

HISAB_RESTUDY_TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab_restudy",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=(
                f"{_HISAB_RESTUDY_KRL_CHECKPOINT_BASE}/weights/"
                f"{_HISAB_RESTUDY_KRL_CHECKPOINT_NAME}"
            ),
            resume_sampler_path=(
                f"{_HISAB_RESTUDY_KRL_CHECKPOINT_BASE}/sampler_weights/"
                f"{_HISAB_RESTUDY_KRL_CHECKPOINT_NAME}"
            ),
            lora_rank=32,
        ),
        rollout=_HISAB_TASK_RL_ROLLOUT,
        eval=_EVAL,
        checkpoint=_CHECKPOINT,
        sft=None,
        rl=RLConfig(
            max_iterations=30,
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
    seed_sources=[
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_rl",
            task_slice=slice(0, 150),
            for_rl=True,
            for_eval=False,
        ),
        partial(
            load_gh_archive_seed_suite,
            name="gh_archive_eval",
            task_slice=slice(150, 200),
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


CONFIG = HISAB_RESTUDY_TASK_RL_RECIPE


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
        # granular_id is the sole gate for the knowledge load: factories
        # that need it (granular SFT loader, knowledge seed suites) assert
        # ctx.knowledge_list is non-empty, so recipes that don't set
        # granular_id can't accidentally use those factories.
        if cfg.granular_id is not None:
            knowledge_list = await load_granular_knowledge(prisma, cfg.granular_id)
            logger.info(
                f"Loaded {len(knowledge_list)} granular knowledge rows from "
                f"'{cfg.granular_id}'."
            )
        else:
            knowledge_list = []
            logger.info(
                "Skipping granular knowledge load (Recipe.granular_id is None)."
            )

        seed_ctx = SeedLoaderContext(
            prisma=prisma,
            library_spec=cfg.library_spec,
            generation_concurrency=pipeline_config.generation_concurrency,
            knowledge_list=knowledge_list,
            # Generator is only used by knowledge-derived seed factories.
            # Built unconditionally — the wrapper is cheap and recipes that
            # don't need it simply never call it.
            generator=GeneratorAgent(model=get_gemini()),
            granular_id=cfg.granular_id,
        )
        seed_suites: list[SeedSuite] = []
        for src in cfg.seed_sources:
            seed_suites.extend(await src(seed_ctx))

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
