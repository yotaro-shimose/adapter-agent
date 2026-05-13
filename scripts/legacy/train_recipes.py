"""Legacy recipe instances retired from run_continue_rl.py.

Kept here for posterity and re-runnability — none of these are wired into
the active CONFIG selection in run_continue_rl.py. To resurrect one, copy
its definition back to run_continue_rl.py and assign it to CONFIG, or
import it from this module:

    from scripts.legacy.train_recipes import HISAB_TASK_RL_FROM_QRA_V2_RECIPE

Side-effects: importing this module imports `run_continue_rl`, which in
turn calls `set_tracing_disabled(True)` and configures logging at module
load time. Harmless but worth knowing.
"""

from __future__ import annotations

import dataclasses
import sys
from functools import partial
from pathlib import Path

import tinker

from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.simple_internalizer import PipelineConfig
from adapter_agent.simple_internalizer.seed_loaders import (
    load_gh_archive_seed_suite,
    load_knowledge_seed_suites,
    load_sft_cache_seed_suite_factory,
)
from adapter_agent.simple_internalizer.sft_qra_loaders import (
    load_granular_sft_suite,
    load_sft_cache_suite,
    load_study_root_sft_suite,
)
from adapter_agent.simple_internalizer.types import (
    RLConfig,
    SFTConfig,
    default_cache_dir,
)

# Pull RunRecipe + shared sub-configs from the active script. `scripts/` is
# not a package, so we add it to sys.path before the import.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from run_continue_rl import (  # type: ignore[import-not-found]  # noqa: E402
    RunRecipe,
    _CHECKPOINT,
    _EVAL,
    _HISAB_QRA_RL_V2_CHECKPOINT_BASE,
    _HISAB_QRA_RL_V2_CHECKPOINT_NAME,
    _HISAB_TASK_RL_ROLLOUT,
    _ROLLOUT,
    _ROLLOUT_HISAB,
)


# ---------------------------------------------------------------------------
# Granular knowledge IDs (legacy granular SFT/RL recipes)
# ---------------------------------------------------------------------------

GRANULAR_ID = "granular_prep_20260430_055104"

# Hisab study lineage. Output of `prepare_granular_knowledge.py` running
# HISAB_STUDY_PREP against study_20260504_070444 with no path prefix
# (8 api/ + 15 concepts/ + 1 MOC = 24 articles total).
HISAB_GRANULAR_ID = "granular_prep_hisab_20260504_073544"


# ---------------------------------------------------------------------------
# Checkpoint bases (constants used only by the legacy recipes below)
# ---------------------------------------------------------------------------

# SFT-completed checkpoint (output of SFT_RECIPE).
_SFT_CHECKPOINT_BASE_8B = "tinker://b8d2d31a-ed4d-511f-bd4b-956eaccdc204:train:0"
_SFT_CHECKPOINT_BASE_32B = "tinker://c3ce4acc-191b-5e0b-8a98-27995fac5384:train:0"

# Output of TASK_RL_RECIPE (numrs2 task RL).
_TASK_RL_CHECKPOINT_BASE = "tinker://4d530b2a-8dff-5335-a4fc-eb5e78fa797b:train:0"
_TASK_RL_ROLLOUT = dataclasses.replace(_ROLLOUT, num_samples=16)

# Output of HISAB_SFT_RECIPE (see /tmp/hisab-sft.log).
_HISAB_SFT_CHECKPOINT_BASE = "tinker://25175663-6abf-5703-90ad-0a92081da02e:train:0"

# Output of HISAB_SFT_FROM_PIPELINE_RECIPE (v1 QRA SFT).
_HISAB_QRA_SFT_CHECKPOINT_BASE = "tinker://f5e180b4-f01a-5d07-893c-91563cf39b3f:train:0"

# Output of HISAB_KNOWLEDGE_RL_RECIPE rl_0060 (1-pass run terminated at
# iter 60 because num_passes=1 stopped before iter 68 and the old code
# didn't save a final checkpoint).
_HISAB_TASK_RL_CHECKPOINT_BASE = "tinker://a7e97833-7934-558e-842d-a29f8a2bd48f:train:0"
_HISAB_TASK_RL_CHECKPOINT_NAME = "rl_0060"

# Output of HISAB_KNOWLEDGE_RL_FROM_QRA_RECIPE rl_0015 (job 18c3f7dd-...).
_HISAB_QRA_RL_CHECKPOINT_BASE = "tinker://18c3f7dd-57e3-5045-baea-8b7ccd9e4051:train:0"
_HISAB_QRA_RL_CHECKPOINT_NAME = "rl_0015"


# ---------------------------------------------------------------------------
# Numrs2 legacy recipes (granular-knowledge-based SFT / RL / Task RL)
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
    seed_sources=[
        partial(
            load_knowledge_seed_suites,
            k_per_knowledge=32,
            name_prefix="knowledge_rl",
            cache_id="knowledge_rl__numrs2",
            for_rl=True,
            for_eval=False,
        ),
        partial(
            load_knowledge_seed_suites,
            k_per_knowledge=1,
            name_prefix="knowledge_eval",
            cache_id="knowledge_eval__numrs2",
            for_rl=False,
            for_eval=True,
        ),
    ],
)


# Task RL run: resume SFT_RECIPE's checkpoint → skip SFT → RL directly on
# gh_archive[0:150] (the real task) → eval on gh_archive[150:200].
TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task",
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
)
# これによって出来上がったチェックポイント
# "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010"


# ---------------------------------------------------------------------------
# Hisab legacy recipes (granular knowledge + v1 pipeline + pre-decomposed)
# ---------------------------------------------------------------------------

# Hisab equivalent of SFT_RECIPE: fresh LoRA → granular SFT (k_sft=32 over
# hisab granular knowledge from study_20260504_070444) + study trajectory
# QRAs distilled from the same experiment → exit (SFT-only).
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
            # 266 verified QRAs, bs=32 -> ~8 batches/epoch x 4 epochs = ~32 updates.
            # Each example seen 4 times: enough to teach, low enough to limit memorize.
            epochs=4,
            batch_size=32,
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
            name="pipeline_v1_qra",
            cache_id="pipeline_v1_qra",
            verified_only=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


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
    seed_sources=[
        partial(
            load_knowledge_seed_suites,
            k_per_knowledge=32,
            name_prefix="knowledge_rl",
            cache_id="knowledge_rl__hisab",
            for_rl=True,
            for_eval=False,
        ),
        partial(
            load_knowledge_seed_suites,
            k_per_knowledge=1,
            name_prefix="knowledge_eval",
            cache_id="knowledge_eval__hisab",
            for_rl=False,
            for_eval=True,
        ),
    ],
    library_spec=LibrarySpec.hisab(),
)


# RL on the QRA-augmented v1 cache produced by study2_pipeline AUGMENT mode.
# Resumes from the QRA-SFT checkpoint (output of HISAB_SFT_FROM_PIPELINE_RECIPE).
HISAB_KNOWLEDGE_RL_FROM_QRA_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_hisab_from_qra",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_QRA_SFT_CHECKPOINT_BASE}/weights/init_sft",
            resume_sampler_path=f"{_HISAB_QRA_SFT_CHECKPOINT_BASE}/sampler_weights/init_sft",
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
            name="pipeline_v1_qra_v2_rl",
            cache_id="pipeline_v1_qra_v2",
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


# Hisab equivalent of TASK_RL_RECIPE: RL directly on hisab gh_archive[0:150],
# eval on [150:200]. Resumes from HISAB_KNOWLEDGE_RL_RECIPE's rl_0060.
HISAB_TASK_RL_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab",
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


# Hisab Task RL resuming the QRA-RL checkpoint (output of
# HISAB_KNOWLEDGE_RL_FROM_QRA_RECIPE rl_0015 — see job 18c3f7dd-...). Same
# shape as HISAB_TASK_RL_RECIPE but pulls from the QRA-RL'd weights rather
# than the granular-knowledge-RL'd weights.
HISAB_TASK_RL_FROM_QRA_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab_from_qra",
    pipeline_config=PipelineConfig(
        simple_train_id="",
        library_name="hisab",
        cache_dir=default_cache_dir("hisab"),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-32B",
            resume_trainer_path=f"{_HISAB_QRA_RL_CHECKPOINT_BASE}/weights/{_HISAB_QRA_RL_CHECKPOINT_NAME}",
            resume_sampler_path=f"{_HISAB_QRA_RL_CHECKPOINT_BASE}/sampler_weights/{_HISAB_QRA_RL_CHECKPOINT_NAME}",
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


# Task RL resuming the v2 QRA-RL checkpoint (output of
# HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE rl_0040). Same shape as
# HISAB_TASK_RL_FROM_QRA_RECIPE but num_passes lowered from 20 → 10. Lives
# in legacy because HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE supersedes it.
HISAB_TASK_RL_FROM_QRA_V2_RECIPE = RunRecipe(
    simple_train_id_prefix="continue_rl_task_hisab_from_qra_v2",
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
