"""Standalone evaluation for numrs2 coding-task benchmark.

Reuses the same eval suite and execution+verification pipeline as
`scripts/run_continue_rl.py` (via `gh_archive_eval`,
`InternalizeExecutor`, `Verifier`) but detaches it from the RL loop.

Two orthogonal axes control what is evaluated:

MODEL_BACKEND — which model runs as the solver:
  - `"tinker"`:  a Tinker sampler loaded from a checkpoint path (use this
                 for evaluating the RL/SFT-trained model).
  - `"agents"`:  any AgentsSDKModel (`from oai_utils import AgentsSDKModel`).
                 Concretely this is any `agents.models.interface.Model` —
                 e.g. `LitellmModel` / `get_gemini()` / `get_gemini_lite()`.

EVAL_STRATEGY — how the model is invoked per task:
  - `"single_turn"`:       one-shot prompt → Rust code → exec + verify (same
                           shape as RL eval).
  - `"ss_solve_verify"`:   multi-turn agent loop via `ss_solve_verify` (the
                           same solver used by `scripts/study.py`), with wiki
                           search + rustdoc lookup tools. Success is decided
                           by the session's own reward.

Edit the `CONFIG` instance near the top of this module to pick axes and
point at a checkpoint or model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Awaitable, Iterable, Literal

import tinker
from agents import set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from oai_utils import AgentsSDKModel
from oai_utils.agent import AgentWrapper
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker_cookbook.renderers import Message
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import (
    get_claude_opus_47,
    get_gemini,
    get_gemini_lite,
    get_gemini_pro,
)
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.rollout_engine import build_solver_system_prompt
from adapter_agent.simple_internalizer.types import SeedSuite

# Suppress Agents SDK tracing telemetry — see scripts/run_continue_rl.py for context.
set_tracing_disabled(True)

# --- Dataset splits (reference; pick one for `EvalConfig.task_slice`) ---
STUDY_SLICE = slice(0, 50)
TRAIN_SLICE = slice(50, 150)
EVAL_SLICE = slice(150, 200)


@dataclass(frozen=True)
class TinkerSolverConfig:
    """Tinker sampler loaded from a checkpoint path (RL/SFT-trained model)."""

    model_name: str
    sampler_path: str | None


@dataclass(frozen=True)
class AgentsSolverConfig:
    """AgentsSDKModel-backed solver (e.g. LitellmModel via gemini)."""

    model: Literal["gemini", "gemini_lite"]


SolverConfig = TinkerSolverConfig | AgentsSolverConfig


@dataclass(frozen=True)
class SsSolveVerifyConfig:
    """Settings for `EVAL_STRATEGY == "ss_solve_verify"` (mirrors study.py).

    The Rust libdir is sourced from `EvalConfig.library_spec.libdir` — only
    eval-time knobs (wiki, max_turns, runtime_mode, …) live here.
    """

    wiki_version: str  # ignored when `use_wiki` is False.
    max_turns: int
    qwen_no_think: bool
    runtime_mode: Literal["docker", "cloudrun"]
    concurrency: int  # typically lower than EvalConfig.concurrency — each run spins its own runtime.
    use_wiki: bool  # If False, the solver runs with an empty wiki (no Prisma needed).


@dataclass(frozen=True)
class SolveVerifyConfig:
    """Settings for `EVAL_STRATEGY == "solve_verify"`.

    Drives the same `solve_verify` (source-tree search + cargo run loop) used
    by study2_decompose / study2_pipeline. Two independent toggles control
    the tool surface:
      - `enable_search_tools`: if False, `<grep>/<read>/<ls>` are rejected.
      - `enable_write_and_run`: if False, `<write_and_run>` is rejected — the
        solver gets exactly one shot via `<submit>` with no test-run loop.
        Used to handicap a strong policy (e.g. Gemini) so its eval score
        reflects one-shot reasoning, not iterative debugging.
    """

    max_turns: int
    qwen_no_think: bool
    runtime_pool_max_size: int
    enable_search_tools: bool = True
    enable_write_and_run: bool = True
    runtime_mode: Literal["docker", "cloudrun"] = "cloudrun"


@dataclass(frozen=True)
class EvalConfig:
    task_slice: slice
    rollout: int
    concurrency: int
    runtime_pool_size: int
    verifier_model: Literal["gemini", "gemini_lite", "gemini_pro", "claude_opus_47"]
    strategy: Literal["single_turn", "ss_solve_verify", "solve_verify"]
    solver: SolverConfig
    ss: SsSolveVerifyConfig
    # Settings for `strategy == "solve_verify"`. Defaulted so existing
    # ss_solve_verify / single_turn recipes don't need to set it.
    sv: SolveVerifyConfig = field(
        default_factory=lambda: SolveVerifyConfig(
            max_turns=10,
            qwen_no_think=True,
            runtime_pool_max_size=50,
        )
    )
    # Library identity. Drives benchmark CSV (load_gh_archive_suite),
    # runtime images (docker_runtime / cloudrun_runtime), the rust libdir
    # used by ss_solve_verify, and the system-prompt library name.
    # Default keeps numrs2-era recipes unchanged; hisab recipes override.
    library_spec: LibrarySpec = field(default_factory=LibrarySpec.numrs2)
    # Eval task source.
    #   "gh_archive":  read benchmark_csv[task_slice] (the original behavior).
    #   "decomposed":  read verified rows from `decomposed_cache_id` in Prisma —
    #                  the mid-level sub-tasks produced by study2_decompose.py.
    #                  `task_slice` is then applied to that row list.
    eval_source: Literal["gh_archive", "decomposed"] = "gh_archive"
    decomposed_cache_id: str | None = None


# --- Shared sub-configs (re-used across named variants below) ---
_TINKER_RL0020 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=(
        "tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020"
    ),
)

# Newer training run (2026-04-30 batch).
_TINKER_SIP2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=(
        "tinker://01c17add-b415-5839-9ea7-2fe09d5748c7:train:0/sampler_weights/rl_0010"
    ),
)

# Latest training run (2026-04-30 batch, c263af3f).
_TINKER_SIP3 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://c263af3f-acfd-5d93-a297-2dc732548b74:train:0/sampler_weights/rl_0010"
    ),
)

# Hisab Knowledge_RL output (1-pass run, terminated at iter 60 because
# checkpoint_interval=10 and num_passes=1 → 68 iters didn't land on a save).
# LEGACY: superseded by _TINKER_HISAB_KRL_V2 below for paper results.
_TINKER_HISAB_KRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://a7e97833-7934-558e-842d-a29f8a2bd48f:train:0/sampler_weights/rl_0060"
    ),
)

# Hisab Knowledge SFT V2 — canonical paper checkpoint produced by
# HISAB_SFT_FROM_PIPELINE_V2_RECIPE. Resume base for HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE.
_TINKER_HISAB_KSFT = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://1237cd7d-e163-5ffb-9ef9-82c98c281079:train:0/sampler_weights/init_sft"
    ),
)

# Hisab Knowledge RL V2 (rl_0054) — canonical paper checkpoint produced by
# HISAB_KNOWLEDGE_RL_FROM_QRA_V2_RECIPE. Resume base for the canonical Task-RL
# recipes (incl. _TINKER_HISAB_TRL2).
_TINKER_HISAB_KRL_V2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://45a766f4-ef41-59d5-bfe1-6c543daf02ed:train:0/sampler_weights/rl_0054"
    ),
)

# Hisab TASK_RL final checkpoint (3 passes × 4 iter/pass = 12 iters; saved by
# the new end-of-loop save logic since 12 isn't a multiple of checkpoint_interval).
_TINKER_HISAB_TRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://b48e4aae-6e11-56ce-9078-c0cfd02db410:train:0/sampler_weights/rl_0012"
    ),
)

# Hisab TASK_RL second run — was launched as 20 passes (= 80 iters), stopped
# at iter 40 (10 passes). 2.5× the practice of _TINKER_HISAB_TRL.
_TINKER_HISAB_TRL2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://9d9c8ae1-c805-5189-9ed2-ee3cc1dd6c16:train:0/sampler_weights/rl_0040"
    ),
)

# Hisab study2 pipeline output (rl_0020).
_TINKER_HISAB_STUDY2 = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://888c29db-b11f-5691-b309-e3c0f1b9f9df:train:0/sampler_weights/rl_0020"
    ),
)

# Hisab Task RL on the DECOMPOSED mid-level pool — output of
# HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE (resumed from QRA-RL v2 / rl_0054 and
# RL'd on `pipeline_v3_decomposed_qra_train`, 30 RL iters).
_TINKER_HISAB_TASK_RL_DECOMPOSED = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://ca15e826-2364-563b-916d-d0bb13b825db:train:0/sampler_weights/rl_0030"
    ),
)

# Numrs2 Knowledge SFT final checkpoint — canonical paper checkpoint produced
# by NUMRS2_KNOWLEDGE_SFT_RECIPE. Resume base for NUMRS2_KNOWLEDGE_RL_RECIPE.
_TINKER_NUMRS2_KSFT = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://4c6bb913-cf76-53c6-9d04-c6e1097e0cb0:train:0/sampler_weights/init_sft"
    ),
)

# Numrs2 Knowledge RL final checkpoint — canonical paper checkpoint produced
# by NUMRS2_KNOWLEDGE_RL_RECIPE (rl_0040). Resume base for NUMRS2_TASK_RL_RECIPE.
_TINKER_NUMRS2_KRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://caaa9922-b354-5e58-80f7-54262e4ca496:train:0/sampler_weights/rl_0040"
    ),
)

# Numrs2 Task RL final checkpoint — canonical paper checkpoint produced by
# NUMRS2_TASK_RL_RECIPE (resumed from numrs2 Knowledge RL caaa9922/rl_0040,
# 10 passes × 4 iter = 40 iter, finished 2026-05-10). Supersedes the SIP-era
# _TINKER_SIP2/_TINKER_SIP3 checkpoints (which are now legacy).
_TINKER_NUMRS2_TRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://be9e6178-ae8f-570d-a987-f2dfd357e565:train:0/sampler_weights/rl_0040"
    ),
)

# Numrs2 Restudy Task RL final checkpoint — produced by
# NUMRS2_RESTUDY_TASK_RL_RECIPE (resumed from Restudy KRL rl_0030, 30 iter on
# gh_archive[0:150], finished 2026-05-12). Peak eval 54.5% at step 25.
# Full lineage: Base → KSFT → KRL → TaskRL (be9e6178/rl_0040) → Restudy KSFT
# → Restudy KRL → Restudy TaskRL (this checkpoint).
_TINKER_NUMRS2_RESTUDY_TRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://35ead364-ce45-5f98-9342-bc78aa6bf23f:train:0/sampler_weights/rl_0030"
    ),
)

# Hisab Restudy Task RL final checkpoint — produced by
# HISAB_RESTUDY_TASK_RL_RECIPE (resumed from Hisab Restudy KRL rl_0030, 30 iter
# on gh_archive[0:150], finished 2026-05-12). In-pipeline peak eval 31.0% at
# step 25 (step 30 eval cancelled by pipeline shutdown).
# Full lineage: hisab TaskRL canonical (ca15e826/rl_0030) → Restudy KSFT v3
# (1:7 on-policy replay) → Restudy KRL → Restudy TaskRL (this checkpoint).
_TINKER_HISAB_RESTUDY_TRL = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=(
        "tinker://1dcb5948-682d-58c9-8402-4bf1780013cf:train:0/sampler_weights/rl_0030"
    ),
)

# Qwen3-8B base model — no fine-tuned sampler, used as the baseline.
_TINKER_BASE = TinkerSolverConfig(
    model_name="Qwen/Qwen3-8B",
    sampler_path=None,
)

# Qwen3-32B base model — larger reference policy.
_TINKER_QWEN32B = TinkerSolverConfig(
    model_name="Qwen/Qwen3-32B",
    sampler_path=None,
)

# Gemini via AgentsSDK / LitellmModel — used as a stronger reference solver.
_AGENTS_GEMINI = AgentsSolverConfig(model="gemini")

_SS_NUMRS_NOWIKI = SsSolveVerifyConfig(
    wiki_version="study_20260430_024306",  # 新版Easy だけど多様なタスクリスト solved by gemini
    # wiki_version="study_20260419_041136",  # Easy のやつ
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=False,
)

_SS_NUMRS_WITHWIKI = SsSolveVerifyConfig(
    wiki_version="study_20260430_024306",  # 新版Easy だけど多様なタスクリスト solved by gemini
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=True,
)

_SS_HISAB_NOWIKI = SsSolveVerifyConfig(
    wiki_version="study_20260504_070444",  # hisab study run
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=False,
)

_SS_HISAB_WITHWIKI = SsSolveVerifyConfig(
    wiki_version="study_20260504_070444",
    max_turns=10,
    qwen_no_think=True,
    runtime_mode="docker",
    concurrency=50,
    use_wiki=True,
)


# --- Named eval variants. Pick one and assign to CONFIG below. ---

# Tinker rl_0020 × ss_solve_verify (rustdoc tools, no wiki).
SIP_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_RL0020,
    ss=_SS_NUMRS_NOWIKI,
)

# Tinker rl_0020 × single_turn (no RAG, no tools — one-shot prompt).
SIP_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_RL0020,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-8B base (no checkpoint) × ss_solve_verify — RAG baseline.
BASE_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_NOWIKI,
)

# Qwen3-8B base × single_turn — bare baseline (no checkpoint, no RAG).
BASE_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-8B base × ss_solve_verify with wiki — full RAG baseline.
BASE_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_BASE,
    ss=_SS_NUMRS_WITHWIKI,
)

# Gemini × single_turn — one-shot prompt, no tools.
GEMINI_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Gemini × ss_solve_verify (rustdoc tools, no wiki).
GEMINI_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_NOWIKI,
)

# Gemini × ss_solve_verify with wiki — full RAG with Gemini policy.
GEMINI_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_WITHWIKI,
)

# Tinker SIP2 (rl_0010, newer run) × single_turn.
SIP2_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_SIP2,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Tinker SIP2 × ss_solve_verify (rustdoc tools, no wiki).
SIP2_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_SIP2,
    ss=_SS_NUMRS_NOWIKI,
)

# Tinker SIP3 (rl_0010, latest run c263af3f) × single_turn.
SIP3_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_SIP3,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Qwen3-32B base × ss_solve_verify (rustdoc tools, no wiki) — larger-model reference.
QWEN32B_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_NUMRS_NOWIKI,
)

# Qwen3-32B base × ss_solve_verify with wiki — larger-model + full RAG.
QWEN32B_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_NUMRS_WITHWIKI,
)


# Numrs2 Knowledge SFT (4c6bb913/init_sft) × single_turn — Base→KSFT progress
# point for the paper's lineage table. library_spec defaults to numrs2.
NUMRS2_KSFT_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_NUMRS2_KSFT,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Numrs2 Knowledge RL (caaa9922/rl_0040) × single_turn — KSFT→KRL progress
# point.
NUMRS2_KRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_NUMRS2_KRL,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Numrs2 Task RL canonical (be9e6178/rl_0040) × single_turn — headline eval
# for the paper's NumRS2 row. library_spec defaults to numrs2 so we don't need
# to set it explicitly.
NUMRS2_TRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_NUMRS2_TRL,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)

# Gemini × solve_verify on numrs2 gh_archive eval slice. RAG-only Gemini —
# enable_search_tools=True (grep/read/ls the library source) but
# enable_write_and_run=False (one shot via <submit>, no test-run loop). This
# is the strong-policy reference that pairs with NUMRS2_TRL_SINGLE: the
# Tinker model gets a one-shot prompt, Gemini gets one-shot output but with
# RAG over the library source. Mirror of HISAB_GEMINI_SOLVE_VERIFY_NOWR_DECOMPOSED
# but on numrs2 gh_archive[150:200], not the decomposed cache.
NUMRS2_GEMINI_SOLVE_VERIFY_NOWR = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=30,
    runtime_pool_size=30,
    verifier_model="gemini_lite",
    strategy="solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_NUMRS_NOWIKI,  # unused for solve_verify
    sv=SolveVerifyConfig(
        max_turns=10,
        qwen_no_think=False,  # unused for Gemini solver
        runtime_pool_max_size=30,
        enable_search_tools=True,
        enable_write_and_run=False,
        runtime_mode="cloudrun",
    ),
    # library_spec defaults to numrs2 — no override needed.
)


# Numrs2 Task RL × single_turn × 8 rollouts/task — mirror of HISAB_TRL_SINGLE_R8.
# Use this when separating knowledge-gap (consistent 0/8) from instability
# (variable 1..7/8) is needed.
NUMRS2_TRL_SINGLE_R8 = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=8,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_NUMRS2_TRL,
    ss=_SS_NUMRS_NOWIKI,
)

# Numrs2 Restudy Task RL (35ead364/rl_0030) × single_turn — headline eval for
# the restudy lineage. Same eval slice (gh_archive[150:200]) as NUMRS2_TRL_SINGLE
# so the two are directly comparable.
NUMRS2_RESTUDY_TRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_NUMRS2_RESTUDY_TRL,
    ss=_SS_NUMRS_NOWIKI,  # unused for single_turn
)


# Hisab Restudy Task RL (1dcb5948/rl_0030) × single_turn — headline eval for
# the hisab restudy lineage. Same eval slice (gh_archive[150:200]) as the
# original HISAB_TASK_RL recipes so progression is directly comparable.
# Verifier swapped to Claude Opus 4.7 (anthropic/claude-opus-4-7) to test
# whether a stronger verifier surfaces a different score from gemini_lite.
HISAB_RESTUDY_TRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_pro",
    strategy="single_turn",
    solver=_TINKER_HISAB_RESTUDY_TRL,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)


# --- Hisab named variants. Mirror the numrs2 set; library_spec drives benchmark
# CSV, runtime images, rust libdir, and the system-prompt library name. ---

# Qwen3-32B base × single_turn — bare hisab baseline.
HISAB_BASE_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_QWEN32B,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Qwen3-32B base × ss_solve_verify (rustdoc tools, no wiki) — RAG baseline.
HISAB_BASE_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Qwen3-32B base × ss_solve_verify with hisab wiki — full RAG baseline.
HISAB_BASE_WIKI = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_QWEN32B,
    ss=_SS_HISAB_WITHWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Hisab Knowledge SFT V2 (1237cd7d/init_sft) × single_turn — Base→KSFT progress
# point for the paper's hisab lineage. canonical.
HISAB_KSFT_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_KSFT,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Hisab Knowledge RL V2 (45a766f4/rl_0054) × single_turn — KSFT→KRL progress
# point. canonical paper checkpoint.
HISAB_KRL_V2_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_KRL_V2,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Hisab Knowledge_RL checkpoint × single_turn.
# LEGACY: uses 1-pass run _TINKER_HISAB_KRL. For paper results use HISAB_KRL_V2_SINGLE.
HISAB_KRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_KRL,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Hisab Knowledge_RL × ss_solve_verify (rustdoc tools, no wiki).
HISAB_KRL_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_HISAB_KRL,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Gemini × single_turn against hisab benchmark.
HISAB_GEMINI_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Gemini × ss_solve_verify (rustdoc tools, no wiki) against hisab benchmark.
HISAB_GEMINI_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Hisab TASK_RL checkpoint × single_turn — the headline eval for the trained model.
HISAB_TRL_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_TRL,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)

# Hisab TASK_RL × single_turn × 8 rollouts/task — measures whether failures
# are knowledge-gap (consistent 0/8) or instability (variable 1..7/8).
HISAB_TRL_SINGLE_R8 = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=8,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_TRL,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Hisab TASK_RL2 (10-pass, rl_0040) × single_turn × 8 rollouts/task — direct
# comparison against HISAB_TRL_SINGLE_R8 (3-pass).
HISAB_TRL2_SINGLE_R8 = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=8,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_TRL2,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Hisab TASK_RL × ss_solve_verify (rustdoc tools, no wiki).
HISAB_TRL_RAG = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_TINKER_HISAB_TRL,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
)

# Hisab study2 pipeline checkpoint (rl_0020) × single_turn.
HISAB_STUDY2_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_STUDY2,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)


# Hisab Base (Qwen3-32B, no sampler) × single_turn against the DECOMPOSED
# mid-level eval set. Sanity-check baseline — expected ~0% so that the KSFT/KRL
# gains are clearly attributable to fine-tuning, not Qwen prior knowledge.
HISAB_BASE_SINGLE_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_QWEN32B,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Hisab Knowledge SFT V2 (1237cd7d/init_sft) × single_turn against the DECOMPOSED
# mid-level eval set. Same shape as HISAB_STUDY2_SINGLE_DECOMPOSED, just a
# different solver checkpoint. Base→KSFT progress point on decomposed bench.
HISAB_KSFT_SINGLE_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_KSFT,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Hisab Knowledge RL V2 (45a766f4/rl_0054) × single_turn against the DECOMPOSED
# mid-level eval set. KSFT→KRL progress point on decomposed bench.
HISAB_KRL_V2_SINGLE_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_KRL_V2,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Gemini × solve_verify on hisab raw gh_archive eval slice. RAG-only Gemini —
# enable_search_tools=True (grep/read/ls the library source) but
# enable_write_and_run=False (one shot via <submit>, no test-run loop).
# Mirror of NUMRS2_GEMINI_SOLVE_VERIFY_NOWR but for hisab. Pairs with
# HISAB_TASK_RL_DECOMPOSED_SINGLE for the strong-policy comparison on raw.
HISAB_GEMINI_SOLVE_VERIFY_NOWR = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=30,
    runtime_pool_size=30,
    verifier_model="claude_opus_47",
    strategy="solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,  # unused for solve_verify
    sv=SolveVerifyConfig(
        max_turns=10,
        qwen_no_think=False,  # unused for Gemini solver
        runtime_pool_max_size=30,
        enable_search_tools=True,
        enable_write_and_run=False,
        runtime_mode="cloudrun",
    ),
    library_spec=LibrarySpec.hisab(),
)


# Hisab Task RL (decomposed-trained, ca15e826/rl_0030) × single_turn against
# the RAW (non-decomposed) gh_archive[150:200] eval slice. Transfer check —
# the decomposed-trained TRL is the canonical paper checkpoint, but its
# headline eval is the decomposed bench (HISAB_TASK_RL_DECOMPOSED_SINGLE_DECOMPOSED).
# This variant measures whether that capability transfers to the original
# integrated 4-7 API tasks.
HISAB_TASK_RL_DECOMPOSED_SINGLE = EvalConfig(
    task_slice=EVAL_SLICE,
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="claude_opus_47",
    strategy="single_turn",
    solver=_TINKER_HISAB_TASK_RL_DECOMPOSED,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
)


# Hisab study2 pipeline checkpoint × single_turn against the DECOMPOSED
# mid-level eval set (verified rows of `pipeline_v3_decomposed_qra_eval`,
# produced by scripts/study2_decompose.py from gh_archive[150:200]). Same
# model, same strategy as HISAB_STUDY2_SINGLE — only the eval source changes
# from gh_archive[150:200] to its decomposed sub-task pool. Apples-to-apples
# difficulty comparison between the original held-out set and its mid-level
# fragments.
HISAB_STUDY2_SINGLE_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_STUDY2,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Same eval set + same single_turn strategy as HISAB_STUDY2_SINGLE_DECOMPOSED,
# but the solver is the checkpoint produced by Task-RL on the DECOMPOSED
# mid-level pool (HISAB_TASK_RL_FROM_DECOMPOSED_RECIPE rl_0030). Direct A/B
# against HISAB_STUDY2_SINGLE_DECOMPOSED — only the RL training data differs
# (mid-level decomposed sub-tasks vs. study2 / pipeline_v2 source pool).
HISAB_TASK_RL_DECOMPOSED_SINGLE_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_TASK_RL_DECOMPOSED,
    ss=_SS_HISAB_NOWIKI,  # unused for single_turn
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Gemini × ss_solve_verify against the DECOMPOSED mid-level eval set —
# strong-policy reference for the same eval slice fragmented by
# study2_decompose.py.
HISAB_GEMINI_RAG_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Gemini × solve_verify (source-tree search ON, but write_and_run OFF) on the
# decomposed mid-level eval set. The handicap: Gemini gets to grep/read/ls the
# library source for as many turns as it likes, but it can NOT test-run code
# — its `<submit>` is the one and only attempt. Used to gauge how much of
# Gemini's headline score on this benchmark is driven by iterative debugging
# vs. one-shot reasoning over the API surface.
HISAB_GEMINI_SOLVE_VERIFY_NOWR_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=30,
    runtime_pool_size=30,
    verifier_model="gemini_lite",
    strategy="solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,  # unused for solve_verify
    sv=SolveVerifyConfig(
        max_turns=10,
        qwen_no_think=False,  # unused for Gemini solver
        runtime_pool_max_size=30,
        enable_search_tools=True,
        enable_write_and_run=False,
        runtime_mode="docker",
    ),
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Same shape but write_and_run ON — the apples-to-apples comparison so the
# write_and_run delta is the only thing varying.
HISAB_GEMINI_SOLVE_VERIFY_DECOMPOSED = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=30,
    runtime_pool_size=30,
    verifier_model="gemini_lite",
    strategy="solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,
    sv=SolveVerifyConfig(
        max_turns=10,
        qwen_no_think=False,
        runtime_pool_max_size=30,
        enable_search_tools=True,
        enable_write_and_run=True,
        runtime_mode="docker",
    ),
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_eval",
)


# Gemini × ss_solve_verify against the decomposed mid-level eval set —
# strong-policy reference. Reads the dryrun cache for the smoke run.
HISAB_GEMINI_RAG_DECOMPOSED_DRYRUN = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="ss_solve_verify",
    solver=_AGENTS_GEMINI,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_dryrun",
)


# Same as HISAB_STUDY2_SINGLE_DECOMPOSED but reads the dryrun cache produced
# by the 5-task smoke run of study2_decompose.py. Use this to sanity-check
# the eval wiring before the full 150-task decompose has finished.
HISAB_STUDY2_SINGLE_DECOMPOSED_DRYRUN = EvalConfig(
    task_slice=slice(None),
    rollout=1,
    concurrency=50,
    runtime_pool_size=50,
    verifier_model="gemini_lite",
    strategy="single_turn",
    solver=_TINKER_HISAB_STUDY2,
    ss=_SS_HISAB_NOWIKI,
    library_spec=LibrarySpec.hisab(),
    eval_source="decomposed",
    decomposed_cache_id="pipeline_v3_decomposed_qra_dryrun",
)


CONFIG = HISAB_GEMINI_SOLVE_VERIFY_NOWR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")


async def _load_decomposed_eval_suite(
    *,
    cache_id: str,
    task_slice: slice,
) -> SeedSuite:
    """Read verified rows from a decompose-pipeline SFT cache and return them
    as a `SeedSuite` for evaluation.

    Each row's `question` becomes a `Task.instruction`. `task_slice` is then
    applied to the resulting list (typically a no-op or a small dev slice).
    """
    prisma = Prisma()
    await prisma.connect()
    try:
        rows = await prisma.sftcacheitem.find_many(
            where={"cache_id": cache_id, "verified": True},
            order={"id": "asc"},
        )
    finally:
        await prisma.disconnect()
    tasks = [Task(instruction=r.question) for r in rows if r.question]
    tasks = tasks[task_slice]
    logger.info(
        f"Loaded {len(tasks)} verified decomposed tasks from cache_id='{cache_id}'."
    )
    return SeedSuite(
        name=f"decomposed:{cache_id}",
        tasks=tasks,
        for_rl=False,
        for_eval=True,
    )


@dataclass
class TaskEvalResult:
    suite: str
    instruction: str
    success_count: int
    total_count: int
    response_lengths: list[int] = field(default_factory=list)
    response_token_lengths: list[int] | None = None


def _response_length_from_trials(trials: Iterable[Message]) -> int:
    total = 0
    for msg in trials:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total += len(part.get("text", ""))
                elif part.get("type") == "thinking":
                    total += len(part.get("thinking", ""))
    return total


def _length_stats(values: Iterable[int]) -> tuple[float, float, int] | None:
    vs = list(values)
    if not vs:
        return None
    mean = sum(vs) / len(vs)
    var = statistics.pvariance(vs) if len(vs) > 1 else 0.0
    return mean, var, max(vs)


async def _gather_eval_with_progress(
    coros: list[Awaitable["TaskEvalResult"]],
    *,
    desc: str,
    max_concurrent: int,
) -> list["TaskEvalResult"]:
    """Run eval coroutines with a semaphore + live progress bar.

    The bar shows running ok/total rollouts and ratio so the wait isn't blind.
    """
    sem = asyncio.Semaphore(max_concurrent)
    success = 0
    rollouts = 0
    completed_tasks = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(coros), desc=desc, dynamic_ncols=True) as pbar:

            async def _worker(coro: Awaitable[TaskEvalResult]) -> TaskEvalResult:
                nonlocal success, rollouts, completed_tasks
                async with sem:
                    r = await coro
                success += r.success_count
                rollouts += r.total_count
                completed_tasks += 1
                ratio = (success / rollouts) if rollouts else 0.0
                head = (
                    (r.instruction[:50] + "..")
                    if len(r.instruction) > 50
                    else r.instruction
                ).replace("\n", " ")
                mark = "✓" if r.success_count > 0 else "✗"
                tqdm.write(
                    f"  [{completed_tasks}/{len(coros)}] {mark} "
                    f"({r.success_count}/{r.total_count}) {head}"
                )
                pbar.set_postfix_str(f"ok={success}/{rollouts} ({ratio:.1%})")
                pbar.update(1)
                return r

            return await asyncio.gather(*[_worker(c) for c in coros])


async def _sample_with_tinker(
    model: TinkerModel,
    instruction: str,
    system_prompt: str,
    num_samples: int,
) -> tuple[list[str], list[int]]:
    prompt = model.renderer.build_generation_prompt(
        [
            Message(role="system", content=system_prompt),
            Message(role="user", content=instruction),
        ]
    )
    sample_results = await model.sampling_client.sample_async(
        prompt=prompt,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(),
    )
    answers: list[str] = []
    token_lengths: list[int] = []
    for seq in sample_results.sequences:
        token_lengths.append(len(seq.tokens))
        msg, ok = model.renderer.parse_response(seq.tokens)
        text = ""
        if ok:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text += part.get("text", "")
            elif isinstance(content, str):
                text = content
        answers.append(text)
    return answers, token_lengths


async def _sample_with_agents(
    model: AgentsSDKModel,
    instruction: str,
    system_prompt: str,
    num_samples: int,
) -> list[str]:
    async def _one() -> str:
        agent = AgentWrapper.create(
            name="EvalSolver",
            instructions=system_prompt,
            model=model,
        )
        try:
            result = await agent.run(instruction, max_turns=2)
        except Exception as e:
            logger.warning(f"Agents SDK run failed: {e}")
            return ""
        final = result.final_output()
        return (
            final
            if isinstance(final, str)
            else (str(final) if final is not None else "")
        )

    return await asyncio.gather(*[_one() for _ in range(num_samples)])


async def _evaluate_task_single_turn(
    task: Task,
    suite_name: str,
    *,
    backend: Literal["tinker", "agents"],
    tinker_model: TinkerModel | None,
    agents_model: AgentsSDKModel | None,
    system_prompt: str,
    executor: InternalizeExecutor,
    num_samples: int,
) -> TaskEvalResult:
    instruction = task.instruction
    token_lengths: list[int] | None
    if backend == "tinker":
        assert tinker_model is not None
        answers, token_lengths = await _sample_with_tinker(
            tinker_model, instruction, system_prompt, num_samples
        )
    else:
        assert agents_model is not None
        answers = await _sample_with_agents(
            agents_model, instruction, system_prompt, num_samples
        )
        token_lengths = None

    outcomes = await asyncio.gather(
        *[
            executor.run_execution_and_verification(
                instruction, reasoning="", answer_text=ans
            )
            for ans in answers
        ]
    )
    success_count = sum(1 for o in outcomes if o.success)
    return TaskEvalResult(
        suite=suite_name,
        instruction=instruction,
        success_count=success_count,
        total_count=num_samples,
        response_lengths=[len(a) for a in answers],
        response_token_lengths=token_lengths,
    )


async def _evaluate_task_ss_solve_verify(
    task: Task,
    suite_name: str,
    *,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
    rust_doc_analyzer: AsyncRustDocAnalyzer,
    wiki_manager: WikiManager | _NullWikiManager,
    runtime_settings: RuntimeSettings,
    max_turns: int,
    qwen_no_think: bool,
    num_samples: int,
    library_name: str,
) -> TaskEvalResult:
    is_tinker = isinstance(solver_model, TinkerModel)

    async def _one() -> tuple[bool, int, int]:
        try:
            ret = await ss_solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                rust_doc_analyzer=rust_doc_analyzer,
                task=task,
                max_turns=max_turns,
                runtime_settings=runtime_settings,
                wiki_manager=wiki_manager,
                qwen_no_think=qwen_no_think,
                library_name=library_name,
            )
        except Exception as e:
            logger.warning(f"ss_solve_verify failed: {e}")
            return False, 0, 0
        trials = getattr(ret, "trials", None) or []
        char_len = _response_length_from_trials(trials)
        traj = getattr(ret, "trajectory", None)
        token_len = (
            sum(len(tr.ac.tokens) for tr in traj.transitions) if traj is not None else 0
        )
        return ret.is_successful(), char_len, token_len

    outcomes = await asyncio.gather(*[_one() for _ in range(num_samples)])
    return TaskEvalResult(
        suite=suite_name,
        instruction=task.instruction,
        success_count=sum(1 for ok, _, _ in outcomes if ok),
        total_count=num_samples,
        response_lengths=[cl for _, cl, _ in outcomes],
        response_token_lengths=([tl for _, _, tl in outcomes] if is_tinker else None),
    )


async def _evaluate_task_solve_verify(
    task: Task,
    suite_name: str,
    *,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
    libdir,
    runtime_pool: RuntimePool,
    max_turns: int,
    qwen_no_think: bool,
    enable_search_tools: bool,
    enable_write_and_run: bool,
    num_samples: int,
    library_name: str,
) -> TaskEvalResult:
    """Eval one task via `solve_verify` (the source-tree-search loop, no
    rustdoc / wiki). Mirrors `_evaluate_task_ss_solve_verify`'s output shape.
    """
    is_tinker = isinstance(solver_model, TinkerModel)

    async def _one() -> tuple[bool, int, int]:
        try:
            ret = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=task,
                libdir=libdir,
                library_name=library_name,
                runtime_pool=runtime_pool,
                max_turns=max_turns,
                qwen_no_think=qwen_no_think,
                enable_search_tools=enable_search_tools,
                enable_write_and_run=enable_write_and_run,
            )
        except Exception as e:
            logger.warning(f"solve_verify failed: {e}")
            return False, 0, 0
        trials = getattr(ret, "trials", None) or []
        char_len = _response_length_from_trials(trials)
        traj = getattr(ret, "trajectory", None)
        token_len = (
            sum(len(tr.ac.tokens) for tr in traj.transitions)
            if traj is not None
            else 0
        )
        return ret.is_successful(), char_len, token_len

    outcomes = await asyncio.gather(*[_one() for _ in range(num_samples)])
    return TaskEvalResult(
        suite=suite_name,
        instruction=task.instruction,
        success_count=sum(1 for ok, _, _ in outcomes if ok),
        total_count=num_samples,
        response_lengths=[cl for _, cl, _ in outcomes],
        response_token_lengths=([tl for _, _, tl in outcomes] if is_tinker else None),
    )


async def _run_single_turn(
    flattened: list[tuple[str, Task]],
    *,
    cfg: EvalConfig,
    tinker_model: TinkerModel | None,
    agents_model: AgentsSDKModel | None,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    runtime_settings = cfg.library_spec.cloudrun_runtime()
    verifier = Verifier(model=verifier_model, library_name=cfg.library_spec.name)
    runtime_pool = RuntimePool(runtime_settings, max_size=cfg.runtime_pool_size)
    executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)
    system_prompt = build_solver_system_prompt(cfg.library_spec.name)
    backend: Literal["tinker", "agents"] = (
        "tinker" if isinstance(cfg.solver, TinkerSolverConfig) else "agents"
    )

    try:

        async def _one(suite_name: str, task: Task) -> TaskEvalResult:
            return await _evaluate_task_single_turn(
                task,
                suite_name,
                backend=backend,
                tinker_model=tinker_model,
                agents_model=agents_model,
                system_prompt=system_prompt,
                executor=executor,
                num_samples=cfg.rollout,
            )

        return await _gather_eval_with_progress(
            [_one(sn, t) for sn, t in flattened],
            desc="single_turn eval",
            max_concurrent=cfg.concurrency,
        )
    finally:
        await runtime_pool.close_all()


class _NullWikiManager:
    """Duck-typed WikiManager with no Prisma backend.

    Enables running `ss_solve_verify` without a wiki: `ls()` yields no
    titles, `read()` returns None for every title, and MOC.md comes back
    empty (which makes `build_simplified_solver_msg_env` skip the
    <MapOfContent> prompt block entirely).
    """

    def __init__(self, version: str = "null") -> None:
        self.version = version

    async def ls(self, path: str | None = None) -> list[str]:
        return []

    async def read(self, title: str) -> str | None:
        return None


async def _run_ss_solve_verify(
    flattened: list[tuple[str, Task]],
    *,
    cfg: EvalConfig,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    ss = cfg.ss
    libdir = cfg.library_spec.libdir
    runtime_settings = (
        cfg.library_spec.docker_runtime()
        if ss.runtime_mode == "docker"
        else cfg.library_spec.cloudrun_runtime()
    )
    wiki_label = ss.wiki_version if ss.use_wiki else "<disabled>"
    logger.info(
        f"Building ss_solve_verify resources "
        f"(libdir={libdir}, wiki={wiki_label}, runtime={ss.runtime_mode})..."
    )
    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(libdir)

    prisma: Prisma | None = None
    wiki_manager: WikiManager | _NullWikiManager
    if ss.use_wiki:
        prisma = Prisma()
        await prisma.connect()
        wiki_manager = WikiManager(prisma, version=ss.wiki_version)
    else:
        wiki_manager = _NullWikiManager()

    try:
        async with rust_doc_analyzer:

            async def _one(suite_name: str, task: Task) -> TaskEvalResult:
                return await _evaluate_task_ss_solve_verify(
                    task,
                    suite_name,
                    solver_model=solver_model,
                    verifier_model=verifier_model,
                    rust_doc_analyzer=rust_doc_analyzer,
                    wiki_manager=wiki_manager,
                    runtime_settings=runtime_settings,
                    max_turns=ss.max_turns,
                    qwen_no_think=ss.qwen_no_think,
                    num_samples=cfg.rollout,
                    library_name=cfg.library_spec.name,
                )

            return await _gather_eval_with_progress(
                [_one(sn, t) for sn, t in flattened],
                desc="ss_solve_verify eval",
                max_concurrent=ss.concurrency,
            )
    finally:
        if prisma is not None:
            await prisma.disconnect()


async def _run_solve_verify(
    flattened: list[tuple[str, Task]],
    *,
    cfg: EvalConfig,
    solver_model: TinkerModel | LitellmModel,
    verifier_model: AgentsSDKModel,
) -> list[TaskEvalResult]:
    """Drive `solve_verify` (source-tree search + cargo run loop, no wiki
    nor rustdoc) with `enable_search_tools` / `enable_write_and_run` toggles
    sourced from `cfg.sv`.
    """
    sv = cfg.sv
    libdir = cfg.library_spec.libdir
    runtime_settings = (
        cfg.library_spec.docker_runtime()
        if sv.runtime_mode == "docker"
        else cfg.library_spec.cloudrun_runtime()
    )
    logger.info(
        f"Building solve_verify resources "
        f"(libdir={libdir}, runtime={sv.runtime_mode}, "
        f"search={sv.enable_search_tools}, "
        f"write_and_run={sv.enable_write_and_run}, max_turns={sv.max_turns})..."
    )
    runtime_pool = RuntimePool(runtime_settings, max_size=sv.runtime_pool_max_size)
    try:

        async def _one(suite_name: str, task: Task) -> TaskEvalResult:
            return await _evaluate_task_solve_verify(
                task,
                suite_name,
                solver_model=solver_model,
                verifier_model=verifier_model,
                libdir=libdir,
                runtime_pool=runtime_pool,
                max_turns=sv.max_turns,
                qwen_no_think=sv.qwen_no_think,
                enable_search_tools=sv.enable_search_tools,
                enable_write_and_run=sv.enable_write_and_run,
                num_samples=cfg.rollout,
                library_name=cfg.library_spec.name,
            )

        return await _gather_eval_with_progress(
            [_one(sn, t) for sn, t in flattened],
            desc="solve_verify eval",
            max_concurrent=cfg.concurrency,
        )
    finally:
        await runtime_pool.close_all()


async def main() -> None:
    load_dotenv()
    cfg = CONFIG

    if cfg.eval_source == "gh_archive":
        eval_suite = load_gh_archive_suite(
            name="gh_archive_eval",
            task_slice=cfg.task_slice,
            for_rl=False,
            for_eval=True,
            csv_path=cfg.library_spec.benchmark_csv,
            difficulty=cfg.library_spec.default_difficulty,
        )
    elif cfg.eval_source == "decomposed":
        if cfg.decomposed_cache_id is None:
            raise ValueError(
                "eval_source='decomposed' requires decomposed_cache_id to be set."
            )
        eval_suite = await _load_decomposed_eval_suite(
            cache_id=cfg.decomposed_cache_id,
            task_slice=cfg.task_slice,
        )
    else:
        raise ValueError(f"Unknown eval_source: {cfg.eval_source}")
    suites: list[SeedSuite] = [eval_suite]

    if cfg.verifier_model == "gemini":
        verifier_model = get_gemini()
    elif cfg.verifier_model == "gemini_pro":
        verifier_model = get_gemini_pro()
    elif cfg.verifier_model == "claude_opus_47":
        verifier_model = get_claude_opus_47()
    else:
        verifier_model = get_gemini_lite()

    tinker_model: TinkerModel | None = None
    agents_model: AgentsSDKModel | None = None
    if isinstance(cfg.solver, TinkerSolverConfig):
        logger.info(
            f"Loading Tinker sampler (base={cfg.solver.model_name}) "
            f"from {cfg.solver.sampler_path}..."
        )
        tinker_model, _, _ = setup_tinkermodel(
            model_name=cfg.solver.model_name,
            path=cfg.solver.sampler_path,
        )
        backend: Literal["tinker", "agents"] = "tinker"
    else:
        logger.info(f"Using AgentsSDKModel policy: {cfg.solver.model}")
        agents_model = (
            get_gemini() if cfg.solver.model == "gemini" else get_gemini_lite()
        )
        backend = "agents"

    flattened: list[tuple[str, Task]] = [(s.name, t) for s in suites for t in s.tasks]
    slice_step = f":{cfg.task_slice.step}" if cfg.task_slice.step is not None else ""
    slice_repr = f"[{cfg.task_slice.start}:{cfg.task_slice.stop}{slice_step}]"
    logger.info(
        f"Evaluating {len(flattened)} tasks × {cfg.rollout} rollouts "
        f"(backend={backend}, strategy={cfg.strategy}, slice={slice_repr})..."
    )

    if cfg.strategy == "single_turn":
        results = await _run_single_turn(
            flattened=flattened,
            cfg=cfg,
            tinker_model=tinker_model,
            agents_model=agents_model,
            verifier_model=verifier_model,
        )
    elif cfg.strategy == "ss_solve_verify":
        solver_model: TinkerModel | LitellmModel
        if isinstance(cfg.solver, TinkerSolverConfig):
            assert tinker_model is not None
            solver_model = tinker_model
        else:
            assert agents_model is not None
            if not isinstance(agents_model, LitellmModel):
                raise ValueError(
                    "ss_solve_verify requires the agents solver to be a LitellmModel "
                    f"(got {type(agents_model).__name__}). Use get_gemini()/get_gemini_lite()."
                )
            solver_model = agents_model
        results = await _run_ss_solve_verify(
            flattened=flattened,
            cfg=cfg,
            solver_model=solver_model,
            verifier_model=verifier_model,
        )
    elif cfg.strategy == "solve_verify":
        if isinstance(cfg.solver, TinkerSolverConfig):
            assert tinker_model is not None
            sv_solver_model: TinkerModel | LitellmModel = tinker_model
        else:
            assert agents_model is not None
            if not isinstance(agents_model, LitellmModel):
                raise ValueError(
                    "solve_verify requires the agents solver to be a LitellmModel "
                    f"(got {type(agents_model).__name__}). Use get_gemini()/get_gemini_lite()."
                )
            sv_solver_model = agents_model
        results = await _run_solve_verify(
            flattened=flattened,
            cfg=cfg,
            solver_model=sv_solver_model,
            verifier_model=verifier_model,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    suite_stats: dict[str, dict[str, int]] = {
        s.name: {"success": 0, "rollouts": 0} for s in suites
    }
    suite_lengths: dict[str, list[int]] = {s.name: [] for s in suites}
    suite_token_lengths: dict[str, list[int]] = {s.name: [] for s in suites}
    has_token_lengths = False
    for r in results:
        suite_stats[r.suite]["success"] += r.success_count
        suite_stats[r.suite]["rollouts"] += r.total_count
        suite_lengths[r.suite].extend(r.response_lengths)
        if r.response_token_lengths is not None:
            suite_token_lengths[r.suite].extend(r.response_token_lengths)
            has_token_lengths = True

    print("\n" + "=" * 82)
    print(f"{'Suite':<40} | {'Success':>8} | {'Total':>8} | {'Ratio':>8}")
    print("-" * 82)
    for sn, stats in suite_stats.items():
        ratio = stats["success"] / stats["rollouts"] if stats["rollouts"] else 0.0
        print(
            f"{sn:<40} | {stats['success']:>8} | {stats['rollouts']:>8} | {ratio:>8.2%}"
        )
    print("=" * 82)

    print("\nResponse length (characters) per suite:")
    print(f"{'Suite':<40} | {'N':>6} | {'Mean':>10} | {'Variance':>12} | {'Max':>8}")
    print("-" * 86)
    for sn, lengths in suite_lengths.items():
        stats = _length_stats(lengths)
        if stats is None:
            print(f"{sn:<40} | {0:>6} | {'-':>10} | {'-':>12} | {'-':>8}")
            continue
        mean, var, mx = stats
        print(f"{sn:<40} | {len(lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}")
    all_lengths = [ln for lns in suite_lengths.values() for ln in lns]
    overall = _length_stats(all_lengths)
    if overall is not None:
        mean, var, mx = overall
        print("-" * 86)
        print(
            f"{'ALL':<40} | {len(all_lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}"
        )

    if has_token_lengths:
        print("\nResponse length (tokens) per suite:")
        print(
            f"{'Suite':<40} | {'N':>6} | {'Mean':>10} | {'Variance':>12} | {'Max':>8}"
        )
        print("-" * 86)
        for sn, lengths in suite_token_lengths.items():
            stats = _length_stats(lengths)
            if stats is None:
                print(f"{sn:<40} | {0:>6} | {'-':>10} | {'-':>12} | {'-':>8}")
                continue
            mean, var, mx = stats
            print(
                f"{sn:<40} | {len(lengths):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}"
            )
        all_tokens = [ln for lns in suite_token_lengths.values() for ln in lns]
        overall_tok = _length_stats(all_tokens)
        if overall_tok is not None:
            mean, var, mx = overall_tok
            print("-" * 86)
            print(
                f"{'ALL':<40} | {len(all_tokens):>6} | {mean:>10.1f} | {var:>12.1f} | {mx:>8}"
            )

    print("\nPer-task breakdown:")
    for r in results:
        prefix = "✓" if r.success_count > 0 else "✗"
        head = (r.instruction[:70] + "..") if len(r.instruction) > 70 else r.instruction
        print(f"  {prefix} [{r.success_count}/{r.total_count}] {head}")


if __name__ == "__main__":
    asyncio.run(main())
