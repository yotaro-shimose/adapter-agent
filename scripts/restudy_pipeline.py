"""restudy_pipeline.py — restudy QRA pipeline.

Per-failure analogue of `study2_pipeline.py`'s plan→investigate→augment→
verify→reason→persist chain. Where study2_pipeline starts from gh_archive
tasks and plans library-agnostic API surface to investigate, this pipeline
starts from a `FailedAttempt` and plans the specific primitives the failed
agent misunderstood. Stages 2–5 (investigate, augment, variant-verify,
reason, persist) are reused verbatim from `study2_pipeline` — we only
swap the plan stage's task wrapping.

Wiring:
  Stage 1 (FAILURE_PLAN): for each `FailedAttempt`, call `plan_one_shot`
    from `restudy_planner.py` (one-shot structured-output planner with
    failure post-mortem framing) → list[str] primitives. Each item names
    (a) the failure mode + (b) the behavior the agent should have known.
  Stages 2–5: hand each primitive off to `qra_pipeline.run_investigate`
    → on verified investigation, `qra_pipeline.augment_verify_reason`
    (which itself runs augment → variant-verify → reason → persist).

Outputs land in dedicated `restudy_*` cache ids so the existing
study2 caches stay intact.

Run with:
    uv run scripts/restudy_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass

from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from prisma import Prisma

from adapter_agent.hierarchical.process.plan_with_tools import InvestigationPlan
from adapter_agent.hierarchical.qra_pipeline import (
    Investigation,
    InvestigationDispatch,
    StageContext,
    augment_verify_reason,
    ensure_cache,
    persist_investigation,
    run_investigate,
)
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.util.logger_util import setup_base_loglevel

# Cross-script import: restudy_planner.py lives in `scripts/`
# which is on sys.path when launched via `uv run scripts/foo.py`. Pinning
# the directory here makes the import robust regardless of launch path.
_SCRIPTS_DIR = "/root/workspace/adapter-agent/scripts"
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from restudy_planner import (  # noqa: E402
    DEFAULT_FAILURE,
    FailedAttempt,
    plan_one_shot,
)

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


# === Sub-configs (mirrors study2_pipeline.py's shape) ================


@dataclass(frozen=True)
class FailureWorkloadConfig:
    """How much downstream work to do per failure.

    `items_per_failure_limit` caps how many primitives we keep from each
    planner run (mirrors study2's items_per_task_limit). `variants_per_item`
    is how many paraphrased questions we generate per verified primitive
    (mirrors study2's WorkloadConfig.variants_per_item, surfaced via
    PipelineRecipe.workload downstream)."""

    items_per_failure_limit: int = 3
    variants_per_item: int = 3


@dataclass(frozen=True)
class FailureStageConfig:
    """Concurrency dials + per-call turn budgets. Investigator and variant
    verifier turn limits match `study2_pipeline.StageConfig` defaults so
    behavior is consistent across the two pipelines."""

    planner_concurrency: int = 50
    investigator_concurrency: int = 50
    augmenter_concurrency: int = 50
    variant_verifier_concurrency: int = 100
    reasoner_concurrency: int = 100

    planner_max_turns: int = 15
    investigation_max_turns: int = 10
    variant_verify_max_turns: int = 4


@dataclass(frozen=True)
class FailureCacheConfig:
    """Cache ids for the three persistence targets. Default ids are
    `restudy_<library>_{inv,aug,qra}` so they don't collide with
    existing study2 caches."""

    inv_cache_id: str
    aug_cache_id: str
    qra_cache_id: str
    reset_target_caches: bool = True


@dataclass(frozen=True)
class FailureRecipe:
    """Top-level config bundle for one pipeline run.

    `failures` is the batch — typically one per failed agent attempt the
    user wants to convert into SFT signal."""

    library_spec: LibrarySpec
    failures: list[FailedAttempt]
    workload: FailureWorkloadConfig
    stage: FailureStageConfig
    cache: FailureCacheConfig

    @property
    def runtime_pool_max_size(self) -> int:
        # PEAK concurrent runtime use: investigators run before
        # variant-verifiers (per failure), so size to their sum across all
        # failures running concurrently.
        return (
            self.stage.investigator_concurrency
            + self.stage.variant_verifier_concurrency
        )

    @property
    def litellm_pool_size(self) -> int:
        s = self.stage
        return (
            s.planner_concurrency
            + s.investigator_concurrency
            + s.augmenter_concurrency
            + s.variant_verifier_concurrency
            + s.reasoner_concurrency
        )


# === Library variants =================================================
# Switch `LIBRARY` to flip the whole pipeline between numrs2 and hisab.
# Each table entry pins (a) which library_spec to use, (b) the
# `simple_train_id` of the upstream TaskRL run to mine failures from,
# and (c) the three cache ids (inv/aug/qra) for persistence.

LIBRARY: str = "hisab"  # "numrs2" | "hisab"

_LIBRARY_VARIANTS: dict[str, dict] = {
    "numrs2": {
        "library_spec": LibrarySpec.numrs2(),
        "rl_failures_train_id": "continue_rl_task_numrs2_20260510_010510",
        "inv_cache_id": "restudy_numrs2_inv",
        "aug_cache_id": "restudy_numrs2_aug",
        "qra_cache_id": "restudy_numrs2_qra",
    },
    "hisab": {
        "library_spec": LibrarySpec.hisab(),
        # Latest hisab TaskRL run with rollouts persisted (from_qra_v2 lineage;
        # trained on gh_archive[0:150]). Note: the canonical paper hisab TaskRL
        # ckpt (ca15e826/rl_0030, from-decomposed) has no rollouts in DB, so
        # we mine failures from this run instead.
        "rl_failures_train_id": "continue_rl_task_hisab_from_qra_v2_20260507_120638",
        "inv_cache_id": "restudy_hisab_inv",
        "aug_cache_id": "restudy_hisab_aug",
        "qra_cache_id": "restudy_hisab_qra",
    },
}

_VARIANT = _LIBRARY_VARIANTS[LIBRARY]


# === Default recipe ==================================================
# Note: `FailureRecipe` already exposes the field paths the shared stage
# runners read via `ctx.recipe.*` (library_spec, stage.{investigation,
# variant_verify}_max_turns, workload.variants_per_item, cache.{inv,aug,
# qra}_cache_id), so it can be plugged into `StageContext.recipe` directly
# without an adapter shim.

DEFAULT_RECIPE = FailureRecipe(
    library_spec=_VARIANT["library_spec"],
    failures=[DEFAULT_FAILURE],
    workload=FailureWorkloadConfig(),
    stage=FailureStageConfig(),
    cache=FailureCacheConfig(
        inv_cache_id=_VARIANT["inv_cache_id"],
        aug_cache_id=_VARIANT["aug_cache_id"],
        qra_cache_id=_VARIANT["qra_cache_id"],
    ),
)

CONFIG: FailureRecipe = DEFAULT_RECIPE


# === Failure loading from RL rollouts ================================
# Pulls every 0%-success task at its latest rl_step from a given
# simple_train_id and turns each into a `FailedAttempt`. Used when we
# want to mine SFT signal out of a completed RL run's failure tail.

RL_FAILURES_TRAIN_ID = _VARIANT["rl_failures_train_id"]
MAX_FAILURES: int | None = None  # set to int for capped run


async def load_zero_success_failures(
    prisma: Prisma, *, simple_train_id: str
) -> list[FailedAttempt]:
    rows = await prisma.simplerlrollout.find_many(
        where={"simple_train_id": simple_train_id},
        order={"rl_step": "desc"},
    )
    # Each task: keep only rollouts at its latest rl_step.
    latest_step_by_task: dict[str, int] = {}
    for r in rows:
        latest_step_by_task.setdefault(r.task_id, r.rl_step)

    # Group at-latest rollouts per task, count successes.
    from collections import defaultdict
    per_task: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.rl_step == latest_step_by_task[r.task_id]:
            per_task[r.task_id].append(r)

    failures: list[FailedAttempt] = []
    for tid, samples in per_task.items():
        if any(s.success for s in samples):
            continue  # not a 0%-success task
        s = samples[0]
        compile_failed = (
            "error[" in s.execution_output
            or "could not compile" in s.execution_output
        )
        failures.append(FailedAttempt(
            question=s.instruction,
            reasoning=s.reasoning,
            answer=s.answer,
            execution_output=s.execution_output,
            verifier_feedback=None if compile_failed else s.verification_output,
        ))
    return failures


# === Failure planning (Stage 1) ======================================


async def _run_failure_plan(
    failure: FailedAttempt,
    failure_id: str,
    ctx: StageContext,
) -> InvestigationPlan | None:
    """Failure-aware planner: produces a list of misused primitives.

    Uses `plan_one_shot` — a single LLM call with structured output.
    The planner names the failure mode and the behavior the agent should
    have known (without resolving exact API names); the investigator
    does the source discovery downstream. Trades planner precision for
    a guaranteed submission per failure (no max_turns starvation)."""
    sem = ctx.sems["plan"]
    progress = ctx.progresses["plan"]
    spec = ctx.recipe.library_spec
    async with sem:
        try:
            plan = await plan_one_shot(
                failure=failure,
                library_name=spec.name,
                library_summary=ctx.library_summary,
                model=ctx.planner_model,
            )
        except Exception as e:
            progress["done"] += 1
            print(
                f"[plan ERR] {progress['done']}/{progress['total']} "
                f"failure={failure_id}: {e}",
                flush=True,
            )
            logger.exception(f"failure planner crashed: failure={failure_id}")
            return None

        progress["done"] += 1
        n = len(plan.items) if plan is not None else 0
        mark = "ok " if plan is not None else "FAIL"
        print(
            f"[plan {mark}] {progress['done']}/{progress['total']} "
            f"failure={failure_id} -> {n} items",
            flush=True,
        )
        return plan


# === Per-failure orchestration =======================================


async def _process_failure(
    failure: FailedAttempt,
    failure_id: str,
    ctx: StageContext,
) -> None:
    """Full chain for one failure: plan → investigate fan-out →
    (augment → variant-verify → reason → persist) per verified investigation.

    Mirrors `study2_pipeline._process_task_from_plan` structurally; the only
    difference is the plan stage uses our failure-aware wrapping rather
    than the gh_archive task instruction."""
    plan = await _run_failure_plan(failure, failure_id, ctx)
    if plan is None or not plan.items:
        return

    items = plan.items[: ctx.recipe.workload.items_per_failure_limit]
    cache_id = ctx.recipe.cache.inv_cache_id

    async def _investigate_then_augment(item: str) -> None:
        dispatch = InvestigationDispatch(
            task_id=f"failure:{failure_id}",
            task_instruction=failure.question,  # original task, for context
            item=item,
            failure_answer=failure.answer,
            failure_execution_output=failure.execution_output,
            failure_verifier_feedback=failure.verifier_feedback,
        )
        inv: Investigation = await run_investigate(dispatch, ctx)
        try:
            await persist_investigation(ctx.prisma, inv, cache_id=cache_id)
        except Exception:
            logger.exception(f"persist investigation failed: failure={failure_id}")

        if not inv.success or not inv.submit_code:
            return
        await augment_verify_reason(inv, ctx)

    await asyncio.gather(*[_investigate_then_augment(it) for it in items])


# === Main ============================================================


async def main() -> None:
    load_dotenv()
    spec = _VARIANT["library_spec"]

    try:
        library_summary = spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    # Pull failures from the RL run before building the recipe so the
    # `failures` list reflects whatever 0%-success tasks the run produced.
    _prisma_for_load = Prisma()
    await _prisma_for_load.connect()
    try:
        loaded_failures = await load_zero_success_failures(
            _prisma_for_load, simple_train_id=RL_FAILURES_TRAIN_ID
        )
    finally:
        await _prisma_for_load.disconnect()
    if not loaded_failures:
        raise SystemExit(
            f"No 0%-success failures found for simple_train_id="
            f"'{RL_FAILURES_TRAIN_ID}'. Did you run the right RL training?"
        )
    # Cap for fast iteration. Set to None to use all loaded failures.
    if MAX_FAILURES is not None:
        loaded_failures = loaded_failures[:MAX_FAILURES]
    print(f"Using {len(loaded_failures)} zero-success failures from "
          f"'{RL_FAILURES_TRAIN_ID}'.", flush=True)

    cfg = FailureRecipe(
        library_spec=spec,
        failures=loaded_failures,
        workload=CONFIG.workload,
        stage=CONFIG.stage,
        cache=CONFIG.cache,
    )

    print("\n" + "=" * 80)
    print(f"FAILURE-RECOVERY QRA PIPELINE — library={spec.name}")
    print(f"  failures           : {len(cfg.failures)}")
    print(
        f"  items/failure cap  : {cfg.workload.items_per_failure_limit}    "
        f"variants/item: {cfg.workload.variants_per_item}"
    )
    print(
        f"  concurrency        : plan={cfg.stage.planner_concurrency} "
        f"inv={cfg.stage.investigator_concurrency} "
        f"aug={cfg.stage.augmenter_concurrency} "
        f"var={cfg.stage.variant_verifier_concurrency} "
        f"qra={cfg.stage.reasoner_concurrency}"
    )
    print(f"  runtime pool size  : {cfg.runtime_pool_max_size}")
    print(
        f"  caches             : inv='{cfg.cache.inv_cache_id}' "
        f"aug='{cfg.cache.aug_cache_id}' qra='{cfg.cache.qra_cache_id}'"
    )
    print("=" * 80)

    async with litellm_concurrent_limit(max_concurrent=cfg.litellm_pool_size):
        prisma = Prisma()
        await prisma.connect()
        runtime_pool: RuntimePool | None = None
        try:
            # Cache (re)init: identical pattern to study2_pipeline.
            await ensure_cache(
                prisma,
                cache_id=cfg.cache.inv_cache_id,
                library_name=spec.name,
                description=(
                    f"restudy: per-primitive investigations from "
                    f"{len(cfg.failures)} failure(s)."
                ),
                reset=cfg.cache.reset_target_caches,
            )
            await ensure_cache(
                prisma,
                cache_id=cfg.cache.aug_cache_id,
                library_name=spec.name,
                description=(
                    f"restudy aug: variants × {cfg.workload.variants_per_item} "
                    f"per primitive, source_inv='{cfg.cache.inv_cache_id}'."
                ),
                reset=cfg.cache.reset_target_caches,
            )
            await ensure_cache(
                prisma,
                cache_id=cfg.cache.qra_cache_id,
                library_name=spec.name,
                description=(
                    "restudy QRA: SFT-ready triples from verified "
                    "primitive variants with reasoning filled in."
                ),
                reset=cfg.cache.reset_target_caches,
            )

            runtime_pool = RuntimePool(
                settings=spec.cloudrun_runtime(),
                max_size=cfg.runtime_pool_max_size,
            )

            n_fail = len(cfg.failures)
            items_cap = cfg.workload.items_per_failure_limit
            variants_per_item = cfg.workload.variants_per_item
            inv_total = n_fail * items_cap
            aug_total = inv_total
            var_total = inv_total * variants_per_item

            ctx = StageContext(
                recipe=cfg,  # duck-typed; FailureRecipe exposes the field paths
                             # the shared stages read (see qra_pipeline.StageContext)
                library_summary=library_summary,
                runtime_pool=runtime_pool,
                prisma=prisma,
                planner_model=get_gemini(),
                solver_model=get_gemini(),
                verifier_model=get_gemini_lite(),
                augmenter_model=get_gemini(),
                reasoner_model=get_gemini(),
                sems={
                    "plan": asyncio.Semaphore(cfg.stage.planner_concurrency),
                    "inv": asyncio.Semaphore(cfg.stage.investigator_concurrency),
                    "aug": asyncio.Semaphore(cfg.stage.augmenter_concurrency),
                    "var": asyncio.Semaphore(cfg.stage.variant_verifier_concurrency),
                    "qra": asyncio.Semaphore(cfg.stage.reasoner_concurrency),
                },
                progresses={
                    "plan": {"done": 0, "total": n_fail},
                    "inv": {"done": 0, "ok": 0, "total": inv_total},
                    "aug": {"done": 0, "total": aug_total},
                    "var": {"done": 0, "ok": 0, "total": var_total},
                    "qra": {"done": 0, "total": var_total},
                },
            )

            await asyncio.gather(
                *[
                    _process_failure(f, f"f{i:03d}", ctx)
                    for i, f in enumerate(cfg.failures)
                ]
            )

            print("\n" + "=" * 80)
            p = ctx.progresses
            print(
                f"DONE — plans: {p['plan']['done']}/{p['plan']['total']} | "
                f"inv: {p['inv']['ok']}/{p['inv']['done']} | "
                f"aug: {p['aug']['done']} | "
                f"var: {p['var']['ok']}/{p['var']['done']} | "
                f"qra: {p['qra']['done']}"
            )
            print(
                f"→ View in graphvis: SFT Caches tab — "
                f"'{cfg.cache.inv_cache_id}', "
                f"'{cfg.cache.aug_cache_id}', "
                f"'{cfg.cache.qra_cache_id}'"
            )

        finally:
            if runtime_pool is not None:
                await runtime_pool.close_all()
            await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
