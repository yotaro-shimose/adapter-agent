"""study2_pipeline.py — stage-aware study2 + study2_augment pipeline.

Two start modes selected by `CONFIG.start_stage`:

  - StartStage.PLAN: full pipeline. For each gh_archive task:
      1. PLAN: source-aware planner enumerates `<InvestigationTarget>` items
         (`plan_with_tools`).
      2. INVESTIGATE: per item, `solve_verify` (search ON) writes + verifies a
         code example. The result is persisted to the inv cache. If the
         verifier accepts, the (instruction, answer) pair feeds the next stage.
      3. AUGMENT: source-aware augmenter (`propose_variants`) proposes N
         variants describing the goal in pure problem-domain language.
      4. VARIANT VERIFY: per variant, `solve_verify` (search OFF) — fed the
         original (instruction, answer) as a `SolvedSubtask` hint — writes
         fresh code. Variants are persisted to the aug cache (verified or not).
      5. REASON: verified variants get chain-of-thought filled in and land
         in the qra cache as SFT-ready triples.

  - StartStage.AUGMENT: skip plan + investigate. Read verified investigations
    from an existing inv cache and run stages 3-5 only — useful when you want
    more QRAs without re-running the expensive verify step.

There is NO barrier between stages. As soon as one task's plan returns,
its items start investigating; as soon as one item verifies, its augmenter
fires; as soon as variants are proposed, they start verifying.

Runtime: cloudrun, one shared RuntimePool sized to the peak concurrent
solve_verify load.

Run with:
    uv run scripts/study2_pipeline.py
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from prisma import Prisma

from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.plan_with_tools import (
    InvestigationPlan,
    plan_with_tools,
)
from adapter_agent.hierarchical.qra_pipeline import (
    Investigation,
    InvestigationDispatch,
    StageContext,
    augment_verify_reason,
    ensure_cache,
    load_investigations_from_cache,
    persist_investigation,
    run_investigate,
)
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.util.logger_util import setup_base_loglevel

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


# === Start-stage selector ===========================================


class StartStage(Enum):
    """Where the pipeline begins.

    PLAN: full pipeline. inv cache is the OUTPUT of investigate stage.
    AUGMENT: skip plan + investigate. inv cache is the INPUT (verified rows
        are reconstituted into Investigation objects and feed the aug stage).
    """

    PLAN = "plan"
    AUGMENT = "augment"


# === Sub-configs ====================================================


@dataclass(frozen=True)
class WorkloadConfig:
    """How much work to do per task.

    `num_tasks` is only consumed in PLAN mode (it slices gh_archive). In
    AUGMENT mode the work volume is set by how many verified rows live in
    the source inv cache.
    """

    num_tasks: slice = slice(0, 30)
    items_per_task_limit: int = 5
    variants_per_item: int = 3


@dataclass(frozen=True)
class StageConfig:
    """Concurrency dials and per-call turn budgets.

    Each stage's semaphore is independent; planner + augmenter + reasoner
    are pure-LLM (no docker/cargo) so they can be pumped higher.
    """

    planner_concurrency: int = 50
    investigator_concurrency: int = 50
    augmenter_concurrency: int = 50
    variant_verifier_concurrency: int = 100
    reasoner_concurrency: int = 100

    planner_max_turns: int = 15
    investigation_max_turns: int = 10
    variant_verify_max_turns: int = 4


@dataclass(frozen=True)
class CacheConfig:
    """Cache ids for each stage's persistence target.

    PLAN mode: all three are outputs; existing rows are dropped and recreated.
    AUGMENT mode: `inv_cache_id` is the read-only input; `aug_cache_id` and
    `qra_cache_id` are outputs and get the drop+create treatment iff
    `reset_target_caches=True`.
    """

    inv_cache_id: str
    aug_cache_id: str
    qra_cache_id: str
    reset_target_caches: bool = True
    verified_only_source: bool = True  # AUGMENT mode: filter inv rows


@dataclass(frozen=True)
class PipelineRecipe:
    start_stage: StartStage
    library_spec: LibrarySpec
    workload: WorkloadConfig
    stage: StageConfig
    cache: CacheConfig

    @property
    def runtime_pool_max_size(self) -> int:
        # PLAN mode runs investigator + variant-verifier concurrently;
        # AUGMENT mode only the latter. Size to the actual peak.
        if self.start_stage == StartStage.PLAN:
            return (
                self.stage.investigator_concurrency
                + self.stage.variant_verifier_concurrency
            )
        return self.stage.variant_verifier_concurrency

    @property
    def litellm_pool_size(self) -> int:
        s = self.stage
        if self.start_stage == StartStage.PLAN:
            return (
                s.planner_concurrency
                + s.investigator_concurrency
                + s.augmenter_concurrency
                + s.variant_verifier_concurrency
                + s.reasoner_concurrency
            )
        return (
            s.augmenter_concurrency
            + s.variant_verifier_concurrency
            + s.reasoner_concurrency
        )


# === Recipes ========================================================

# Original behaviour: full plan → investigate → augment → verify → reason.
FULL_PIPELINE_V1 = PipelineRecipe(
    start_stage=StartStage.PLAN,
    library_spec=LibrarySpec.hisab(),
    workload=WorkloadConfig(),
    stage=StageConfig(),
    cache=CacheConfig(
        inv_cache_id="pipeline_v1_inv",
        aug_cache_id="pipeline_v1_aug",
        qra_cache_id="pipeline_v1_qra",
    ),
)

# Re-run aug → verify → reason against the v1 inv cache to grow QRA volume
# without re-investigating. Outputs land in `_v2`-suffixed caches so the
# original aug/qra caches stay intact and side-by-side.
AUGMENT_FROM_V1 = PipelineRecipe(
    start_stage=StartStage.AUGMENT,
    library_spec=LibrarySpec.hisab(),
    workload=WorkloadConfig(variants_per_item=8),
    stage=StageConfig(),
    cache=CacheConfig(
        inv_cache_id="pipeline_v1_inv",
        aug_cache_id="pipeline_v1_aug_v2",
        qra_cache_id="pipeline_v1_qra_v2",
    ),
)

# Version 2: full plan → investigate → augment → verify → reason from
# scratch, sized 5x larger than v1 — 150 gh_archive tasks instead of 30,
# and 8 augmented variants per investigation instead of 3. Lands in its
# own `pipeline_v2_*` caches so v1's caches remain untouched.
FULL_PIPELINE_V2 = PipelineRecipe(
    start_stage=StartStage.PLAN,
    library_spec=LibrarySpec.hisab(),
    workload=WorkloadConfig(
        num_tasks=slice(0, 150),
        variants_per_item=8,
    ),
    stage=StageConfig(),
    cache=CacheConfig(
        inv_cache_id="pipeline_v2_inv",
        aug_cache_id="pipeline_v2_aug",
        qra_cache_id="pipeline_v2_qra",
    ),
)


# Smoke run on numrs2: same shape as v1 but downsized to 10 tasks × 5 items
# × 3 variants. Dedicated `_numrs2` cache ids so the existing hisab v1/v2
# caches stay intact.
SMOKE_NUMRS2 = PipelineRecipe(
    start_stage=StartStage.PLAN,
    library_spec=LibrarySpec.numrs2(),
    workload=WorkloadConfig(
        num_tasks=slice(0, 10),
        items_per_task_limit=5,
        variants_per_item=3,
    ),
    stage=StageConfig(),
    cache=CacheConfig(
        inv_cache_id="pipeline_smoke_numrs2_inv",
        aug_cache_id="pipeline_smoke_numrs2_aug",
        qra_cache_id="pipeline_smoke_numrs2_qra",
    ),
)

# Numrs2 equivalent of FULL_PIPELINE_V2: 150 gh_archive tasks × 5 items × 8
# variants. Dedicated `_numrs2`-suffixed cache ids so the hisab v1/v2 caches
# stay intact and side-by-side.
FULL_PIPELINE_V2_NUMRS2 = PipelineRecipe(
    start_stage=StartStage.PLAN,
    library_spec=LibrarySpec.numrs2(),
    workload=WorkloadConfig(
        num_tasks=slice(0, 150),
        variants_per_item=8,
    ),
    stage=StageConfig(),
    cache=CacheConfig(
        inv_cache_id="pipeline_v2_inv_numrs2",
        aug_cache_id="pipeline_v2_aug_numrs2",
        qra_cache_id="pipeline_v2_qra_numrs2",
    ),
)


CONFIG: PipelineRecipe = FULL_PIPELINE_V2_NUMRS2


# --- Stage runners (each acquires its own semaphore) ---
# Note: data classes, helpers, persistence, and shared stage runners
# (run_investigate / run_augment / run_variant_verify / run_reason /
# augment_verify_reason) now live in `adapter_agent.hierarchical.qra_pipeline`.
# Only `_run_plan` stays inline here because its task framing is
# study2-specific (gh_archive Task → investigation plan).


async def _run_plan(task: Task, ctx: StageContext) -> InvestigationPlan | None:
    sem = ctx.sems["plan"]
    progress = ctx.progresses["plan"]
    spec = ctx.recipe.library_spec
    async with sem:
        try:
            plan = await plan_with_tools(
                task_instruction=task.instruction,
                library_name=spec.name,
                libdir=spec.libdir,
                library_summary=ctx.library_summary,
                solver_model=ctx.planner_model,
                max_turns=ctx.recipe.stage.planner_max_turns,
            )
            progress["done"] += 1
            n = len(plan.items) if plan is not None else 0
            mark = "ok " if plan is not None else "FAIL"
            print(
                f"[plan {mark}] {progress['done']}/{progress['total']} "
                f"task={task.id} -> {n} items",
                flush=True,
            )
            return plan
        except Exception as e:
            progress["done"] += 1
            print(
                f"[plan ERR] {progress['done']}/{progress['total']} task={task.id}: {e}",
                flush=True,
            )
            logger.exception(f"plan crashed for task={task.id}")
            return None


async def _process_task_from_plan(task: Task, ctx: StageContext) -> None:
    """PLAN mode: plan → investigate fan-out → (aug → var → qra)."""
    plan = await _run_plan(task, ctx)
    if plan is None or not plan.items:
        return

    items = plan.items[: ctx.recipe.workload.items_per_task_limit]
    cache = ctx.recipe.cache

    async def _investigate_then_augment(item: str) -> None:
        dispatch = InvestigationDispatch(
            task_id=task.id, task_instruction=task.instruction, item=item
        )
        inv = await run_investigate(dispatch, ctx)
        try:
            await persist_investigation(ctx.prisma, inv, cache_id=cache.inv_cache_id)
        except Exception:
            logger.exception(f"persist investigation failed: task={task.id}")

        if not inv.success or not inv.submit_code:
            return  # nothing to augment from
        await augment_verify_reason(inv, ctx)

    await asyncio.gather(*[_investigate_then_augment(it) for it in items])


async def _process_investigation(inv: Investigation, ctx: StageContext) -> None:
    """AUGMENT mode: skip plan + investigate. Run aug → var → qra only."""
    await augment_verify_reason(inv, ctx)


# --- Main ---


async def main() -> None:
    load_dotenv()
    cfg = CONFIG
    spec = cfg.library_spec
    workload = cfg.workload
    stage_cfg = cfg.stage
    cache_cfg = cfg.cache

    try:
        library_summary = spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print("\n" + "=" * 80)
    print(f"PIPELINE — start_stage={cfg.start_stage.value}")
    print(
        f"  concurrency: plan={stage_cfg.planner_concurrency} "
        f"inv={stage_cfg.investigator_concurrency} "
        f"aug={stage_cfg.augmenter_concurrency} "
        f"var={stage_cfg.variant_verifier_concurrency} "
        f"qra={stage_cfg.reasoner_concurrency}"
    )
    print(f"  runtime pool: cloudrun, max_size={cfg.runtime_pool_max_size}")
    print(
        f"  caches: inv='{cache_cfg.inv_cache_id}' "
        f"aug='{cache_cfg.aug_cache_id}' qra='{cache_cfg.qra_cache_id}'"
    )
    print("=" * 80)

    # Pin LiteLLM's global httpx client to a pool sized for our peak
    # concurrent LLM load — without this, default httpx limits (~100
    # connections) leak into CLOSE_WAIT under sustained high concurrency
    # and the pipeline progressively stalls.
    async with litellm_concurrent_limit(max_concurrent=cfg.litellm_pool_size):
        prisma = Prisma()
        await prisma.connect()
        runtime_pool: RuntimePool | None = None
        try:
            # PLAN mode owns the inv cache; AUGMENT mode reads it.
            if cfg.start_stage == StartStage.PLAN:
                await ensure_cache(
                    prisma,
                    cache_id=cache_cfg.inv_cache_id,
                    library_name=spec.name,
                    description=(
                        f"study2 (pipelined): plan→investigate output, "
                        f"workload={workload}."
                    ),
                    reset=cache_cfg.reset_target_caches,
                )
            await ensure_cache(
                prisma,
                cache_id=cache_cfg.aug_cache_id,
                library_name=spec.name,
                description=(
                    f"study2_aug (pipelined): variants × {workload.variants_per_item}, "
                    f"start_stage={cfg.start_stage.value}, "
                    f"source_inv='{cache_cfg.inv_cache_id}'."
                ),
                reset=cache_cfg.reset_target_caches,
            )
            await ensure_cache(
                prisma,
                cache_id=cache_cfg.qra_cache_id,
                library_name=spec.name,
                description=(
                    f"study2 QRA (pipelined): SFT-ready triples from verified variants, "
                    f"start_stage={cfg.start_stage.value}."
                ),
                reset=cache_cfg.reset_target_caches,
            )

            runtime_pool = RuntimePool(
                settings=spec.cloudrun_runtime(),
                max_size=cfg.runtime_pool_max_size,
            )

            ctx = StageContext(
                recipe=cfg,
                library_summary=library_summary,
                runtime_pool=runtime_pool,
                prisma=prisma,
                planner_model=get_gemini(),
                solver_model=get_gemini(),
                verifier_model=get_gemini_lite(),
                augmenter_model=get_gemini(),
                reasoner_model=get_gemini(),
                sems={
                    "plan": asyncio.Semaphore(stage_cfg.planner_concurrency),
                    "inv": asyncio.Semaphore(stage_cfg.investigator_concurrency),
                    "aug": asyncio.Semaphore(stage_cfg.augmenter_concurrency),
                    "var": asyncio.Semaphore(stage_cfg.variant_verifier_concurrency),
                    "qra": asyncio.Semaphore(stage_cfg.reasoner_concurrency),
                },
            )

            if cfg.start_stage == StartStage.PLAN:
                tasks = load_gh_archive(
                    difficulty=spec.default_difficulty,
                    csv_path=spec.benchmark_csv,
                )[workload.num_tasks]
                logger.info(f"Loaded {len(tasks)} tasks from {spec.benchmark_csv}.")

                # Upper bounds (actual totals depend on plan/inv survival).
                n_tasks = len(tasks)
                inv_total = n_tasks * workload.items_per_task_limit
                aug_total = inv_total
                var_total = inv_total * workload.variants_per_item
                qra_total = var_total
                ctx.progresses = {
                    "plan": {"done": 0, "total": n_tasks},
                    "inv": {"done": 0, "ok": 0, "total": inv_total},
                    "aug": {"done": 0, "total": aug_total},
                    "var": {"done": 0, "ok": 0, "total": var_total},
                    "qra": {"done": 0, "total": qra_total},
                }

                await asyncio.gather(*[_process_task_from_plan(t, ctx) for t in tasks])
            else:  # AUGMENT
                investigations = await load_investigations_from_cache(
                    prisma,
                    cache_cfg.inv_cache_id,
                    verified_only=cache_cfg.verified_only_source,
                )
                logger.info(
                    f"Loaded {len(investigations)} investigations from "
                    f"cache_id='{cache_cfg.inv_cache_id}' "
                    f"(verified_only={cache_cfg.verified_only_source})."
                )
                if not investigations:
                    print("No investigations to augment. Exiting.")
                    return

                aug_total = len(investigations)
                var_total = aug_total * workload.variants_per_item
                qra_total = var_total
                # plan/inv stages are skipped — keep the keys present (zero
                # totals) so any stray progress writes are harmless.
                ctx.progresses = {
                    "plan": {"done": 0, "total": 0},
                    "inv": {"done": 0, "ok": 0, "total": 0},
                    "aug": {"done": 0, "total": aug_total},
                    "var": {"done": 0, "ok": 0, "total": var_total},
                    "qra": {"done": 0, "total": qra_total},
                }

                await asyncio.gather(
                    *[_process_investigation(inv, ctx) for inv in investigations]
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
                f"→ View in graphvis: SFT Caches tab — '{cache_cfg.inv_cache_id}', "
                f"'{cache_cfg.aug_cache_id}', '{cache_cfg.qra_cache_id}'"
            )

        finally:
            if runtime_pool is not None:
                await runtime_pool.close_all()
            await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
