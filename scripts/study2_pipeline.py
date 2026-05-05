"""study2_pipeline.py — fully pipelined study2 + study2_augment.

For each gh_archive task we run a coroutine that:
  1. PLAN: source-aware planner enumerates `<InvestigationTarget>` items
     (`plan_with_tools`).
  2. INVESTIGATE: per item, `solve_verify` (search ON) writes + verifies a
     code example. The result is persisted to `study2` SFT cache. If the
     verifier accepts, the (instruction, answer) pair feeds the next stage.
  3. AUGMENT: source-aware augmenter (`propose_variants`) proposes N
     variants that exercise the same API but describe the goal in pure
     problem-domain language (no API names leaked).
  4. VARIANT VERIFY: per variant, `solve_verify` (search OFF) — fed the
     original (instruction, answer) as a `SolvedSubtask` hint — writes
     fresh code and only verified variants are persisted to `study2_aug`.

There is NO barrier between stages. As soon as one task's plan returns,
its items start investigating; as soon as one item verifies, its augmenter
fires; as soon as variants are proposed, they start verifying. The four
semaphores below independently bound concurrency for each stage.

Runtime: cloudrun, one shared RuntimePool sized to peak concurrent
solve_verify load (INVESTIGATOR + VARIANT_VERIFIER).

Run with:
    uv run scripts/study2_pipeline.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from agents import set_tracing_disabled
from dotenv import load_dotenv
from prisma import Json, Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.augment import (
    Variants,
    propose_variants,
)
from adapter_agent.hierarchical.process.plan_with_tools import (
    InvestigationPlan,
    plan_with_tools,
)
from adapter_agent.hierarchical.process.reasoner import fill_reasoning
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import RewireSessionResultSuccess
from adapter_agent.rl.solved_subtask import SolvedSubtask
from adapter_agent.util.logger_util import setup_base_loglevel

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


# === Workload ===
NUM_TASKS = slice(0, 30)
ITEMS_PER_TASK_LIMIT = 5
VARIANTS_PER_ITEM = 3

# === Concurrency dials (independent semaphores per stage) ===
PLANNER_CONCURRENCY = 50
INVESTIGATOR_CONCURRENCY = 50
AUGMENTER_CONCURRENCY = 50
VARIANT_VERIFIER_CONCURRENCY = 100
# Reasoning stage: pure LLM, no docker / cargo. Pump it up.
REASONER_CONCURRENCY = 100

# === Per-call turn budgets ===
PLANNER_MAX_TURNS = 16
INVESTIGATION_MAX_TURNS = 12
AUGMENT_MAX_TURNS = 16
VARIANT_VERIFY_MAX_TURNS = 12

# === Cache ids ===
# `EXPERIMENT_ID` namespaces both caches under one experiment so multiple
# pipeline runs (or standalone study2 runs) don't trample each other. Bump
# it when you want a fresh experiment kept side-by-side with previous ones;
# leave it stable to overwrite on each run.
EXPERIMENT_ID = "pipeline_v1"
STUDY2_CACHE_ID = f"{EXPERIMENT_ID}_inv"
AUGMENT_CACHE_ID = f"{EXPERIMENT_ID}_aug"
# QRA cache holds verified variants with reasoning filled in (the SFT-ready
# triples). Only verified variants — original investigations are skipped.
QRA_CACHE_ID = f"{EXPERIMENT_ID}_qra"

# === Runtime ===
# Shared cloudrun pool sized to the peak concurrent solve_verify load.
# (Planner + augmenter don't need runtimes — pure LLM calls.)
RUNTIME_POOL_MAX_SIZE = INVESTIGATOR_CONCURRENCY + VARIANT_VERIFIER_CONCURRENCY


# --- Data classes ---


@dataclass(frozen=True)
class InvestigationDispatch:
    task_id: str
    task_instruction: str
    item: str


@dataclass
class Investigation:
    dispatch: InvestigationDispatch
    success: bool
    conclusion: str
    submit_code: str | None
    verifier_reasoning: str | None
    trials: list | None = None
    reward: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class AugmentDispatch:
    """One verified-investigation × one variant."""

    investigation: Investigation
    variant_instruction: str


@dataclass
class AugmentResult:
    dispatch: AugmentDispatch
    success: bool
    conclusion: str
    submit_code: str | None
    verifier_reasoning: str | None
    trials: list | None = None
    reward: float | None = None
    error: str | None = None


# --- Helpers (mirrors study2.py / study2_augment.py — kept inline so the
#     pipeline script is self-contained and changes here don't ripple back). ---


def _serialize_trials(trials) -> list:
    return [
        PydanticTinkerBaseMessage.model_validate(m).model_dump(
            mode="json", exclude_none=True
        )
        for m in trials
    ]


def _extract_submit(trials) -> str | None:
    for msg in reversed(trials):
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        else:
            continue
        m = re.search(r"<submit>(.*?)</submit>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _build_investigation_task(item: str, library_name: str) -> str:
    return f"""\
Investigate the following aspect of the `{library_name}` library and produce a
runnable Rust code example that demonstrates it.

<InvestigationTarget>
{item}
</InvestigationTarget>

Requirements for your final `<submit>`:
- A complete `fn main()` program (no missing pieces).
- It MUST exercise the actual `{library_name}` API named by the investigation
  target — not a hand-rolled equivalent in plain std.
- It MUST compile and run successfully via `cargo run`.
- It SHOULD print output that makes the demonstrated behavior visible
  (e.g. computed values, intermediate states).
- Keep it minimal — just enough to clearly show how the API is used.
"""


def _build_variant_task(variant: str, library_name: str) -> str:
    return f"""\
{variant}

Requirements for your final `<submit>`:
- A complete `fn main()` program (no missing pieces).
- It MUST exercise the actual `{library_name}` API needed by the task — not
  a hand-rolled equivalent in plain std.
- It MUST compile and run successfully via `cargo run`.
- It SHOULD print output that makes the demonstrated behavior visible.
- Keep it minimal — just enough to clearly satisfy the task.
"""


_INVESTIGATION_TARGET_RE = re.compile(
    r"<InvestigationTarget>\s*([\s\S]*?)\s*</InvestigationTarget>"
)


# --- Persistence ---


async def _persist_investigation(prisma: Prisma, inv: Investigation) -> None:
    d = inv.dispatch
    item_label = d.item if len(d.item) <= 120 else d.item[:117] + "..."
    data: dict = {
        "cache_id": STUDY2_CACHE_ID,
        "knowledge_id": d.task_id,
        "knowledge_title": item_label,
        "question": _build_investigation_task(d.item, "").replace("``", "").strip(),
        "reasoning": "",
        "answer": inv.submit_code or "",
        "verified": inv.success,
        "verifier_reasoning": inv.verifier_reasoning or (inv.error or ""),
        "conclusion": inv.conclusion,
    }
    if inv.trials is not None:
        data["trials_json"] = Json(inv.trials)
    if inv.reward is not None:
        data["reward"] = inv.reward
    await prisma.sftcacheitem.create(data=data)


async def _persist_qra(prisma: Prisma, r: AugmentResult, reasoning: str) -> None:
    """Persist a verified variant with reasoning filled in — the SFT-ready
    QRA triple. Only called for variants that already verified."""
    d = r.dispatch
    title = d.variant_instruction
    title = title if len(title) <= 120 else title[:117] + "..."
    await prisma.sftcacheitem.create(data={
        "cache_id": QRA_CACHE_ID,
        "knowledge_id": f"{d.investigation.dispatch.task_id}#qra",
        "knowledge_title": title,
        "question": d.variant_instruction,
        "reasoning": reasoning,
        "answer": r.submit_code or "",
        # Source variant was already verified — reasoning is descriptive.
        "verified": True,
        "verifier_reasoning": "",
        "conclusion": "reasoning_filled",
    })


async def _persist_variant(prisma: Prisma, r: AugmentResult) -> None:
    d = r.dispatch
    title = d.variant_instruction
    title = title if len(title) <= 120 else title[:117] + "..."
    data: dict = {
        "cache_id": AUGMENT_CACHE_ID,
        # Source-link knowledge_id back to the original task.
        "knowledge_id": f"{d.investigation.dispatch.task_id}#aug",
        "knowledge_title": title,
        "question": d.variant_instruction,
        "reasoning": "",
        "answer": r.submit_code or "",
        "verified": r.success,
        "verifier_reasoning": r.verifier_reasoning or (r.error or ""),
        "conclusion": r.conclusion,
    }
    if r.trials is not None:
        data["trials_json"] = Json(r.trials)
    if r.reward is not None:
        data["reward"] = r.reward
    await prisma.sftcacheitem.create(data=data)


# --- Stage runners (each acquires its own semaphore) ---


async def _run_plan(
    task: Task,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    planner_model,
    sem: asyncio.Semaphore,
    progress: dict,
) -> InvestigationPlan | None:
    async with sem:
        try:
            plan = await plan_with_tools(
                task_instruction=task.instruction,
                library_name=library_spec.name,
                libdir=library_spec.libdir,
                library_summary=library_summary,
                solver_model=planner_model,
                max_turns=PLANNER_MAX_TURNS,
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


async def _run_investigate(
    dispatch: InvestigationDispatch,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    solver_model,
    verifier_model,
    runtime_pool: RuntimePool,
    sem: asyncio.Semaphore,
    progress: dict,
) -> Investigation:
    async with sem:
        try:
            result = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=Task(instruction=_build_investigation_task(
                    dispatch.item, library_spec.name,
                )),
                libdir=library_spec.libdir,
                library_name=library_spec.name,
                runtime_pool=runtime_pool,
                max_turns=INVESTIGATION_MAX_TURNS,
                reference_knowledge=library_summary,
                enable_search_tools=True,
            )
        except Exception as e:
            progress["done"] += 1
            print(
                f"[inv ERR] {progress['done']}/{progress['total']} "
                f"task={dispatch.task_id}: {e}",
                flush=True,
            )
            logger.exception(f"investigator crashed: task={dispatch.task_id}")
            return Investigation(
                dispatch=dispatch,
                success=False,
                conclusion="exception",
                submit_code=None,
                verifier_reasoning=None,
                error=str(e),
            )

        success = isinstance(result, RewireSessionResultSuccess)
        submit = None
        trials_serialized: list | None = None
        if hasattr(result, "trials") and result.trials is not None:
            submit = _extract_submit(result.trials)
            try:
                trials_serialized = _serialize_trials(result.trials)
            except Exception:
                logger.exception("failed to serialize investigation trials")
        progress["done"] += 1
        if success:
            progress["ok"] += 1
        mark = "OK  " if success else "FAIL"
        conclusion = getattr(result, "conclusion", "unknown")
        print(
            f"[inv {mark}] {progress['done']}/{progress['total']} "
            f"(ok={progress['ok']}) task={dispatch.task_id} conclusion={conclusion}",
            flush=True,
        )
        return Investigation(
            dispatch=dispatch,
            success=success,
            conclusion=conclusion,
            submit_code=submit,
            verifier_reasoning=getattr(result, "reasoning", None),
            trials=trials_serialized,
            reward=getattr(result, "reward", None),
        )


async def _run_augment(
    inv: Investigation,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    augmenter_model,
    sem: asyncio.Semaphore,
    progress: dict,
) -> Variants | None:
    async with sem:
        try:
            v = await propose_variants(
                original_instruction=inv.dispatch.item,
                original_answer=inv.submit_code or "",
                library_name=library_spec.name,
                libdir=library_spec.libdir,
                library_summary=library_summary,
                n_variants=VARIANTS_PER_ITEM,
                solver_model=augmenter_model,
                max_turns=AUGMENT_MAX_TURNS,
            )
            progress["done"] += 1
            n = len(v.variants) if v is not None else 0
            mark = "ok " if v is not None else "FAIL"
            print(
                f"[aug {mark}] {progress['done']}/{progress['total']} "
                f"task={inv.dispatch.task_id} -> {n} variants",
                flush=True,
            )
            return v
        except Exception as e:
            progress["done"] += 1
            print(
                f"[aug ERR] {progress['done']}/{progress['total']} "
                f"task={inv.dispatch.task_id}: {e}",
                flush=True,
            )
            logger.exception(f"augmenter crashed: task={inv.dispatch.task_id}")
            return None


async def _run_variant_verify(
    dispatch: AugmentDispatch,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    solver_model,
    verifier_model,
    runtime_pool: RuntimePool,
    sem: asyncio.Semaphore,
    progress: dict,
) -> AugmentResult:
    async with sem:
        # Hand the verified original (instruction, answer) to the solver as
        # a SolvedSubtask hint so it knows which API to use without searching.
        solved = [SolvedSubtask(
            instruction=dispatch.investigation.dispatch.item,
            submit_code=dispatch.investigation.submit_code or "",
        )]
        try:
            result = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=Task(instruction=_build_variant_task(
                    dispatch.variant_instruction, library_spec.name,
                )),
                libdir=library_spec.libdir,
                library_name=library_spec.name,
                runtime_pool=runtime_pool,
                max_turns=VARIANT_VERIFY_MAX_TURNS,
                solved_subtasks=solved,
                reference_knowledge=library_summary,
                # SolvedSubtask hint already carries the API; allowing source
                # search here would let the solver drift to unrelated APIs.
                enable_search_tools=False,
            )
        except Exception as e:
            progress["done"] += 1
            print(
                f"[var ERR] {progress['done']}/{progress['total']} "
                f"task={dispatch.investigation.dispatch.task_id}: {e}",
                flush=True,
            )
            logger.exception(
                f"variant verifier crashed: task={dispatch.investigation.dispatch.task_id}"
            )
            return AugmentResult(
                dispatch=dispatch,
                success=False,
                conclusion="exception",
                submit_code=None,
                verifier_reasoning=None,
                error=str(e),
            )

        success = isinstance(result, RewireSessionResultSuccess)
        submit = None
        trials_serialized: list | None = None
        if hasattr(result, "trials") and result.trials is not None:
            submit = _extract_submit(result.trials)
            try:
                trials_serialized = _serialize_trials(result.trials)
            except Exception:
                logger.exception("failed to serialize variant trials")
        progress["done"] += 1
        if success:
            progress["ok"] += 1
        mark = "OK  " if success else "FAIL"
        conclusion = getattr(result, "conclusion", "unknown")
        print(
            f"[var {mark}] {progress['done']}/{progress['total']} "
            f"(ok={progress['ok']}) task={dispatch.investigation.dispatch.task_id} "
            f"conclusion={conclusion}",
            flush=True,
        )
        return AugmentResult(
            dispatch=dispatch,
            success=success,
            conclusion=conclusion,
            submit_code=submit,
            verifier_reasoning=getattr(result, "reasoning", None),
            trials=trials_serialized,
            reward=getattr(result, "reward", None),
        )


async def _run_reason(
    r: AugmentResult,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    reasoner_model,
    sem: asyncio.Semaphore,
    progress: dict,
) -> str | None:
    """Generate the chain-of-thought between the variant question and its
    verified answer. Caller persists the result if non-None."""
    async with sem:
        try:
            reasoning = await fill_reasoning(
                question=r.dispatch.variant_instruction,
                answer=r.submit_code or "",
                library_name=library_spec.name,
                library_summary=library_summary,
                model=reasoner_model,
            )
            progress["done"] += 1
            mark = "ok " if reasoning else "FAIL"
            words = len(reasoning.split()) if reasoning else 0
            print(
                f"[qra {mark}] {progress['done']}/{progress['total']} "
                f"task={r.dispatch.investigation.dispatch.task_id} words={words}",
                flush=True,
            )
            return reasoning
        except Exception as e:
            progress["done"] += 1
            print(
                f"[qra ERR] {progress['done']}/{progress['total']} "
                f"task={r.dispatch.investigation.dispatch.task_id}: {e}",
                flush=True,
            )
            logger.exception(
                f"reasoner crashed: task={r.dispatch.investigation.dispatch.task_id}"
            )
            return None


# --- Per-task pipeline ---


async def _process_task(
    task: Task,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    planner_model,
    solver_model,
    verifier_model,
    augmenter_model,
    reasoner_model,
    runtime_pool: RuntimePool,
    sems: dict,
    progresses: dict,
    prisma: Prisma,
) -> None:
    """Run plan → investigate fan-out → augment fan-out → variant-verify."""
    plan = await _run_plan(
        task,
        library_spec=library_spec,
        library_summary=library_summary,
        planner_model=planner_model,
        sem=sems["plan"],
        progress=progresses["plan"],
    )
    if plan is None or not plan.items:
        return

    items = plan.items[:ITEMS_PER_TASK_LIMIT]

    async def _investigate_then_augment(item: str) -> None:
        dispatch = InvestigationDispatch(
            task_id=task.id, task_instruction=task.instruction, item=item,
        )
        inv = await _run_investigate(
            dispatch,
            library_spec=library_spec,
            library_summary=library_summary,
            solver_model=solver_model,
            verifier_model=verifier_model,
            runtime_pool=runtime_pool,
            sem=sems["inv"],
            progress=progresses["inv"],
        )
        try:
            await _persist_investigation(prisma, inv)
        except Exception:
            logger.exception(f"persist investigation failed: task={task.id}")

        if not inv.success or not inv.submit_code:
            return  # nothing to augment from

        variants = await _run_augment(
            inv,
            library_spec=library_spec,
            library_summary=library_summary,
            augmenter_model=augmenter_model,
            sem=sems["aug"],
            progress=progresses["aug"],
        )
        if variants is None or not variants.variants:
            return

        dispatches = [
            AugmentDispatch(investigation=inv, variant_instruction=v)
            for v in variants.variants[:VARIANTS_PER_ITEM]
        ]

        async def _verify_and_persist(d: AugmentDispatch) -> None:
            r = await _run_variant_verify(
                d,
                library_spec=library_spec,
                library_summary=library_summary,
                solver_model=solver_model,
                verifier_model=verifier_model,
                runtime_pool=runtime_pool,
                sem=sems["var"],
                progress=progresses["var"],
            )
            try:
                await _persist_variant(prisma, r)
            except Exception:
                logger.exception(f"persist variant failed: task={task.id}")

            # Stage 5 — reasoning fill (only on verified variants).
            if not r.success or not r.submit_code:
                return
            reasoning = await _run_reason(
                r,
                library_spec=library_spec,
                library_summary=library_summary,
                reasoner_model=reasoner_model,
                sem=sems["qra"],
                progress=progresses["qra"],
            )
            if reasoning is None:
                return
            try:
                await _persist_qra(prisma, r, reasoning)
            except Exception:
                logger.exception(f"persist qra failed: task={task.id}")

        await asyncio.gather(*[_verify_and_persist(d) for d in dispatches])

    await asyncio.gather(*[_investigate_then_augment(it) for it in items])


# --- Main ---


async def main() -> None:
    load_dotenv()
    library_spec = LibrarySpec.hisab()

    tasks = load_gh_archive(difficulty=None, csv_path=library_spec.benchmark_csv)[
        NUM_TASKS
    ]
    logger.info(f"Loaded {len(tasks)} tasks from {library_spec.benchmark_csv}.")

    try:
        library_summary = library_spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    # Upper bounds for progress counters (actual totals depend on plan/inv survival).
    n_tasks = len(tasks)
    inv_total = n_tasks * ITEMS_PER_TASK_LIMIT
    aug_total = inv_total
    var_total = inv_total * VARIANTS_PER_ITEM

    print("\n" + "=" * 80)
    print(
        f"PIPELINE — {n_tasks} tasks, up to {ITEMS_PER_TASK_LIMIT} items × "
        f"{VARIANTS_PER_ITEM} variants per task."
    )
    print(
        f"  concurrency: plan={PLANNER_CONCURRENCY} inv={INVESTIGATOR_CONCURRENCY} "
        f"aug={AUGMENTER_CONCURRENCY} var={VARIANT_VERIFIER_CONCURRENCY} "
        f"qra={REASONER_CONCURRENCY}"
    )
    print(f"  runtime pool: cloudrun, max_size={RUNTIME_POOL_MAX_SIZE}")
    print("=" * 80)

    prisma = Prisma()
    await prisma.connect()
    runtime_pool: RuntimePool | None = None
    try:
        # Reset both target caches so each run starts fresh.
        for cid, desc in [
            (STUDY2_CACHE_ID,
             f"study2 (pipelined): {n_tasks} tasks × ≤{ITEMS_PER_TASK_LIMIT} items."),
            (AUGMENT_CACHE_ID,
             f"study2_aug (pipelined): variants from study2 × {VARIANTS_PER_ITEM}."),
        ]:
            if await prisma.sftcache.find_unique(where={"id": cid}) is not None:
                await prisma.sftcache.delete(where={"id": cid})
                logger.info(f"Cleared existing '{cid}' SFT cache.")
            await prisma.sftcache.create(data={
                "id": cid,
                "library_name": library_spec.name,
                "description": desc,
            })

        # One shared cloudrun pool covers both investigator + variant verifier.
        runtime_pool = RuntimePool(
            settings=library_spec.cloudrun_runtime(),
            max_size=RUNTIME_POOL_MAX_SIZE,
        )

        planner_model = get_gemini()
        solver_model = get_gemini()
        verifier_model = get_gemini_lite()
        augmenter_model = get_gemini()

        sems = {
            "plan": asyncio.Semaphore(PLANNER_CONCURRENCY),
            "inv": asyncio.Semaphore(INVESTIGATOR_CONCURRENCY),
            "aug": asyncio.Semaphore(AUGMENTER_CONCURRENCY),
            "var": asyncio.Semaphore(VARIANT_VERIFIER_CONCURRENCY),
        }
        progresses = {
            "plan": {"done": 0, "total": n_tasks},
            "inv": {"done": 0, "ok": 0, "total": inv_total},
            "aug": {"done": 0, "total": aug_total},
            "var": {"done": 0, "ok": 0, "total": var_total},
        }

        await asyncio.gather(*[
            _process_task(
                t,
                library_spec=library_spec,
                library_summary=library_summary,
                planner_model=planner_model,
                solver_model=solver_model,
                verifier_model=verifier_model,
                augmenter_model=augmenter_model,
                runtime_pool=runtime_pool,
                sems=sems,
                progresses=progresses,
                prisma=prisma,
            )
            for t in tasks
        ])

        print("\n" + "=" * 80)
        print(
            f"DONE — plans: {progresses['plan']['done']}/{n_tasks} | "
            f"inv: {progresses['inv']['ok']}/{progresses['inv']['done']} | "
            f"aug: {progresses['aug']['done']} | "
            f"var: {progresses['var']['ok']}/{progresses['var']['done']}"
        )
        print(
            f"→ View in graphvis: SFT Caches tab — '{STUDY2_CACHE_ID}' and "
            f"'{AUGMENT_CACHE_ID}'"
        )

    finally:
        if runtime_pool is not None:
            await runtime_pool.close_all()
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
