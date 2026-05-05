"""study2.py — plan + investigate per-task `{library}` API needs.

Two phases:
  1. Planner (Gemini): read each gh_archive problem statement and enumerate
     which `{library}` features/APIs the solver needs to know first.
  2. Investigator (solve_verify with Gemini as solver): for each item,
     dispatch a task "Investigate {item} and return a runnable code example".
     The solver iterates with cargo + grep / read / ls over the library
     source, the verifier accepts only when the submission compiles and
     exercises the real API.

Run with:
    uv run scripts/study2.py
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
from adapter_agent.hierarchical.process.plan_with_tools import (
    InvestigationPlan,
    plan_with_tools,
)
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import RewireSessionResultSuccess
from adapter_agent.util.logger_util import setup_base_loglevel

set_tracing_disabled(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


NUM_TASKS = slice(0, 30)
ITEMS_PER_TASK_LIMIT = 5  # cap per-task investigations so a run stays bounded
PLANNER_CONCURRENCY = 10
PLANNER_MAX_TURNS = 16  # upper bound on grep/read/ls steps before submit
INVESTIGATION_CONCURRENCY = 5
INVESTIGATION_MAX_TURNS = 12
# `SftCache.id` to write study2 results under. Reused across runs — each run
# clears the existing rows so graphvis always shows the latest dispatch.
CACHE_ID = "study2"


@dataclass
class TaskPlan:
    task: Task
    plan: InvestigationPlan | None
    error: str | None = None


async def _plan_one(
    task: Task,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    solver_model,
    planner_max_turns: int,
    sem: asyncio.Semaphore,
) -> TaskPlan:
    """Run the source-aware planner for one task. Wraps `plan_with_tools`
    with shared throttling + uniform error wrapping."""
    async with sem:
        try:
            plan = await plan_with_tools(
                task_instruction=task.instruction,
                library_name=library_spec.name,
                libdir=library_spec.libdir,
                library_summary=library_summary,
                solver_model=solver_model,
                max_turns=planner_max_turns,
            )
            return TaskPlan(task=task, plan=plan)
        except Exception as e:
            logger.exception(f"Planner failed for task {task.id}")
            return TaskPlan(task=task, plan=None, error=str(e))


# --- Investigation phase ---


@dataclass(frozen=True)
class InvestigationDispatch:
    """One unit of work for the investigation phase: which gh_archive task it
    came from, the original problem statement, and the specific lookup target."""

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
    trials: list | None = None  # JSON-friendly list[dict]
    reward: float | None = None
    error: str | None = None


def _serialize_trials(trials) -> list:
    """JSON-friendly view of TinkerMessage list — same path the trajectories
    table uses (`PydanticTinkerBaseMessage.model_dump`)."""
    return [
        PydanticTinkerBaseMessage.model_validate(m).model_dump(
            mode="json", exclude_none=True
        )
        for m in trials
    ]


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


async def _investigate_one(
    dispatch: InvestigationDispatch,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    solver_model,
    verifier_model,
    runtime_pool,
    sem: asyncio.Semaphore,
) -> Investigation:
    async with sem:
        logger.info(f"[{dispatch.task_id}] investigating: {dispatch.item[:120]}")
        try:
            result = await solve_verify(
                solver_model=solver_model,
                verifier_model=verifier_model,
                task=Task(instruction=_build_investigation_task(dispatch.item, library_spec.name)),
                libdir=library_spec.libdir,
                library_name=library_spec.name,
                runtime_pool=runtime_pool,
                max_turns=INVESTIGATION_MAX_TURNS,
                reference_knowledge=library_summary,
            )
        except Exception as e:
            logger.exception(f"[{dispatch.task_id}] investigator crashed: {dispatch.item[:80]}")
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
                logger.exception(f"[{dispatch.task_id}] failed to serialize trials")
        return Investigation(
            dispatch=dispatch,
            success=success,
            conclusion=getattr(result, "conclusion", "unknown"),
            submit_code=submit,
            verifier_reasoning=getattr(result, "reasoning", None),
            trials=trials_serialized,
            reward=getattr(result, "reward", None),
        )


async def _persist_investigation(prisma: Prisma, inv: Investigation) -> None:
    """Persist one investigation as a row in `sft_cache_items`. We reuse the
    existing schema verbatim so the graphvis "SFT Caches" tab can render it
    out of the box. Field mapping:
      - knowledge_id        ← gh_archive task_id
      - knowledge_title     ← truncated investigation item label
      - question            ← the full investigation prompt text
      - answer              ← submitted Rust code (or empty if failed)
      - verified / conclusion / verifier_reasoning / trials_json / reward
                            ← straight pass-through from the ss_solve_verify result.
    """
    d = inv.dispatch
    item_label = d.item if len(d.item) <= 120 else d.item[:117] + "..."
    data: dict = {
        "cache_id": CACHE_ID,
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
    logger.info(
        f"Loaded library summary ({len(library_summary)} chars) from {library_spec.summary_path}."
    )

    prisma = Prisma()
    await prisma.connect()

    # Reset the study2 cache so each run starts fresh. Cascade drops the items.
    if await prisma.sftcache.find_unique(where={"id": CACHE_ID}) is not None:
        await prisma.sftcache.delete(where={"id": CACHE_ID})
        logger.info(f"Cleared existing '{CACHE_ID}' SFT cache.")
    await prisma.sftcache.create(
        data={
            "id": CACHE_ID,
            "library_name": library_spec.name,
            "description": (
                f"study2: per-task investigation plan + ss_solve_verify code examples "
                f"({len(tasks)} task(s), up to {ITEMS_PER_TASK_LIMIT} items each)."
            ),
        }
    )

    planner_solver_model = get_gemini()
    sem = asyncio.Semaphore(PLANNER_CONCURRENCY)
    results = await asyncio.gather(
        *[
            _plan_one(
                t,
                library_spec=library_spec,
                library_summary=library_summary,
                solver_model=planner_solver_model,
                planner_max_turns=PLANNER_MAX_TURNS,
                sem=sem,
            )
            for t in tasks
        ]
    )

    print("\n" + "=" * 80)
    print(f"INVESTIGATION PLANS — library={library_spec.name}, n={len(results)}")
    print("=" * 80)
    dispatches: list[InvestigationDispatch] = []
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] task_id={r.task.id}")
        print(
            f"  Q: {r.task.instruction[:240]}{'...' if len(r.task.instruction) > 240 else ''}"
        )
        if r.error:
            print(f"  !! planner failed: {r.error}")
            continue
        if r.plan is None:
            print("  !! planner returned no plan (max_turns exhausted or context exceeded)")
            continue
        if not r.plan.items:
            print("  (no investigation items returned)")
            continue
        items = r.plan.items[:ITEMS_PER_TASK_LIMIT]
        print(f"  Items ({len(items)} of {len(r.plan.items)} shown):")
        for j, item in enumerate(items, 1):
            print(f"    {j}. {item}")
            dispatches.append(
                InvestigationDispatch(
                    task_id=r.task.id,
                    task_instruction=r.task.instruction,
                    item=item,
                )
            )

    if not dispatches:
        print("\nNo investigation items to dispatch. Exiting.")
        await prisma.disconnect()
        return

    # --- Phase 2: investigate each item via solve_verify ---
    print("\n" + "=" * 80)
    print(
        f"INVESTIGATIONS — dispatching {len(dispatches)} items "
        f"(concurrency={INVESTIGATION_CONCURRENCY})"
    )
    print("=" * 80)

    solver_model = get_gemini()
    verifier_model = get_gemini_lite()
    runtime_pool = RuntimePool(
        settings=library_spec.docker_runtime(),
        max_size=INVESTIGATION_CONCURRENCY,
    )

    inv_sem = asyncio.Semaphore(INVESTIGATION_CONCURRENCY)
    investigations: list[Investigation] = []

    async def _run_and_persist(d: InvestigationDispatch) -> Investigation:
        inv = await _investigate_one(
            d,
            library_spec=library_spec,
            library_summary=library_summary,
            solver_model=solver_model,
            verifier_model=verifier_model,
            runtime_pool=runtime_pool,
            sem=inv_sem,
        )
        try:
            await _persist_investigation(prisma, inv)
        except Exception:
            logger.exception(f"[{d.task_id}] failed to persist investigation row")
        return inv

    try:
        investigations = await asyncio.gather(
            *[_run_and_persist(d) for d in dispatches]
        )
    finally:
        await runtime_pool.close_all()
        await prisma.disconnect()

    print("\n" + "=" * 80)
    print(f"INVESTIGATION RESULTS — {sum(1 for x in investigations if x.success)}/{len(investigations)} verified")
    print("=" * 80)
    for i, inv in enumerate(investigations, 1):
        marker = "✓" if inv.success else "✗"
        print(f"\n[{i}] {marker} ({inv.conclusion}) task={inv.dispatch.task_id}")
        print(f"  Item: {inv.dispatch.item}")
        if inv.error:
            print(f"  !! crashed: {inv.error}")
            continue
        if inv.verifier_reasoning:
            r = inv.verifier_reasoning
            print(f"  Verifier: {r[:300]}{'...' if len(r) > 300 else ''}")
        if inv.submit_code:
            preview = inv.submit_code if len(inv.submit_code) <= 800 else inv.submit_code[:800] + "..."
            print("  Submit:")
            print("    " + preview.replace("\n", "\n    "))
    print(f"\n→ View in graphvis: SFT Caches tab, cache_id='{CACHE_ID}'")


if __name__ == "__main__":
    asyncio.run(main())
