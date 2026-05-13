"""study2_decompose.py — decompose gh_archive tasks into mid-level sub-tasks.

The training-data gap this fills: `pipeline_v2_qra` teaches single-API drills
("recall the right API for one operation"), gh_archive[0:150] requires composing
4-7 APIs in a domain-specific pipeline. This script produces the MIDDLE — each
gh_archive task is split into 2-5 sub-tasks that compose 2-3 APIs each, in the
same domain language.

Pipeline (mirrors study2_pipeline.py shape, but only 3 stages):

  1. DECOMPOSE: gh_archive[0:N] → propose_decomposition → list[sub_task]
  2. VERIFY:    each sub_task → solve_verify (search ON) → (success, code)
  3. REASON:    each verified sub_task → fill_reasoning → SFT-ready triple

There is NO barrier between stages. As soon as one task is decomposed, its
sub-tasks start verifying; as soon as one verifies, its reasoner fires.

Output cache: `pipeline_v3_decomposed_qra` (verified rows, SFT-ready).
A parallel `pipeline_v3_decomposed_aug` cache holds every attempted sub-task
(verified or not) for inspection.

Run with:
    uv run scripts/study2_decompose.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field

from agents import set_tracing_disabled
from dotenv import load_dotenv
from oai_utils.litellm import litellm_concurrent_limit
from prisma import Json, Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.decompose import (
    Decomposition,
    propose_decomposition,
)
from adapter_agent.hierarchical.process.reasoner import fill_reasoning
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


# === Config =========================================================


@dataclass(frozen=True)
class DecomposeConfig:
    library_spec: LibrarySpec = field(default_factory=LibrarySpec.hisab)

    # Source slice from gh_archive — train slice (the 150 RL tasks).
    num_tasks: slice = slice(0, 150)

    # Concurrency dials. decompose + reason are pure-LLM (no docker) so they
    # can run hot. verify hits the RuntimePool — keep it sized to the pool.
    # Capped to keep total Gemini concurrency (sum of all three) ≤ ~170 so
    # we stay clear of upstream rate limits.
    decomposer_concurrency: int = 30
    verifier_concurrency: int = 80
    reasoner_concurrency: int = 60

    verify_max_turns: int = 15

    # Output caches. Built from gh_archive[0:150] (the RL train slice). The
    # verified rows feed Task-RL via run_continue_rl.py as a curriculum-aligned
    # mid-level seed pool — same shape as `pipeline_v3_decomposed_qra_eval`
    # but on the train side.
    aug_cache_id: str = "pipeline_v3_decomposed_aug_train"  # all attempts
    qra_cache_id: str = "pipeline_v3_decomposed_qra_train"  # verified + reasoning
    reset_target_caches: bool = True

    @property
    def runtime_pool_max_size(self) -> int:
        return self.verifier_concurrency

    @property
    def litellm_pool_size(self) -> int:
        return (
            self.decomposer_concurrency
            + self.verifier_concurrency
            + self.reasoner_concurrency
        )


CONFIG = DecomposeConfig()


# === Data classes ===================================================


@dataclass(frozen=True)
class SubTaskDispatch:
    """One gh_archive task × one proposed sub-task — the unit of verify work."""

    parent_task_id: str
    parent_instruction: str
    sub_task: str


@dataclass
class VerifyResult:
    dispatch: SubTaskDispatch
    success: bool
    conclusion: str
    submit_code: str | None
    verifier_reasoning: str | None
    trials: list | None = None
    reward: float | None = None
    error: str | None = None


@dataclass
class StageContext:
    cfg: DecomposeConfig
    library_summary: str
    runtime_pool: RuntimePool
    prisma: Prisma

    decomposer_model: object = None
    solver_model: object = None
    verifier_model: object = None
    reasoner_model: object = None

    sems: dict = field(default_factory=dict)
    progresses: dict = field(default_factory=dict)


# === Helpers (lifted from study2_pipeline.py) ======================


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
            text = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        else:
            continue
        m = re.search(r"<submit>(.*?)</submit>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _build_verify_task(sub_task: str, library_name: str) -> str:
    return f"""\
{sub_task}

Requirements for your final `<submit>`:
- A complete `fn main()` program (no missing pieces).
- It MUST exercise the actual `{library_name}` API needed by the task — not
  a hand-rolled equivalent in plain std.
- It MUST compile and run successfully via `cargo run`.
- It SHOULD print output that makes the demonstrated behavior visible.
- Keep it minimal — just enough to clearly satisfy the task.
"""


# === Persistence ===================================================


async def _persist_attempt(
    prisma: Prisma, r: VerifyResult, *, cache_id: str
) -> None:
    d = r.dispatch
    title = d.sub_task
    title = title if len(title) <= 120 else title[:117] + "..."
    data: dict = {
        "cache_id": cache_id,
        "knowledge_id": f"{d.parent_task_id}#decomposed",
        "knowledge_title": title,
        "question": d.sub_task,
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


async def _persist_qra(
    prisma: Prisma, r: VerifyResult, reasoning: str, *, cache_id: str
) -> None:
    d = r.dispatch
    title = d.sub_task
    title = title if len(title) <= 120 else title[:117] + "..."
    assert r.submit_code is not None
    await prisma.sftcacheitem.create(
        data={
            "cache_id": cache_id,
            "knowledge_id": f"{d.parent_task_id}#qra",
            "knowledge_title": title,
            "question": d.sub_task,
            "reasoning": reasoning,
            "answer": r.submit_code,
            "verified": True,
            "verifier_reasoning": "",
            "conclusion": "reasoning_filled",
        }
    )


# === Stage runners =================================================


async def _run_decompose(
    task: Task, ctx: StageContext
) -> Decomposition | None:
    sem = ctx.sems["decompose"]
    progress = ctx.progresses["decompose"]
    spec = ctx.cfg.library_spec
    async with sem:
        try:
            d = await propose_decomposition(
                original_instruction=task.instruction,
                library_name=spec.name,
                library_summary=ctx.library_summary,
                solver_model=ctx.decomposer_model,
            )
            progress["done"] += 1
            n = len(d.sub_tasks) if d is not None else 0
            mark = "ok " if d is not None else "FAIL"
            print(
                f"[dec {mark}] {progress['done']}/{progress['total']} "
                f"task={task.id} -> {n} sub-tasks",
                flush=True,
            )
            return d
        except Exception as e:
            progress["done"] += 1
            print(
                f"[dec ERR] {progress['done']}/{progress['total']} "
                f"task={task.id}: {e}",
                flush=True,
            )
            logger.exception(f"decomposer crashed: task={task.id}")
            return None


async def _run_verify(
    dispatch: SubTaskDispatch, ctx: StageContext
) -> VerifyResult:
    sem = ctx.sems["verify"]
    progress = ctx.progresses["verify"]
    spec = ctx.cfg.library_spec
    async with sem:
        try:
            result = await solve_verify(
                solver_model=ctx.solver_model,
                verifier_model=ctx.verifier_model,
                task=Task(
                    instruction=_build_verify_task(dispatch.sub_task, spec.name)
                ),
                libdir=spec.libdir,
                library_name=spec.name,
                runtime_pool=ctx.runtime_pool,
                max_turns=ctx.cfg.verify_max_turns,
                solved_subtasks=None,  # no SolvedSubtask hint — solver searches
                reference_knowledge=ctx.library_summary,
                # Search ON: sub-tasks preserve domain vocabulary, so the solver
                # benefits from grep/read on the source tree, same as the
                # original gh_archive solver does.
                enable_search_tools=True,
            )
        except Exception as e:
            progress["done"] += 1
            print(
                f"[ver ERR] {progress['done']}/{progress['total']} "
                f"task={dispatch.parent_task_id}: {e}",
                flush=True,
            )
            logger.exception(
                f"verifier crashed: task={dispatch.parent_task_id}"
            )
            return VerifyResult(
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
                logger.exception("failed to serialize verify trials")
        progress["done"] += 1
        if success:
            progress["ok"] += 1
        mark = "OK  " if success else "FAIL"
        conclusion = getattr(result, "conclusion", "unknown")
        print(
            f"[ver {mark}] {progress['done']}/{progress['total']} "
            f"(ok={progress['ok']}) task={dispatch.parent_task_id} "
            f"conclusion={conclusion}",
            flush=True,
        )
        return VerifyResult(
            dispatch=dispatch,
            success=success,
            conclusion=conclusion,
            submit_code=submit,
            verifier_reasoning=getattr(result, "reasoning", None),
            trials=trials_serialized,
            reward=getattr(result, "reward", None),
        )


async def _run_reason(r: VerifyResult, ctx: StageContext) -> str | None:
    sem = ctx.sems["reason"]
    progress = ctx.progresses["reason"]
    spec = ctx.cfg.library_spec
    async with sem:
        try:
            assert r.submit_code is not None
            reasoning = await fill_reasoning(
                question=r.dispatch.sub_task,
                answer=r.submit_code,
                library_name=spec.name,
                library_summary=ctx.library_summary,
                model=ctx.reasoner_model,
            )
            progress["done"] += 1
            mark = "ok " if reasoning else "FAIL"
            words = len(reasoning.split()) if reasoning else 0
            print(
                f"[rea {mark}] {progress['done']}/{progress['total']} "
                f"task={r.dispatch.parent_task_id} words={words}",
                flush=True,
            )
            return reasoning
        except Exception as e:
            progress["done"] += 1
            print(
                f"[rea ERR] {progress['done']}/{progress['total']} "
                f"task={r.dispatch.parent_task_id}: {e}",
                flush=True,
            )
            logger.exception(
                f"reasoner crashed: task={r.dispatch.parent_task_id}"
            )
            return None


# === Composite flow ================================================


async def _process_task(task: Task, ctx: StageContext) -> None:
    """One gh_archive task: decompose → verify each sub-task → reason on
    verified ones. Each stage fires as soon as upstream produces work."""
    decomp = await _run_decompose(task, ctx)
    if decomp is None or not decomp.sub_tasks:
        return

    dispatches = [
        SubTaskDispatch(
            parent_task_id=task.id,
            parent_instruction=task.instruction,
            sub_task=s,
        )
        for s in decomp.sub_tasks
    ]
    # Keep verify/reason totals tracking actual work; bump now that decomposer
    # produced a concrete count.
    ctx.progresses["verify"]["total"] += len(dispatches)
    ctx.progresses["reason"]["total"] += len(dispatches)

    async def _verify_then_reason(d: SubTaskDispatch) -> None:
        r = await _run_verify(d, ctx)
        try:
            await _persist_attempt(
                ctx.prisma, r, cache_id=ctx.cfg.aug_cache_id
            )
        except Exception:
            logger.exception(
                f"persist attempt failed: task={d.parent_task_id}"
            )

        if not r.success or not r.submit_code:
            ctx.progresses["reason"]["total"] -= 1  # never reached reason
            return
        reasoning = await _run_reason(r, ctx)
        if reasoning is None:
            return
        try:
            await _persist_qra(
                ctx.prisma, r, reasoning, cache_id=ctx.cfg.qra_cache_id
            )
        except Exception:
            logger.exception(
                f"persist qra failed: task={d.parent_task_id}"
            )

    await asyncio.gather(*[_verify_then_reason(d) for d in dispatches])


# === Cache (re)init ================================================


async def _ensure_cache(
    prisma: Prisma,
    *,
    cache_id: str,
    library_name: str,
    description: str,
    reset: bool,
) -> None:
    existing = await prisma.sftcache.find_unique(where={"id": cache_id})
    if existing is not None and reset:
        await prisma.sftcache.delete(where={"id": cache_id})
        logger.info(f"Cleared existing '{cache_id}' SFT cache.")
        existing = None
    if existing is None:
        await prisma.sftcache.create(
            data={
                "id": cache_id,
                "library_name": library_name,
                "description": description,
            }
        )


# === Main ==========================================================


async def main() -> None:
    load_dotenv()
    cfg = CONFIG
    spec = cfg.library_spec

    try:
        library_summary = spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print("\n" + "=" * 80)
    print("DECOMPOSE PIPELINE — gh_archive → mid-level sub-tasks")
    print(
        f"  concurrency: dec={cfg.decomposer_concurrency} "
        f"ver={cfg.verifier_concurrency} rea={cfg.reasoner_concurrency}"
    )
    print(f"  runtime pool: cloudrun, max_size={cfg.runtime_pool_max_size}")
    print(
        f"  caches: aug='{cfg.aug_cache_id}' qra='{cfg.qra_cache_id}'"
    )
    print("=" * 80)

    async with litellm_concurrent_limit(max_concurrent=cfg.litellm_pool_size):
        prisma = Prisma()
        await prisma.connect()
        runtime_pool: RuntimePool | None = None
        try:
            await _ensure_cache(
                prisma,
                cache_id=cfg.aug_cache_id,
                library_name=spec.name,
                description=(
                    "decomposed gh_archive sub-tasks (every attempt, verified "
                    "or not) — produced by study2_decompose.py."
                ),
                reset=cfg.reset_target_caches,
            )
            await _ensure_cache(
                prisma,
                cache_id=cfg.qra_cache_id,
                library_name=spec.name,
                description=(
                    "decomposed gh_archive sub-tasks: SFT-ready QRA triples "
                    "(verified + reasoning) — produced by study2_decompose.py."
                ),
                reset=cfg.reset_target_caches,
            )

            runtime_pool = RuntimePool(
                settings=spec.cloudrun_runtime(),
                max_size=cfg.runtime_pool_max_size,
            )

            ctx = StageContext(
                cfg=cfg,
                library_summary=library_summary,
                runtime_pool=runtime_pool,
                prisma=prisma,
                decomposer_model=get_gemini(),
                solver_model=get_gemini(),
                verifier_model=get_gemini_lite(),
                reasoner_model=get_gemini(),
                sems={
                    "decompose": asyncio.Semaphore(cfg.decomposer_concurrency),
                    "verify": asyncio.Semaphore(cfg.verifier_concurrency),
                    "reason": asyncio.Semaphore(cfg.reasoner_concurrency),
                },
            )

            tasks = load_gh_archive(
                difficulty=None, csv_path=spec.benchmark_csv
            )[cfg.num_tasks]
            logger.info(
                f"Loaded {len(tasks)} tasks from {spec.benchmark_csv}."
            )

            n_tasks = len(tasks)
            ctx.progresses = {
                "decompose": {"done": 0, "total": n_tasks},
                # verify/reason totals start at 0 and grow as decompose
                # produces sub-tasks (since the count per task is dynamic).
                "verify": {"done": 0, "ok": 0, "total": 0},
                "reason": {"done": 0, "total": 0},
            }

            await asyncio.gather(*[_process_task(t, ctx) for t in tasks])

            print("\n" + "=" * 80)
            p = ctx.progresses
            print(
                f"DONE — decompose: {p['decompose']['done']}/{p['decompose']['total']} | "
                f"verify: {p['verify']['ok']}/{p['verify']['done']} | "
                f"reason: {p['reason']['done']}"
            )
            print(
                f"→ View in graphvis: SFT Caches tab — "
                f"'{cfg.aug_cache_id}', '{cfg.qra_cache_id}'"
            )

        finally:
            if runtime_pool is not None:
                await runtime_pool.close_all()
            await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
