"""qra_pipeline — shared stages for QRA-generation pipelines.

Holds the per-stage runners, persistence helpers, dataclasses, and cache
helpers that are common to any pipeline that turns library investigations
into SFT-ready (question, reasoning, answer) triples.

Originally lived inline in `scripts/study2_pipeline.py`. Extracted here so
the failure-recovery pipeline (`scripts/regenerate_to_qra_pipeline.py`)
can reuse the exact same investigate → augment → variant-verify →
reason → persist chain without duplicating helpers or reaching across
scripts.

The driving pipeline is responsible for:
  - building its own plan stage (study2 plans per gh_archive task;
    failure-recovery plans per `FailedAttempt`);
  - constructing a `StageContext` that satisfies the duck-typed
    `recipe` field (see `StageContext.recipe` doc);
  - calling `run_investigate(...)` (or seeding `Investigation` objects
    some other way) and then `augment_verify_reason(...)` per verified
    investigation.

The `recipe` attribute on `StageContext` is intentionally typed `Any`:
the stages read only a handful of paths (`recipe.library_spec`,
`recipe.stage.investigation_max_turns`,
`recipe.stage.variant_verify_max_turns`,
`recipe.workload.variants_per_item`,
`recipe.cache.{inv,aug,qra}_cache_id`). Each pipeline supplies its own
config dataclass that exposes those fields.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from prisma import Json, Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.process.augment import Variants, propose_variants
from adapter_agent.hierarchical.process.reasoner import fill_reasoning
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.env.session_result import RewireSessionResultSuccess
from adapter_agent.rl.solved_subtask import SolvedSubtask

logger = logging.getLogger(__name__)


# === Data classes ====================================================


@dataclass(frozen=True)
class InvestigationDispatch:
    task_id: str
    task_instruction: str
    item: str
    # Optional original-failure context (failure-recovery pipeline). When
    # present, the investigator sees the failed agent's code + error/feedback,
    # giving it concrete grep hints instead of an abstract plan item.
    failure_answer: str | None = None
    failure_execution_output: str | None = None
    failure_verifier_feedback: str | None = None


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


@dataclass
class StageContext:
    """Shared per-run state threaded through every stage runner.

    `recipe` is intentionally `Any` — each driving pipeline supplies its
    own config dataclass. The stage runners only access these paths:
      - `recipe.library_spec`  (LibrarySpec)
      - `recipe.stage.investigation_max_turns`  (int)
      - `recipe.stage.variant_verify_max_turns`  (int)
      - `recipe.workload.variants_per_item`  (int)
      - `recipe.cache.inv_cache_id`  (str)
      - `recipe.cache.aug_cache_id`  (str)
      - `recipe.cache.qra_cache_id`  (str)

    `sems` keys read by the shared stages: `"inv"`, `"aug"`, `"var"`,
    `"qra"`. `progresses` keys: same plus optional `"plan"` driven by
    the per-pipeline plan stage.

    Bundling this state cuts each runner's signature from ~10 kwargs to
    two (the dispatch object + this context).
    """

    recipe: Any
    library_summary: str
    runtime_pool: RuntimePool
    prisma: Prisma

    planner_model: object = None
    solver_model: object = None
    verifier_model: object = None
    augmenter_model: object = None
    reasoner_model: object = None

    sems: dict = field(default_factory=dict)
    progresses: dict = field(default_factory=dict)


# === Helpers =========================================================


_INVESTIGATION_TARGET_RE = re.compile(
    r"<InvestigationTarget>\s*([\s\S]*?)\s*</InvestigationTarget>"
)


def serialize_trials(trials) -> list:
    return [
        PydanticTinkerBaseMessage.model_validate(m).model_dump(
            mode="json", exclude_none=True
        )
        for m in trials
    ]


def extract_submit(trials) -> str | None:
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


def extract_investigation_target(question: str) -> str:
    """The inv cache stores `Investigate ... <InvestigationTarget>{item}</InvestigationTarget>`.
    Pull the item back out so downstream code sees the actual investigation
    topic rather than the boilerplate template."""
    m = _INVESTIGATION_TARGET_RE.search(question)
    return m.group(1).strip() if m else question.strip()


def build_investigation_task(
    item: str,
    library_name: str,
    *,
    failure_answer: str | None = None,
    failure_execution_output: str | None = None,
    failure_verifier_feedback: str | None = None,
) -> str:
    failure_block = ""
    if failure_answer or failure_execution_output or failure_verifier_feedback:
        parts: list[str] = [
            "",
            "<OriginalFailure>",
            "Context: the investigation target was extracted from a real failed",
            f"agent attempt to use `{library_name}`. The agent's submitted code",
            "and runtime output are below — use them as concrete grep targets",
            "(error line numbers, offending identifier names) to locate the",
            "correct API, instead of guessing names.",
        ]
        if failure_answer:
            parts.extend([
                "",
                "<FailedAnswer>",
                failure_answer.strip(),
                "</FailedAnswer>",
            ])
        if failure_execution_output:
            parts.extend([
                "",
                "<ExecutionOutput>",
                failure_execution_output.strip(),
                "</ExecutionOutput>",
            ])
        if failure_verifier_feedback:
            parts.extend([
                "",
                "<VerifierFeedback>",
                failure_verifier_feedback.strip(),
                "</VerifierFeedback>",
            ])
        parts.append("</OriginalFailure>")
        failure_block = "\n".join(parts) + "\n"

    return f"""\
Investigate the following aspect of the `{library_name}` library and produce a
runnable Rust code example that demonstrates it.

<InvestigationTarget>
{item}
</InvestigationTarget>
{failure_block}
Requirements for your final `<submit>`:
- A complete `fn main()` program (no missing pieces).
- It MUST exercise the actual `{library_name}` API named by the investigation
  target — not a hand-rolled equivalent in plain std.
- It MUST compile and run successfully via `cargo run`.
- It SHOULD print output that makes the demonstrated behavior visible
  (e.g. computed values, intermediate states).
- Keep it minimal — just enough to clearly show how the API is used.
- Do NOT submit code calling identifiers you have not verified exist
  (via `<grep>` / `<read>`) in the library source — fabricating names
  wastes turns on compile-error loops.
"""


def build_variant_task(variant: str, library_name: str) -> str:
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


# === Persistence =====================================================


async def persist_investigation(
    prisma: Prisma, inv: Investigation, *, cache_id: str
) -> None:
    d = inv.dispatch
    item_label = d.item if len(d.item) <= 120 else d.item[:117] + "..."
    data: dict = {
        "cache_id": cache_id,
        "knowledge_id": d.task_id,
        "knowledge_title": item_label,
        "question": build_investigation_task(d.item, "").replace("``", "").strip(),
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


async def persist_qra(
    prisma: Prisma, r: AugmentResult, reasoning: str, *, cache_id: str
) -> None:
    """Persist a verified variant with reasoning filled in — the SFT-ready
    QRA triple. Only called for variants that already verified."""
    d = r.dispatch
    title = d.variant_instruction
    title = title if len(title) <= 120 else title[:117] + "..."
    assert r.submit_code is not None  # only called for verified variants
    await prisma.sftcacheitem.create(
        data={
            "cache_id": cache_id,
            "knowledge_id": f"{d.investigation.dispatch.task_id}#qra",
            "knowledge_title": title,
            "question": d.variant_instruction,
            "reasoning": reasoning,
            "answer": r.submit_code,
            # Source variant was already verified — reasoning is descriptive.
            "verified": True,
            "verifier_reasoning": "",
            "conclusion": "reasoning_filled",
        }
    )


async def persist_variant(prisma: Prisma, r: AugmentResult, *, cache_id: str) -> None:
    d = r.dispatch
    title = d.variant_instruction
    title = title if len(title) <= 120 else title[:117] + "..."
    data: dict = {
        "cache_id": cache_id,
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


# === Stage runners ===================================================


async def run_investigate(
    dispatch: InvestigationDispatch, ctx: StageContext
) -> Investigation:
    sem = ctx.sems["inv"]
    progress = ctx.progresses["inv"]
    spec = ctx.recipe.library_spec
    async with sem:
        try:
            result = await solve_verify(
                solver_model=ctx.solver_model,
                verifier_model=ctx.verifier_model,
                task=Task(
                    instruction=build_investigation_task(
                        dispatch.item,
                        spec.name,
                        failure_answer=dispatch.failure_answer,
                        failure_execution_output=dispatch.failure_execution_output,
                        failure_verifier_feedback=dispatch.failure_verifier_feedback,
                    )
                ),
                libdir=spec.libdir,
                library_name=spec.name,
                runtime_pool=ctx.runtime_pool,
                max_turns=ctx.recipe.stage.investigation_max_turns,
                reference_knowledge=ctx.library_summary,
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
            submit = extract_submit(result.trials)
            try:
                trials_serialized = serialize_trials(result.trials)
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


async def run_augment(inv: Investigation, ctx: StageContext) -> Variants | None:
    sem = ctx.sems["aug"]
    progress = ctx.progresses["aug"]
    spec = ctx.recipe.library_spec
    async with sem:
        try:
            assert inv.submit_code is not None  # caller filters out None/empty
            v = await propose_variants(
                original_instruction=inv.dispatch.item,
                original_answer=inv.submit_code,
                library_name=spec.name,
                library_summary=ctx.library_summary,
                n_variants=ctx.recipe.workload.variants_per_item,
                solver_model=ctx.augmenter_model,
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


async def run_variant_verify(
    dispatch: AugmentDispatch, ctx: StageContext
) -> AugmentResult:
    sem = ctx.sems["var"]
    progress = ctx.progresses["var"]
    spec = ctx.recipe.library_spec
    async with sem:
        # Hand the verified original (instruction, answer) to the solver as
        # a SolvedSubtask hint so it knows which API to use without searching.
        assert dispatch.investigation.submit_code is not None  # caller filters
        solved = [
            SolvedSubtask(
                instruction=dispatch.investigation.dispatch.item,
                submit_code=dispatch.investigation.submit_code,
            )
        ]
        try:
            result = await solve_verify(
                solver_model=ctx.solver_model,
                verifier_model=ctx.verifier_model,
                task=Task(
                    instruction=build_variant_task(
                        dispatch.variant_instruction, spec.name
                    )
                ),
                libdir=spec.libdir,
                library_name=spec.name,
                runtime_pool=ctx.runtime_pool,
                max_turns=ctx.recipe.stage.variant_verify_max_turns,
                solved_subtasks=solved,
                reference_knowledge=ctx.library_summary,
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
            submit = extract_submit(result.trials)
            try:
                trials_serialized = serialize_trials(result.trials)
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


async def run_reason(r: AugmentResult, ctx: StageContext) -> str | None:
    """Generate the chain-of-thought between the variant question and its
    verified answer. Caller persists the result if non-None."""
    sem = ctx.sems["qra"]
    progress = ctx.progresses["qra"]
    spec = ctx.recipe.library_spec
    async with sem:
        try:
            assert r.submit_code is not None  # caller filters unverified variants
            reasoning = await fill_reasoning(
                question=r.dispatch.variant_instruction,
                answer=r.submit_code,
                library_name=spec.name,
                library_summary=ctx.library_summary,
                model=ctx.reasoner_model,
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


# === Composite =======================================================


async def augment_verify_reason(inv: Investigation, ctx: StageContext) -> None:
    """Stages 3-5: augment → variant-verify → reason. Shared between any
    driver that produces verified `Investigation` objects (study2's PLAN
    mode, study2's AUGMENT mode that reloads inv-cache rows, the failure-
    recovery pipeline, etc.). Persistence target ids come from
    `ctx.recipe.cache.{aug,qra}_cache_id`."""
    cache = ctx.recipe.cache
    workload = ctx.recipe.workload

    variants = await run_augment(inv, ctx)
    if variants is None or not variants.variants:
        return

    dispatches = [
        AugmentDispatch(investigation=inv, variant_instruction=v)
        for v in variants.variants[: workload.variants_per_item]
    ]

    async def _verify_and_persist(d: AugmentDispatch) -> None:
        r = await run_variant_verify(d, ctx)
        try:
            await persist_variant(ctx.prisma, r, cache_id=cache.aug_cache_id)
        except Exception:
            logger.exception(
                f"persist variant failed: task={d.investigation.dispatch.task_id}"
            )

        if not r.success or not r.submit_code:
            return
        assert r.submit_code is not None  # narrowed by guard above
        reasoning = await run_reason(r, ctx)
        if reasoning is None:
            return
        try:
            await persist_qra(ctx.prisma, r, reasoning, cache_id=cache.qra_cache_id)
        except Exception:
            logger.exception(
                f"persist qra failed: task={d.investigation.dispatch.task_id}"
            )

    await asyncio.gather(*[_verify_and_persist(d) for d in dispatches])


# === Cache helpers ===================================================


async def ensure_cache(
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


async def load_investigations_from_cache(
    prisma: Prisma, cache_id: str, *, verified_only: bool
) -> list[Investigation]:
    """Reconstitute `Investigation` objects from a previously-persisted inv
    cache. The DB schema has the verified bit, the original answer, and the
    question template — that's everything `augment_verify_reason` needs
    downstream.

    `task_id` recovery: `persist_investigation` writes the original task_id
    into `knowledge_id` verbatim, so we read it straight back."""
    where: dict = {"cache_id": cache_id}
    if verified_only:
        where["verified"] = True
    rows = await prisma.sftcacheitem.find_many(where=where)
    investigations: list[Investigation] = []
    for r in rows:
        if not r.answer:
            continue  # nothing to anchor a variant on
        item = extract_investigation_target(r.question)
        dispatch = InvestigationDispatch(
            task_id=r.knowledge_id,
            task_instruction="",  # not used downstream of aug stage
            item=item,
        )
        investigations.append(
            Investigation(
                dispatch=dispatch,
                success=r.verified,
                conclusion=r.conclusion,
                submit_code=r.answer,
                verifier_reasoning=r.verifier_reasoning,
            )
        )
    return investigations
