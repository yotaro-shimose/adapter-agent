"""study2_augment.py — augment verified knowledge from study2 with variants.

Given the verified items in the `study2` SFT cache, this script:
  1. For each item, runs an augmenter agent (Gemini + grep/read/ls source
     access) that proposes N variant task instructions exercising the same
     `library_name` API.
  2. For each variant, runs `solve_verify` with the original answer fed in
     as a `solved_subtask` — the solver writes a fresh answer; only
     verifier-passing variants are kept.
  3. Persists verified variants to the `study2_aug` SFT cache.

Run with:
    uv run scripts/study2_augment.py
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from agents import set_tracing_disabled
from dotenv import load_dotenv
from prisma import Json, Prisma

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.hierarchical.process.augment import (
    Variants,
    propose_variants,
)
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


# Source SFT cache: read verified items from here.
SOURCE_CACHE_ID = "study2"
# Output SFT cache: variants land here.
TARGET_CACHE_ID = "study2_aug"

# How many variants to propose per source item. Configurable.
VARIANTS_PER_ITEM = 3
# Only augment items that actually verified — failures aren't worth varying.
VERIFIED_ONLY = True
# Cap how many source items to process per run. None = all of them.
# Useful for short experiments — set to 5 to dry-run, then None for full pass.
SOURCE_LIMIT: int | None = 5

AUGMENTER_CONCURRENCY = 50
AUGMENTER_MAX_TURNS = 16

INVESTIGATION_CONCURRENCY = 25
INVESTIGATION_MAX_TURNS = 12


@dataclass
class SourceItem:
    item_id: int
    knowledge_id: str
    knowledge_title: str
    instruction: str  # the InvestigationTarget body, or full question fallback
    answer: str
    library_name: str


@dataclass(frozen=True)
class AugmentDispatch:
    """One verified-source × one variant — the unit dispatched to solve_verify."""

    source: SourceItem
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


_INVESTIGATION_TARGET_RE = re.compile(
    r"<InvestigationTarget>\s*([\s\S]*?)\s*</InvestigationTarget>"
)


def _extract_instruction(question: str) -> str:
    """study2 stores `Investigate ... <InvestigationTarget>{item}</InvestigationTarget>`.
    Pull the item back out so the augmenter sees the actual investigation topic
    rather than the boilerplate template."""
    m = _INVESTIGATION_TARGET_RE.search(question)
    return m.group(1).strip() if m else question.strip()


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


def _build_variant_task(variant: str, library_name: str) -> str:
    """Wrap a variant instruction into a solve_verify-ready task. The variant
    text already IS the concrete task — we just append the standard
    requirements suffix so the solver knows what to <submit>."""
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


async def _load_source_items(
    prisma: Prisma, library_name: str
) -> list[SourceItem]:
    where: dict = {"cache_id": SOURCE_CACHE_ID}
    if VERIFIED_ONLY:
        where["verified"] = True
    rows = await prisma.sftcacheitem.find_many(where=where)
    items: list[SourceItem] = []
    for r in rows:
        if not r.answer:
            continue  # nothing to anchor a variant on
        items.append(SourceItem(
            item_id=r.id,
            knowledge_id=r.knowledge_id,
            knowledge_title=r.knowledge_title,
            instruction=_extract_instruction(r.question),
            answer=r.answer,
            library_name=library_name,
        ))
    return items


async def _augment_one(
    src: SourceItem,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    augmenter_model,
    n_variants: int,
    max_turns: int,
    sem: asyncio.Semaphore,
    progress: dict,
) -> tuple[SourceItem, Variants | None, str | None]:
    async with sem:
        print(
            f"[aug start] item={src.item_id} kid={src.knowledge_id} "
            f"title={src.knowledge_title[:60]}",
            flush=True,
        )
        try:
            v = await propose_variants(
                original_instruction=src.instruction,
                original_answer=src.answer,
                library_name=library_spec.name,
                libdir=library_spec.libdir,
                library_summary=library_summary,
                n_variants=n_variants,
                solver_model=augmenter_model,
                max_turns=max_turns,
            )
            progress["done"] += 1
            n = len(v.variants) if v is not None else 0
            mark = "ok " if v is not None else "FAIL"
            print(
                f"[aug {mark}] {progress['done']}/{progress['total']} "
                f"item={src.item_id} -> {n} variants",
                flush=True,
            )
            return src, v, None
        except Exception as e:
            progress["done"] += 1
            print(
                f"[aug ERR] {progress['done']}/{progress['total']} "
                f"item={src.item_id}: {e}",
                flush=True,
            )
            logger.exception(f"augmenter crashed for item={src.item_id}")
            return src, None, str(e)


async def _verify_one(
    dispatch: AugmentDispatch,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    solver_model,
    verifier_model,
    runtime_pool,
    sem: asyncio.Semaphore,
    progress: dict,
) -> AugmentResult:
    async with sem:
        print(
            f"[ver start] item={dispatch.source.item_id} "
            f"variant: {dispatch.variant_instruction[:80]}",
            flush=True,
        )
        # Hand the original (instruction, answer) in as a solved subtask so
        # the solver knows which API is in scope without rediscovering it.
        solved = [SolvedSubtask(
            instruction=dispatch.source.instruction,
            submit_code=dispatch.source.answer,
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
                max_turns=INVESTIGATION_MAX_TURNS,
                solved_subtasks=solved,
                reference_knowledge=library_summary,
                # The SolvedSubtask hint already carries the API; allowing
                # source search here would let the solver drift to unrelated
                # APIs, defeating the augmentation goal.
                enable_search_tools=False,
            )
        except Exception as e:
            progress["done"] += 1
            print(
                f"[ver ERR] {progress['done']}/{progress['total']} "
                f"item={dispatch.source.item_id}: {e}",
                flush=True,
            )
            logger.exception(f"[item={dispatch.source.item_id}] solver crashed")
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
                logger.exception(
                    f"[item={dispatch.source.item_id}] failed to serialize trials"
                )
        progress["done"] += 1
        if success:
            progress["ok"] += 1
        mark = "OK  " if success else "FAIL"
        conclusion = getattr(result, "conclusion", "unknown")
        print(
            f"[ver {mark}] {progress['done']}/{progress['total']} "
            f"(ok={progress['ok']}) item={dispatch.source.item_id} "
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


async def _persist(prisma: Prisma, r: AugmentResult) -> None:
    d = r.dispatch
    title = d.variant_instruction
    title = title if len(title) <= 120 else title[:117] + "..."
    data: dict = {
        "cache_id": TARGET_CACHE_ID,
        # Keep the source link visible: `<orig task id>#aug<source item id>`.
        "knowledge_id": f"{d.source.knowledge_id}#aug{d.source.item_id}",
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


async def main() -> None:
    load_dotenv()
    library_spec = LibrarySpec.hisab()

    try:
        library_summary = library_spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    logger.info(
        f"Loaded library summary ({len(library_summary)} chars) from "
        f"{library_spec.summary_path}."
    )

    prisma = Prisma()
    await prisma.connect()
    runtime_pool: RuntimePool | None = None
    try:
        sources = await _load_source_items(prisma, library_spec.name)
        n_loaded = len(sources)
        if SOURCE_LIMIT is not None:
            sources = sources[:SOURCE_LIMIT]
        logger.info(
            f"Loaded {n_loaded} source item(s) from cache_id={SOURCE_CACHE_ID} "
            f"(verified_only={VERIFIED_ONLY}); using {len(sources)} "
            f"(SOURCE_LIMIT={SOURCE_LIMIT})."
        )
        if not sources:
            print("No source items to augment. Exiting.")
            return

        # Reset target cache so each run is fresh.
        if await prisma.sftcache.find_unique(where={"id": TARGET_CACHE_ID}) is not None:
            await prisma.sftcache.delete(where={"id": TARGET_CACHE_ID})
            logger.info(f"Cleared existing '{TARGET_CACHE_ID}' SFT cache.")
        await prisma.sftcache.create(
            data={
                "id": TARGET_CACHE_ID,
                "library_name": library_spec.name,
                "description": (
                    f"study2_aug: {VARIANTS_PER_ITEM} variants × {len(sources)} verified "
                    f"items from {SOURCE_CACHE_ID}, verified via solve_verify."
                ),
            }
        )

        # Pipelined: as soon as a source's augmenter returns variants, we
        # immediately fire off `solve_verify` for each variant — phase 2 does
        # NOT wait for phase 1 to fully drain. The two semaphores bound the
        # actual concurrency.
        print("\n" + "=" * 80)
        print(
            f"PIPELINING — {len(sources)} sources × up to {VARIANTS_PER_ITEM} variants "
            f"(aug_conc={AUGMENTER_CONCURRENCY}, ver_conc={INVESTIGATION_CONCURRENCY})"
        )
        print("=" * 80)

        augmenter_model = get_gemini()
        solver_model = get_gemini()
        verifier_model = get_gemini_lite()
        runtime_pool = RuntimePool(
            settings=library_spec.docker_runtime(),
            max_size=INVESTIGATION_CONCURRENCY,
        )

        aug_sem = asyncio.Semaphore(AUGMENTER_CONCURRENCY)
        v_sem = asyncio.Semaphore(INVESTIGATION_CONCURRENCY)

        aug_progress = {"done": 0, "total": len(sources)}
        # Theoretical upper bound; actual total depends on which augmenters succeed.
        ver_progress = {
            "done": 0,
            "ok": 0,
            "total": len(sources) * VARIANTS_PER_ITEM,
        }

        async def _verify_and_persist(d: AugmentDispatch) -> AugmentResult:
            r = await _verify_one(
                d,
                library_spec=library_spec,
                library_summary=library_summary,
                solver_model=solver_model,
                verifier_model=verifier_model,
                runtime_pool=runtime_pool,
                sem=v_sem,
                progress=ver_progress,
            )
            try:
                await _persist(prisma, r)
                print(
                    f"[persisted] item={d.source.item_id} verified={r.success}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[persist ERR] item={d.source.item_id}: {e}",
                    flush=True,
                )
                logger.exception(f"[item={d.source.item_id}] persist failed")
            return r

        async def _augment_then_verify(src: SourceItem) -> list[AugmentResult]:
            src, variants, err = await _augment_one(
                src,
                library_spec=library_spec,
                library_summary=library_summary,
                augmenter_model=augmenter_model,
                n_variants=VARIANTS_PER_ITEM,
                max_turns=AUGMENTER_MAX_TURNS,
                sem=aug_sem,
                progress=aug_progress,
            )
            if err is not None or variants is None:
                return []
            dispatches = [
                AugmentDispatch(source=src, variant_instruction=v)
                for v in variants.variants[:VARIANTS_PER_ITEM]
            ]
            return await asyncio.gather(*[_verify_and_persist(d) for d in dispatches])

        per_source_results = await asyncio.gather(
            *[_augment_then_verify(s) for s in sources]
        )
        results = [r for sub in per_source_results for r in sub]

        n_ok = sum(1 for r in results if r.success)
        print("\n" + "=" * 80)
        print(f"RESULTS — {n_ok}/{len(results)} variants verified.")
        print("=" * 80)
        print(f"→ View in graphvis: SFT Caches tab, cache_id='{TARGET_CACHE_ID}'")

    finally:
        if runtime_pool is not None:
            await runtime_pool.close_all()
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
