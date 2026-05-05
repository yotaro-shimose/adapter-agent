"""qra_fill.py — fill the Reasoning between Q and A for SFT data.

Reads verified (Q, A) pairs from a source SFT cache (default `study2_aug`),
asks the reasoner agent to generate the chain-of-thought R that connects
them, and persists the resulting (Q, R, A) triples to a new SFT cache
(default `study2_aug_qra`).

PoC: starts with `SOURCE_LIMIT = 5` so you can sanity-check the reasoning
quality before running on the full set. Set `SOURCE_LIMIT = None` for full.

Run with:
    uv run scripts/qra_fill.py
"""

import asyncio
import logging
from dataclasses import dataclass

from dotenv import load_dotenv
from prisma import Prisma

from adapter_agent.hierarchical.process.reasoner import fill_reasoning
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.model_helper import get_gemini
from adapter_agent.util.logger_util import setup_base_loglevel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
setup_base_loglevel()
logger = logging.getLogger(__name__)


# === Source / target ===
# Points at study2_pipeline.py's variant cache by default — run this script
# standalone after a pipeline finishes (or after a pre-QRA pipeline ran) to
# fill reasoning over its verified variants.
#
# Other source candidates:
#   - "study2_aug"  : standalone study2_augment.py output
#   - "study2"      : raw investigation Q&A (no variants)
# Match `EXPERIMENT_ID` in study2_pipeline.py if you want a different run.
SOURCE_CACHE_ID = "pipeline_v1_aug"
TARGET_CACHE_ID = "pipeline_v1_qra"
# Only fill reasoning for items that already verified — failures don't have
# a meaningful answer to anchor the reasoning on.
VERIFIED_ONLY = True
# Cap how many source items to process per run. None = all.
SOURCE_LIMIT: int | None = None

# === Concurrency ===
# Pure LLM calls, no docker — pump it up.
CONCURRENCY = 50


@dataclass
class QraInput:
    item_id: int
    knowledge_id: str
    knowledge_title: str
    question: str
    answer: str


@dataclass
class QraResult:
    src: QraInput
    reasoning: str | None
    error: str | None = None


async def _load_source_items(prisma: Prisma) -> list[QraInput]:
    where: dict = {"cache_id": SOURCE_CACHE_ID}
    if VERIFIED_ONLY:
        where["verified"] = True
    rows = await prisma.sftcacheitem.find_many(where=where)
    items: list[QraInput] = []
    for r in rows:
        if not r.answer:
            continue  # nothing to anchor reasoning on
        items.append(QraInput(
            item_id=r.id,
            knowledge_id=r.knowledge_id,
            knowledge_title=r.knowledge_title,
            question=r.question,
            answer=r.answer,
        ))
    return items


async def _fill_one(
    src: QraInput,
    *,
    library_spec: LibrarySpec,
    library_summary: str,
    model,
    sem: asyncio.Semaphore,
    progress: dict,
) -> QraResult:
    async with sem:
        try:
            reasoning = await fill_reasoning(
                question=src.question,
                answer=src.answer,
                library_name=library_spec.name,
                library_summary=library_summary,
                model=model,
            )
            progress["done"] += 1
            mark = "ok " if reasoning else "FAIL"
            words = len(reasoning.split()) if reasoning else 0
            print(
                f"[qra {mark}] {progress['done']}/{progress['total']} "
                f"item={src.item_id} words={words}",
                flush=True,
            )
            return QraResult(src=src, reasoning=reasoning)
        except Exception as e:
            progress["done"] += 1
            print(
                f"[qra ERR] {progress['done']}/{progress['total']} "
                f"item={src.item_id}: {e}",
                flush=True,
            )
            logger.exception(f"reasoner crashed for item={src.item_id}")
            return QraResult(src=src, reasoning=None, error=str(e))


async def _persist(prisma: Prisma, r: QraResult) -> None:
    s = r.src
    await prisma.sftcacheitem.create(data={
        "cache_id": TARGET_CACHE_ID,
        # Source-link back to the original Q&A row.
        "knowledge_id": f"{s.knowledge_id}#qra{s.item_id}",
        "knowledge_title": s.knowledge_title,
        "question": s.question,
        "reasoning": r.reasoning or "",
        "answer": s.answer,
        # Inherit "verified" from the source — if the source was verified
        # the QA pair is sound, and the reasoning is purely descriptive.
        "verified": True,
        "verifier_reasoning": "",
        "conclusion": "reasoning_filled" if r.reasoning else "reasoning_failed",
    })


async def main() -> None:
    load_dotenv()
    library_spec = LibrarySpec.hisab()

    try:
        library_summary = library_spec.read_summary()
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    prisma = Prisma()
    await prisma.connect()
    try:
        sources = await _load_source_items(prisma)
        n_loaded = len(sources)
        if SOURCE_LIMIT is not None:
            sources = sources[:SOURCE_LIMIT]
        logger.info(
            f"Loaded {n_loaded} source item(s) from cache_id={SOURCE_CACHE_ID} "
            f"(verified_only={VERIFIED_ONLY}); using {len(sources)} "
            f"(SOURCE_LIMIT={SOURCE_LIMIT})."
        )
        if not sources:
            print("No source items to process. Exiting.")
            return

        # Reset target cache so each run starts fresh.
        if await prisma.sftcache.find_unique(where={"id": TARGET_CACHE_ID}) is not None:
            await prisma.sftcache.delete(where={"id": TARGET_CACHE_ID})
            logger.info(f"Cleared existing '{TARGET_CACHE_ID}' SFT cache.")
        await prisma.sftcache.create(data={
            "id": TARGET_CACHE_ID,
            "library_name": library_spec.name,
            "description": (
                f"qra_fill: chain-of-thought reasoning filled in for "
                f"{len(sources)} verified QA pair(s) from {SOURCE_CACHE_ID}."
            ),
        })

        print("\n" + "=" * 80)
        print(
            f"FILLING REASONING — {len(sources)} items "
            f"(concurrency={CONCURRENCY})"
        )
        print("=" * 80)

        model = get_gemini()
        sem = asyncio.Semaphore(CONCURRENCY)
        progress = {"done": 0, "total": len(sources)}

        async def _fill_and_persist(src: QraInput) -> QraResult:
            r = await _fill_one(
                src,
                library_spec=library_spec,
                library_summary=library_summary,
                model=model,
                sem=sem,
                progress=progress,
            )
            try:
                await _persist(prisma, r)
            except Exception as e:
                print(
                    f"[persist ERR] item={src.item_id}: {e}",
                    flush=True,
                )
                logger.exception(f"persist failed: item={src.item_id}")
            return r

        results = await asyncio.gather(*[_fill_and_persist(s) for s in sources])
        n_ok = sum(1 for r in results if r.reasoning)

        print("\n" + "=" * 80)
        print(f"RESULTS — {n_ok}/{len(results)} items got reasoning.")
        print("=" * 80)
        print(f"→ View in graphvis: SFT Caches tab, cache_id='{TARGET_CACHE_ID}'")

    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
