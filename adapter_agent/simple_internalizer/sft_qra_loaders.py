"""SFT QRA loaders — turn arbitrary data sources into `SftSuite` objects.

The SimplePipeline's SFT stage takes a `list[SftSuite]` as its only input.
This module collects the available loaders so callers can mix and match:

  * `load_granular_sft_suite`     — generate k QRAs per `Knowledge` row
                                    (the original `_prepare_sft_qras` body).
  * `load_sft_cache_suite`        — read pre-existing QRA rows from the
                                    `sft_cache_items` table by cache_id.
                                    Used for the augmentation pipeline's
                                    output (`pipeline_v1_qra` etc.).
  * `load_study_root_sft_suite`   — distill solved Study trajectories per
                                    root task (wraps the existing
                                    `load_study_root_qras_cached`).

All loaders share signature `(ctx: SftLoaderContext, *, name, ...) -> SftSuite`
so callers can pre-bind source-specific args via `functools.partial` and
let `main()` provide the runtime context uniformly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from prisma import Prisma

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.simple_internalizer.data_sources import (
    ensure_sft_cache,
    generate_qras_cached,
    load_study_root_qras_cached,
)
from adapter_agent.simple_internalizer.types import SftSuite
from adapter_agent.study.qra_distiller import QRADistiller

logger = logging.getLogger(__name__)


@dataclass
class SftLoaderContext:
    """Runtime dependencies passed to every loader. Each loader picks the
    fields it actually needs and ignores the rest. Optional fields stay
    `None` when no source in the run requires them."""

    prisma: Prisma
    library_name: str
    cache_dir: Path
    generation_concurrency: int
    knowledge_list: list[Knowledge] | None = None  # required for granular loader
    generator: GeneratorAgent | None = None        # required for granular loader
    distiller: QRADistiller | None = None          # required for study-root loader


# Public type alias so recipes can spell their `sft_sources` list cleanly.
SftSuiteFactory = Callable[[SftLoaderContext], Awaitable[SftSuite]]


async def load_granular_sft_suite(
    ctx: SftLoaderContext,
    *,
    name: str,
    cache_id: str,
    k_per_knowledge: int,
) -> SftSuite:
    """Generate `k_per_knowledge` QRAs per item in `ctx.knowledge_list`,
    caching to `sft_cache_items` under `cache_id`. Returns the flattened
    list as one suite.

    Lifted verbatim from `SimplePipeline._prepare_sft_qras` so behavior
    stays identical for callers migrating off the old monolithic pipeline.
    """
    assert ctx.knowledge_list is not None, (
        "load_granular_sft_suite needs ctx.knowledge_list"
    )
    assert ctx.generator is not None, (
        "load_granular_sft_suite needs ctx.generator"
    )
    await ensure_sft_cache(
        ctx.prisma,
        cache_id,
        library_name=ctx.library_name,
        description=f"SFT bootstrap k_per_knowledge={k_per_knowledge}",
    )
    per_knowledge_qras: list[list[QRA]] = await asyncio.gather(*[
        generate_qras_cached(
            generator=ctx.generator,
            knowledge=k,
            count=k_per_knowledge,
            cache_id=cache_id,
            prisma_client=ctx.prisma,
            generation_concurrency=ctx.generation_concurrency,
        )
        for k in ctx.knowledge_list
    ])
    _print_granular_summary(ctx.knowledge_list, per_knowledge_qras, k_per_knowledge)
    flat = [q for qs in per_knowledge_qras for q in qs]
    logger.info(
        f"Granular SFT suite '{name}' (cache_id={cache_id}): "
        f"{len(flat)} QRAs from {len(ctx.knowledge_list)} knowledges."
    )
    return SftSuite(name=name, qras=flat)


async def load_sft_cache_suite(
    ctx: SftLoaderContext,
    *,
    name: str,
    cache_id: str,
    verified_only: bool = True,
) -> SftSuite:
    """Read existing QRA rows from `sft_cache_items` by cache_id.

    Used for SFT data produced out-of-band (e.g. by the augmentation
    pipeline writing to `pipeline_v1_qra`). Items must have non-empty
    answer + reasoning to be usable for SFT — rows missing either are
    skipped with a warning.
    """
    where: dict = {"cache_id": cache_id}
    if verified_only:
        where["verified"] = True
    rows = await ctx.prisma.sftcacheitem.find_many(where=where)

    qras: list[QRA] = []
    skipped = 0
    for r in rows:
        if not r.answer or not r.reasoning:
            skipped += 1
            continue
        qras.append(QRA(question=r.question, reasoning=r.reasoning, answer=r.answer))
    if skipped:
        logger.warning(
            f"SftCache suite '{name}' (cache_id={cache_id}): skipped {skipped} "
            f"row(s) missing answer or reasoning."
        )
    logger.info(
        f"SftCache suite '{name}' (cache_id={cache_id}): {len(qras)} QRAs "
        f"(verified_only={verified_only})."
    )
    return SftSuite(name=name, qras=qras)


async def load_study_root_sft_suite(
    ctx: SftLoaderContext,
    *,
    name: str,
    experiment_name: str,
    traj_qra_id: str,
    distill_concurrency: int = 50,
) -> SftSuite:
    """Distill 1 QRA per solved root task in a Study experiment. Wraps the
    pre-existing `load_study_root_qras_cached` so the loader API is uniform."""
    assert ctx.distiller is not None, (
        "load_study_root_sft_suite needs ctx.distiller"
    )
    qras = await load_study_root_qras_cached(
        prisma_client=ctx.prisma,
        experiment_name=experiment_name,
        traj_qra_id=traj_qra_id,
        distiller=ctx.distiller,
        cache_dir=ctx.cache_dir,
        distill_concurrency=distill_concurrency,
    )
    logger.info(
        f"StudyRoot suite '{name}' (experiment={experiment_name}, "
        f"traj_qra_id={traj_qra_id}): {len(qras)} QRAs."
    )
    return SftSuite(name=name, qras=qras)


# --- Internals ---


def _print_granular_summary(
    knowledge_list: list[Knowledge],
    per_knowledge_qras: list[list[QRA]],
    target: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"{'Knowledge Title':<50} | {'Target':<6} | {'Success':<7} | {'Status'}")
    print("-" * 80)
    for k, qras in zip(knowledge_list, per_knowledge_qras):
        success = len(qras)
        if success == target:
            status = "✅ OK"
        elif success > 0:
            status = "⚠️  PARTIAL"
        else:
            status = "❌ FAILED"
        title = (k.title[:47] + "...") if len(k.title) > 50 else k.title
        print(f"{title:<50} | {target:<6} | {success:<7} | {status}")
    print("=" * 80 + "\n")
