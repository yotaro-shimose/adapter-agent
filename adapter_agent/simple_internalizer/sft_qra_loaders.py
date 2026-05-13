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
    take_n: int | None = None,
    take_seed: int = 42,
) -> SftSuite:
    """Read existing QRA rows from `sft_cache_items` by cache_id.

    Used for SFT data produced out-of-band (e.g. by the augmentation
    pipeline writing to `pipeline_v1_qra`). Items must have non-empty
    answer + reasoning to be usable for SFT — rows missing either are
    skipped with a warning.

    `take_n` deterministically subsamples the verified pool down to
    that count (using `random.Random(take_seed)`) — used to make a
    large replay pool fit cleanly into the per-batch mix alongside a
    much smaller fine-tune pool. `None` keeps everything.
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

    pre_subsample = len(qras)
    if take_n is not None and pre_subsample > take_n:
        import random as _random
        rng = _random.Random(take_seed)
        qras = rng.sample(qras, take_n)
        logger.info(
            f"SftCache suite '{name}' (cache_id={cache_id}): subsampled "
            f"{pre_subsample} → {len(qras)} QRAs (seed={take_seed})."
        )
    else:
        logger.info(
            f"SftCache suite '{name}' (cache_id={cache_id}): {len(qras)} QRAs "
            f"(verified_only={verified_only})."
        )
    return SftSuite(name=name, qras=qras)


async def load_rl_rollout_replay_suite(
    ctx: SftLoaderContext,
    *,
    name: str,
    simple_train_ids: list[str],
    take_n: int,
    take_seed: int = 42,
) -> SftSuite:
    """On-policy replay suite — pull successful (question, reasoning, answer)
    rollouts from `simplerlrollout` for use as SFT replay anchor.

    Rationale: SFT on out-of-band (e.g. Gemini-generated) QRAs drifts the
    model toward the data source's distribution; using the model's OWN past
    successful rollouts as replay keeps that anchor on-policy, so the
    forgetting brake doesn't introduce its own distribution shift.

    Sampling is two-stage to maximize *task* diversity and *step* recency:
      1. For each `simple_train_id`, group success rollouts by `task_id`
         and keep only the highest-`rl_step` success per task (one row).
      2. Concatenate the resulting per-task-latest-success rows across all
         provided train ids, then `random.Random(take_seed).sample` down
         to `take_n`. Task ids never collide across TaskRL (gh_archive) and
         KRL (pipeline_v2_qra) namespaces in practice.

    Typical use: pass [task_rl_run_id, knowledge_rl_run_id] so a single
    suite covers both the task-level and knowledge-level distributions
    the model has been trained on.
    """
    import random as _random

    per_run_pools: list[tuple[str, dict[str, tuple[int, str, str, str]]]] = []
    for tid in simple_train_ids:
        rows = await ctx.prisma.simplerlrollout.find_many(
            where={"simple_train_id": tid, "success": True},
            order={"rl_step": "desc"},
        )
        # task_id -> (rl_step, instruction, reasoning, answer) at latest step
        latest: dict[str, tuple[int, str, str, str]] = {}
        for r in rows:
            if not r.answer or not r.reasoning:
                continue
            if r.task_id in latest:
                # rows are ordered rl_step desc, so the first hit already
                # wins; subsequent same-task rows are older.
                continue
            latest[r.task_id] = (r.rl_step, r.instruction, r.reasoning, r.answer)
        per_run_pools.append((tid, latest))
        logger.info(
            f"RolloutReplay '{name}' / {tid}: {len(latest)} unique tasks "
            f"with success (latest rl_step per task)."
        )

    flat: list[tuple[str, str, str]] = []
    for _, latest in per_run_pools:
        for _step, q, r_text, a in latest.values():
            flat.append((q, r_text, a))

    if not flat:
        logger.warning(
            f"RolloutReplay '{name}': no success rollouts found across "
            f"{len(simple_train_ids)} run id(s)."
        )
        return SftSuite(name=name, qras=[])

    rng = _random.Random(take_seed)
    if len(flat) > take_n:
        picked = rng.sample(flat, take_n)
    else:
        picked = flat
    qras = [QRA(question=q, reasoning=r_text, answer=a) for q, r_text, a in picked]
    logger.info(
        f"RolloutReplay suite '{name}': pooled {len(flat)} task-latest "
        f"successes from {len(simple_train_ids)} run(s) → sampled "
        f"{len(qras)} (seed={take_seed})."
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
