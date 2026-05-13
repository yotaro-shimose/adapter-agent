"""Seed-suite factories for RL / eval task pools.

Mirrors `sft_qra_loaders.py` for the RL side: every recipe declares its
RL/eval task sources as a list of `SeedSuiteFactory` callables, each of
which returns one or more `SeedSuite` objects given a shared context.

A factory returns `list[SeedSuite]` rather than a single suite so per-
knowledge fan-out (`build_knowledge_suites`) and single-suite loaders
(`load_gh_archive_suite`) can share the same shape.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from prisma import Prisma

from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.types import Knowledge, Task
from adapter_agent.library.library_spec import LibrarySpec

from .data_sources import (
    build_knowledge_suites,
    load_gh_archive_suite,
    load_sft_cache_seed_suite,
)
from .types import SeedSuite

logger = logging.getLogger(__name__)


@dataclass
class SeedLoaderContext:
    """Shared dependencies for seed-suite factories. Built once per run.

    Mirrors `SftLoaderContext`. Loaders pick what they need; deps that a
    given recipe doesn't use can stay None (factories assert when missing).
    """

    prisma: Prisma
    library_spec: LibrarySpec
    generation_concurrency: int
    knowledge_list: list[Knowledge]
    generator: GeneratorAgent | None = None
    granular_id: str | None = None


SeedSuiteFactory = Callable[[SeedLoaderContext], Awaitable[list[SeedSuite]]]


async def load_knowledge_seed_suites(
    ctx: SeedLoaderContext,
    *,
    k_per_knowledge: int,
    name_prefix: str,
    cache_id: str,
    for_rl: bool,
    for_eval: bool,
) -> list[SeedSuite]:
    """One SeedSuite per granular-knowledge item, each with `k_per_knowledge`
    LLM-generated questions. Wraps `build_knowledge_suites` so it slots into
    the seed-source factory list."""
    assert ctx.generator is not None, (
        "load_knowledge_seed_suites needs ctx.generator"
    )
    assert ctx.knowledge_list, (
        "load_knowledge_seed_suites needs a non-empty ctx.knowledge_list "
        "(set Recipe.granular_id and ensure rows exist)."
    )
    suites = await build_knowledge_suites(
        generator=ctx.generator,
        knowledge_list=ctx.knowledge_list,
        k_per_knowledge=k_per_knowledge,
        cache_id=cache_id,
        prisma_client=ctx.prisma,
        name_prefix=name_prefix,
        granular_id=ctx.granular_id,
        library_name=ctx.library_spec.name,
        for_rl=for_rl,
        for_eval=for_eval,
        generation_concurrency=ctx.generation_concurrency,
    )
    logger.info(
        f"Knowledge seed suites '{name_prefix}': {len(suites)} suites "
        f"(k_per_knowledge={k_per_knowledge}, for_rl={for_rl}, for_eval={for_eval})."
    )
    return suites


async def load_gh_archive_seed_suite(
    ctx: SeedLoaderContext,
    *,
    name: str,
    task_slice: slice,
    for_rl: bool,
    for_eval: bool,
) -> list[SeedSuite]:
    """gh_archive benchmark slice as one SeedSuite. csv_path / difficulty
    come from `ctx.library_spec`."""
    suite = load_gh_archive_suite(
        name=name,
        task_slice=task_slice,
        for_rl=for_rl,
        for_eval=for_eval,
        csv_path=ctx.library_spec.benchmark_csv,
        difficulty=ctx.library_spec.default_difficulty,
    )
    sl = task_slice
    step_repr = f":{sl.step}" if sl.step is not None else ""
    logger.info(
        f"gh_archive seed suite '{name}': {len(suite.tasks)} tasks "
        f"(slice=[{sl.start}:{sl.stop}{step_repr}], for_rl={for_rl}, for_eval={for_eval})."
    )
    return [suite]


async def load_sft_cache_seed_suite_factory(
    ctx: SeedLoaderContext,
    *,
    name: str,
    cache_id: str,
    for_rl: bool,
    for_eval: bool,
    verified_only: bool = True,
) -> list[SeedSuite]:
    """SftCache rows as one SeedSuite (questions only). Used for RL/eval
    over question pools produced by out-of-band pipelines (e.g. study2_pipeline
    AUGMENT mode writing to `pipeline_v1_qra_v2`)."""
    suite = await load_sft_cache_seed_suite(
        ctx.prisma,
        name=name,
        cache_id=cache_id,
        for_rl=for_rl,
        for_eval=for_eval,
        verified_only=verified_only,
    )
    logger.info(
        f"SftCache seed suite '{name}' (cache_id={cache_id}): "
        f"{len(suite.tasks)} tasks (for_rl={for_rl}, for_eval={for_eval})."
    )
    return [suite]


async def load_sft_cache_seed_suite_factory_bucket_filtered(
    ctx: SeedLoaderContext,
    *,
    name: str,
    cache_id: str,
    routing_csv_path: str,
    include_buckets: tuple[str, ...],
    for_rl: bool,
    for_eval: bool,
    verified_only: bool = True,
) -> list[SeedSuite]:
    """Variant of `load_sft_cache_seed_suite_factory` that drops tasks whose
    routing bucket (from a `passatk_*` CSV) is NOT in `include_buckets`.

    The CSV is expected to have at minimum the columns `instruction` and
    `bucket` (the schema produced by `passatk_restudy.py`). Tasks are matched
    against the cache by exact `instruction` string equality — same key the
    cache uses for its `question` field.
    """
    import csv
    from pathlib import Path
    csv_path = Path(routing_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Routing CSV not found at {csv_path}. Run the passatk script "
            "that produces it first."
        )
    allowed_instructions: set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("bucket") in include_buckets:
                allowed_instructions.add(row["instruction"])

    suite = await load_sft_cache_seed_suite(
        ctx.prisma,
        name=name,
        cache_id=cache_id,
        for_rl=for_rl,
        for_eval=for_eval,
        verified_only=verified_only,
    )
    before = len(suite.tasks)
    kept = [t for t in suite.tasks if t.instruction in allowed_instructions]
    filtered = SeedSuite(name=name, tasks=kept, for_rl=for_rl, for_eval=for_eval)
    logger.info(
        f"SftCache seed suite '{name}' (cache_id={cache_id}, "
        f"filter={include_buckets}): {len(kept)}/{before} tasks "
        f"(for_rl={for_rl}, for_eval={for_eval})."
    )
    return [filtered]


__all__ = [
    "SeedLoaderContext",
    "SeedSuiteFactory",
    "load_knowledge_seed_suites",
    "load_gh_archive_seed_suite",
    "load_sft_cache_seed_suite_factory",
    "load_sft_cache_seed_suite_factory_bucket_filtered",
    "Task",  # convenience re-export, occasionally handy in recipe modules
]
