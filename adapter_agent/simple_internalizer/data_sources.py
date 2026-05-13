"""Data-loading helpers for SimplePipeline.

Keeps data acquisition separate from pipeline wiring: each function returns
a ready-to-inject domain object (Knowledge list, QRA list, or SeedSuite).
"""

import asyncio
import logging
import pickle
from pathlib import Path

from prisma import Prisma

from adapter_agent.data import PydanticTinkerBaseMessage, QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.types import Knowledge, Task
from adapter_agent.study.qra_distiller import QRADistiller

from .types import SeedSuite

logger = logging.getLogger(__name__)


async def load_granular_knowledge(
    prisma_client: Prisma, granular_id: str
) -> list[Knowledge]:
    granulars = await prisma_client.granularknowledge.find_many(
        where={"simple_train_id": granular_id}
    )
    if not granulars:
        raise ValueError(
            f"No granular knowledge found for granular-id={granular_id}."
        )
    return [Knowledge(id=g.id, title=g.title, content=g.content) for g in granulars]


async def load_study_solved_suite(
    prisma_client: Prisma,
    experiment_id: str,
    name: str = "study_solved",
    for_rl: bool = True,
    for_eval: bool = True,
) -> SeedSuite:
    """Fetch root tasks marked solved in a study experiment's graph."""
    experiment = await prisma_client.experiment.find_unique(
        where={"experiment_name": experiment_id}
    )
    if experiment is None or not experiment.graph_json:
        raise ValueError(f"Experiment '{experiment_id}' has no graph_json.")

    graph = experiment.graph_json
    pseudo_root_id = "pseudo_root"
    root_child_ids: set[str] = {
        e["target"]
        for e in graph.get("edges", [])
        if e.get("type") == "decomposition" and e.get("source") == pseudo_root_id
    }
    tasks: list[Task] = []
    for node in graph.get("nodes", []):
        if node.get("type") != "task" or node["id"] not in root_child_ids:
            continue
        if not node.get("metadata", {}).get("is_solved"):
            continue
        instruction = node["metadata"]["instruction"]
        tasks.append(Task(id=node["id"], instruction=instruction))
    return SeedSuite(name=name, tasks=tasks, for_rl=for_rl, for_eval=for_eval)


def load_gh_archive_suite(
    name: str,
    task_slice: slice,
    for_rl: bool,
    for_eval: bool,
    csv_path: Path | None = None,
    difficulty: str | None = "Easy",
) -> SeedSuite:
    """Pick a contiguous window out of load_gh_archive() as a SeedSuite.

    `csv_path` selects which library's benchmark to read; pass
    ``LibrarySpec.<name>().benchmark_csv`` to switch. ``None`` (default)
    falls back to ``load_gh_archive``'s built-in numrs2 path.

    `difficulty` filters rows. Default ``"Easy"`` matches the historical
    numrs2 behavior; pass ``None`` (or use ``LibrarySpec.default_difficulty``)
    to disable the filter — required for hisab where Easy rows are sparse.
    """
    if csv_path is not None:
        archive = load_gh_archive(difficulty=difficulty, csv_path=csv_path)
    else:
        archive = load_gh_archive(difficulty=difficulty)
    tasks: list[Task] = archive[task_slice]
    return SeedSuite(name=name, tasks=tasks, for_rl=for_rl, for_eval=for_eval)


async def load_sft_cache_seed_suite(
    prisma_client: Prisma,
    *,
    name: str,
    cache_id: str,
    for_rl: bool,
    for_eval: bool,
    verified_only: bool = True,
) -> SeedSuite:
    """Build a SeedSuite from rows in `sft_cache_items` for `cache_id`.

    Mirrors `load_sft_cache_suite` (which returns an SftSuite) but produces
    RL/eval seed tasks from the same rows — `Task.instruction = row.question`.
    Used when an out-of-band pipeline (e.g. study2_pipeline AUGMENT mode)
    has already produced verified question/answer pairs and we want to RL
    on those questions.
    """
    where: dict = {"cache_id": cache_id}
    if verified_only:
        where["verified"] = True
    rows = await prisma_client.sftcacheitem.find_many(where=where)
    tasks = [Task(instruction=r.question) for r in rows if r.question]
    return SeedSuite(name=name, tasks=tasks, for_rl=for_rl, for_eval=for_eval)


async def generate_qras_cached(
    generator: GeneratorAgent,
    knowledge: Knowledge,
    count: int,
    cache_id: str,
    prisma_client: Prisma,
    generation_concurrency: int,
    is_coding: bool = True,
) -> list[QRA]:
    """Generate `count` QRAs for one knowledge item, caching rows in `sft_cache_items`.

    Lookup key: (cache_id, knowledge_id). If at least `count` rows already exist
    we return the first `count`; otherwise we generate the deficit, INSERT, and
    return the union. Caller must ensure `cache_id` exists in `sft_caches`
    (use `ensure_sft_cache(...)` once per batch).
    """
    existing = await prisma_client.sftcacheitem.find_many(
        where={"cache_id": cache_id, "knowledge_id": knowledge.id},
        order={"id": "asc"},
        take=count,
    )
    if len(existing) >= count:
        logger.info(
            f"Cache hit: {count} {cache_id} QRAs for '{knowledge.title}'."
        )
        return [
            QRA(question=r.question, reasoning=r.reasoning, answer=r.answer)
            for r in existing[:count]
        ]

    deficit = count - len(existing)
    logger.info(
        f"Generating {deficit} {cache_id} QRAs for '{knowledge.title}' "
        f"(found {len(existing)} in cache)..."
    )
    sem = asyncio.Semaphore(generation_concurrency)

    async def _gen() -> QRA:
        async with sem:
            while True:
                qra = await (
                    generator.generate_sft(knowledge)
                    if is_coding
                    else generator.generate_sft_noncode(knowledge)
                )
                if qra is not None:
                    return qra
                await asyncio.sleep(0.1)

    new_qras: list[QRA] = await asyncio.gather(*[_gen() for _ in range(deficit)])

    if new_qras:
        await prisma_client.sftcacheitem.create_many(
            data=[
                {
                    "cache_id": cache_id,
                    "knowledge_id": knowledge.id,
                    "knowledge_title": knowledge.title,
                    "question": q.question,
                    "reasoning": q.reasoning,
                    "answer": q.answer,
                }
                for q in new_qras
            ]
        )

    return [
        QRA(question=r.question, reasoning=r.reasoning, answer=r.answer)
        for r in existing
    ] + new_qras


async def ensure_sft_cache(
    prisma_client: Prisma,
    cache_id: str,
    *,
    granular_id: str | None = None,
    library_name: str | None = None,
    description: str | None = None,
) -> None:
    """Idempotent upsert of the parent `sft_caches` row. Call once per batch
    before invoking `generate_qras_cached` so the FK in `sft_cache_items` resolves.
    """
    await prisma_client.sftcache.upsert(
        where={"id": cache_id},
        data={
            "create": {
                "id": cache_id,
                "granular_id": granular_id,
                "library_name": library_name,
                "description": description,
            },
            "update": {},
        },
    )


async def load_study_root_qras_cached(
    prisma_client: Prisma,
    experiment_name: str,
    traj_qra_id: str,
    distiller: QRADistiller,
    cache_dir: Path,
    distill_concurrency: int = 50,
) -> list[QRA]:
    """Distill 1 solved trajectory per root task into a QRA, cached on disk.

    The cache is scoped under `cache_dir / traj_qra_id /` so independent
    experiments (different distill prompts, model versions, etc.) don't
    collide. Bump `traj_qra_id` whenever distillation logic changes.

    Pulls the experiment's `graph_json`, finds root tasks via the
    `pseudo_root` decomposition edges (matching `load_study_solved_suite`),
    fetches the latest `is_sft_candidate=True` trajectory per root task,
    and runs `distiller.distill(instruction, trajectory)` on each. No
    verification is performed — failed distillations are silently skipped.
    """
    scoped_dir = cache_dir / traj_qra_id
    cache_file = scoped_dir / f"study_root_qras_{experiment_name}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached: list[QRA] = pickle.load(f)
        logger.info(
            f"Loaded {len(cached)} study root QRAs for '{experiment_name}' "
            f"(traj_qra_id={traj_qra_id}) from cache."
        )
        return cached

    experiment = await prisma_client.experiment.find_unique(
        where={"experiment_name": experiment_name}
    )
    if experiment is None or not experiment.graph_json:
        raise ValueError(f"Experiment '{experiment_name}' has no graph_json.")

    graph = experiment.graph_json
    pseudo_root_id = "pseudo_root"
    root_task_ids: set[str] = {
        e["target"]
        for e in graph.get("edges", [])
        if e.get("type") == "decomposition" and e.get("source") == pseudo_root_id
    }
    if not root_task_ids:
        raise ValueError(
            f"Experiment '{experiment_name}' has no root tasks under pseudo_root."
        )

    trajectories = await prisma_client.trajectory.find_many(
        where={
            "experiment_name": experiment_name,
            "is_sft_candidate": True,
            "task_id": {"in": list(root_task_ids)},
        },
        order={"created_at": "desc"},
    )
    seen: set[str] = set()
    picked: list = []
    for t in trajectories:
        if t.task_id in seen:
            continue
        seen.add(t.task_id)
        picked.append(t)
    logger.info(
        f"Found {len(picked)} root-task trajectories to distill "
        f"({len(trajectories)} solved rows across {len(root_task_ids)} root tasks)."
    )

    sem = asyncio.Semaphore(distill_concurrency)

    async def _distill_one(t) -> QRA | None:
        async with sem:
            raw = t.trials_json
            if not isinstance(raw, list) or not raw:
                return None
            try:
                messages = [
                    PydanticTinkerBaseMessage.model_validate(m).to_tinker_message()
                    for m in raw
                ]
            except Exception:
                logger.exception(
                    f"Failed to deserialize trajectory for task {t.task_id}"
                )
                return None
            instruction = t.instruction or ""
            return await distiller.distill(instruction, messages)

    results = await asyncio.gather(*[_distill_one(t) for t in picked])
    qras: list[QRA] = [q for q in results if q is not None]
    logger.info(
        f"Distilled {len(qras)}/{len(picked)} study root QRAs for '{experiment_name}'."
    )

    scoped_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(qras, f)

    return qras


async def build_knowledge_suites(
    generator: GeneratorAgent,
    knowledge_list: list[Knowledge],
    k_per_knowledge: int,
    cache_id: str,
    prisma_client: Prisma,
    for_rl: bool,
    for_eval: bool,
    name_prefix: str | None = None,
    granular_id: str | None = None,
    library_name: str | None = None,
    generation_concurrency: int = 400,
) -> list[SeedSuite]:
    """One SeedSuite per knowledge item, each holding `k_per_knowledge`
    LLM-generated question tasks. Suite name = f"{name_prefix or cache_id}__{knowledge.title}"
    so per-knowledge metrics & DB rows stay separable."""
    await ensure_sft_cache(
        prisma_client,
        cache_id,
        granular_id=granular_id,
        library_name=library_name,
        description=f"build_knowledge_suites k={k_per_knowledge}",
    )
    suites_per_k = await asyncio.gather(
        *[
            generate_qras_cached(
                generator=generator,
                knowledge=k,
                count=k_per_knowledge,
                cache_id=cache_id,
                prisma_client=prisma_client,
                generation_concurrency=generation_concurrency,
            )
            for k in knowledge_list
        ]
    )
    suite_prefix = name_prefix or cache_id
    suites: list[SeedSuite] = []
    for k, qras in zip(knowledge_list, suites_per_k):
        tasks = [Task(instruction=q.question) for q in qras]
        suites.append(
            SeedSuite(
                name=f"{suite_prefix}__{k.title}",
                tasks=tasks,
                for_rl=for_rl,
                for_eval=for_eval,
            )
        )
    return suites
