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
) -> SeedSuite:
    """Pick a contiguous window out of load_gh_archive() as a SeedSuite."""
    tasks: list[Task] = load_gh_archive()[task_slice]
    return SeedSuite(name=name, tasks=tasks, for_rl=for_rl, for_eval=for_eval)


async def generate_qras_cached(
    generator: GeneratorAgent,
    knowledge: Knowledge,
    count: int,
    prefix: str,
    cache_dir: Path,
    generation_concurrency: int,
    is_coding: bool = True,
) -> list[QRA]:
    """Generate `count` QRAs for one knowledge item, caching the result on disk.

    The semaphore caps in-flight LLM calls. The QRA generator is allowed to
    return None (model failure); we retry until a valid QRA comes back.
    """
    cache_file = cache_dir / f"{knowledge.id}_{prefix}_{count}.pkl"
    if cache_file.exists():
        logger.info(
            f"Loading {count} {prefix} QRAs for '{knowledge.title}' from cache."
        )
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info(f"Generating {count} {prefix} QRAs for '{knowledge.title}'...")
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

    results: list[QRA] = await asyncio.gather(*[_gen() for _ in range(count)])

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)

    return results


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
    cache_dir: Path,
    name_prefix: str,
    for_rl: bool,
    for_eval: bool,
    generation_concurrency: int = 400,
) -> list[SeedSuite]:
    """One SeedSuite per knowledge item, each holding `k_per_knowledge`
    LLM-generated question tasks. Suite name = f"{name_prefix}__{knowledge.title}"
    so per-knowledge metrics & DB rows stay separable."""
    suites_per_k = await asyncio.gather(
        *[
            generate_qras_cached(
                generator=generator,
                knowledge=k,
                count=k_per_knowledge,
                prefix=name_prefix,
                cache_dir=cache_dir,
                generation_concurrency=generation_concurrency,
            )
            for k in knowledge_list
        ]
    )
    suites: list[SeedSuite] = []
    for k, qras in zip(knowledge_list, suites_per_k):
        tasks = [Task(instruction=q.question) for q in qras]
        suites.append(
            SeedSuite(
                name=f"{name_prefix}__{k.title}",
                tasks=tasks,
                for_rl=for_rl,
                for_eval=for_eval,
            )
        )
    return suites
