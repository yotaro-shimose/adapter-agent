import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.gh import load_gh_archive
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.internalize.studier import KnowledgeStudier
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.knowledge_db import KnowledgeDB
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.env.session_result import (
    RewireSessionResultFailure,
    RewireSessionResultNormal,
)
from adapter_agent.rl.rl_database import RLDatabase
from adapter_agent.rl.task_net import (
    StudyTask,
    StudyTaskCompleted,
    TaskContext,
    TaskNetwork,
    TaskResultContext,
    is_study,
)
from adapter_agent.util.exception import AllTasksCompleted
from adapter_agent.util.logger_util import setup_base_loglevel

DEFAULT_RUST_LIBDIR = Path("repositories/numrs")

Backend = Literal["tinker", "gemini"]
SolverKind = Backend  # legacy alias


@dataclass
class WorkerResources:
    """Per-worker resources that can be constructed independently in each process."""

    solver_model: TinkerModel | LitellmModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    analyzer: Analyzer
    prisma: Prisma
    library_name: str
    qwen_no_think: bool
    verifier_qwen_no_think: bool


def build_solver_model(
    solver_kind: SolverKind,
    solver_model_name: str,
    solver_model_ckpt_path: str | None = None,
) -> TinkerModel | LitellmModel:
    if solver_kind == "tinker":
        service_client = tinker.ServiceClient()
        solver_model, _tokenizer, _renderer = setup_tinkermodel(
            service_client=service_client,
            path=solver_model_ckpt_path,
            model_name=solver_model_name,
        )
        return solver_model
    if solver_kind == "gemini":
        return get_gemini()
    raise ValueError(f"Unknown solver_kind: {solver_kind}")


def build_support_model(backend: Backend) -> AgentsSDKModel:
    """Build the shared model used by Curator / Analyzer / Verifier."""
    if backend == "tinker":
        service_client = tinker.ServiceClient()
        model, _tok, _renderer = setup_tinkermodel(
            service_client=service_client,
            model_name="Qwen/Qwen3-32B",
        )
        return model
    if backend == "gemini":
        return get_gemini()
    raise ValueError(f"Unknown backend: {backend}")


async def build_worker_resources(
    wiki_version: str,
    solver_kind: SolverKind,
    solver_model_name: str,
    analyzer_model: AgentsSDKModel,
    verifier_model: AgentsSDKModel,
    library_name: str,
    solver_model_ckpt_path: str | None = None,
    rust_libdir: Path = DEFAULT_RUST_LIBDIR,
    analyzer_qwen_no_think: bool = False,
    verifier_qwen_no_think: bool = False,
) -> WorkerResources:
    """Construct per-process resources that StudyActor depends on.

    Each Ray worker process should call this on startup. DB connections,
    analyzers, and model clients are not picklable, so they must be built
    in the process that uses them.
    """
    solver_model = build_solver_model(
        solver_kind=solver_kind,
        solver_model_name=solver_model_name,
        solver_model_ckpt_path=solver_model_ckpt_path,
    )
    qwen_no_think = solver_kind == "tinker"

    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(rust_libdir)

    prisma = Prisma()
    await prisma.connect()
    wiki_manager = WikiManager(prisma, version=wiki_version)

    analyzer = Analyzer(model=analyzer_model, qwen_no_think=analyzer_qwen_no_think)

    return WorkerResources(
        solver_model=solver_model,
        verifier_model=verifier_model,
        rust_doc_analyzer=rust_doc_analyzer,
        wiki_manager=wiki_manager,
        analyzer=analyzer,
        prisma=prisma,
        library_name=library_name,
        qwen_no_think=qwen_no_think,
        verifier_qwen_no_think=verifier_qwen_no_think,
    )


def build_study_actor(
    resources: WorkerResources,
    task_network: TaskNetwork,
    rl_db: RLDatabase,
    studier: KnowledgeStudier,
    json_path: Path,
) -> "StudyActor":
    """Assemble a StudyActor from per-process resources and shared handles.

    `task_network`, `rl_db`, `studier` may be the real objects (single-process
    case) or proxies that forward calls to a remote actor (multi-process case).
    They are duck-typed, so any object exposing the same async methods works.
    """
    return StudyActor(
        task_network=task_network,
        solver_model=resources.solver_model,
        verifier_model=resources.verifier_model,
        rust_doc_analyzer=resources.rust_doc_analyzer,
        wiki_manager=resources.wiki_manager,
        rl_db=rl_db,
        studier=studier,
        json_path=json_path,
        analyzer=resources.analyzer,
        library_name=resources.library_name,
        qwen_no_think=resources.qwen_no_think,
        verifier_qwen_no_think=resources.verifier_qwen_no_think,
    )


@dataclass
class StudyActor:
    task_network: TaskNetwork
    solver_model: TinkerModel | LitellmModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    rl_db: RLDatabase
    studier: KnowledgeStudier
    json_path: Path
    analyzer: Analyzer
    library_name: str
    qwen_no_think: bool
    verifier_qwen_no_think: bool

    async def run(self):
        while True:
            try:
                async with await TaskContext.anext_task_from_network(
                    self.task_network
                ) as current:
                    if is_study(current):
                        await self.study(current)
                    else:
                        # SlicingTask is now disabled in TaskNetwork,
                        # but we keep this as a no-op just in case.
                        pass
            except AllTasksCompleted:
                print("All tasks completed. Worker exiting.")
                break

    async def study(self, current: TaskResultContext[StudyTask, StudyTaskCompleted]):
        await self._sync_graph()
        task = current.task
        solved_subtasks = self.task_network.get_solved_subtasks(task.id)

        ret = await ss_solve_verify(
            solver_model=self.solver_model,
            verifier_model=self.verifier_model,
            rust_doc_analyzer=self.rust_doc_analyzer,
            task=task.task,
            max_turns=10,
            qwen_no_think=self.qwen_no_think,
            verifier_qwen_no_think=self.verifier_qwen_no_think,
            library_name=self.library_name,
            runtime_settings=RuntimeSettings.docker_numrs2(),
            wiki_manager=self.wiki_manager,
            solved_subtasks=solved_subtasks,
        )

        if isinstance(ret, RewireSessionResultNormal):
            print("Session completed with conclusion:", ret.conclusion)

            # Save trajectory immediately
            # Note: knowledge_ids is currently empty, will be updated by Studier later
            await self.rl_db.add_trajectory(
                task_id=task.id,
                instruction=task.task.instruction,
                conclusion=ret.conclusion,
                reward=ret.reward,
                trajectory=ret.trials,
                knowledge_ids=[],
                final_knowledge=None,
                final_knowledge_title=None,
            )

            # Enqueue for curation if successful; Studier handles uniqueness and the WikiCurator run.
            if ret.reward > 0:
                await self.studier.enqueue_trajectory(
                    task_id=task.id,
                    instruction=task.task.instruction,
                    trajectory=ret.trials,
                )

        # Mark task as completed IMMEDIATELY in TaskNetwork
        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
        else:
            try:
                subtask = await self.analyzer.analyze_trajectory(ret.trials)
                print(f"New Task: {subtask.instruction}")
                current.register_result(task.complete(ret, new_task=subtask))
            except Exception:
                current.register_result(task.complete(ret, new_task=None))
                logging.exception("Subtask generation failed")

        await self._sync_graph()
        return

    async def _sync_graph(self):
        """Persist the task graph to disk and to the RL database.

        Abstracted so that proxy implementations backed by a remote coordinator
        can fuse these two calls (they both read task_network state, so sending
        the state twice over RPC is wasteful). See RemoteTaskNetwork in
        study_ray.py for the fused variant.
        """
        fused = getattr(self.task_network, "sync_graph", None)
        if fused is not None:
            await fused(self.json_path)
        else:
            await self.task_network.save_json(self.json_path)
            await self.rl_db.update_graph_json(self.task_network.to_dict())


@dataclass
class ExperimentContext:
    """Shared, single-process state for a study experiment."""

    experiment_name: str
    task_network: TaskNetwork
    rl_db: RLDatabase
    knowledge_db: KnowledgeDB
    studier: KnowledgeStudier
    wiki_manager: WikiManager
    prisma: Prisma
    json_path: Path


async def setup_experiment(
    experiment_name: str,
    curator_model: AgentsSDKModel,
    library_name: str,
    reset: bool = True,
    num_tasks: slice = slice(0, 50),
    json_path: Path = Path("graphvis/public/data.json"),
    curator_qwen_no_think: bool = False,
) -> ExperimentContext:
    """Initialize the shared state that a study run operates over.

    This includes the RLDatabase, TaskNetwork, WikiManager, KnowledgeStudier,
    and KnowledgeDB. The returned context is intended to live in a single
    coordinator process. Workers should interact with it either directly
    (single-process) or via remote proxies (multi-process).
    """
    rl_db = RLDatabase()
    await rl_db.connect()
    await rl_db.register_experiment(experiment_name)
    print(f"Experiment started: {experiment_name}")

    prisma = Prisma()
    await prisma.connect()
    wiki_manager = WikiManager(prisma, version=experiment_name)

    if reset:
        await wiki_manager.reset()

    knowledge_db = KnowledgeDB.for_experiment(experiment_name)
    await knowledge_db.initialize()

    gh_tasks = load_gh_archive()
    task_network = TaskNetwork(tasks_pool=gh_tasks[num_tasks])

    await rl_db.update_graph_json(task_network.to_dict())

    studier = KnowledgeStudier(
        curator_model=curator_model,
        curator_qwen_no_think=curator_qwen_no_think,
        library_name=library_name,
        wiki_manager=wiki_manager,
        rl_db=rl_db,
        runtime_settings=RuntimeSettings.docker_numrs2(),
        task_network=task_network,
    )
    await studier.start()

    return ExperimentContext(
        experiment_name=experiment_name,
        task_network=task_network,
        rl_db=rl_db,
        knowledge_db=knowledge_db,
        studier=studier,
        wiki_manager=wiki_manager,
        prisma=prisma,
        json_path=json_path,
    )


async def teardown_experiment(ctx: ExperimentContext):
    await asyncio.wait_for(ctx.studier.stop(), timeout=300)
    await ctx.rl_db.close()
    await ctx.knowledge_db.close()
    await ctx.prisma.disconnect()


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """End-to-end study experiment configuration. All knobs in one place."""

    backend: Backend
    library_name: str

    # Solver-specific (relevant when backend == "tinker"; ignored for gemini).
    solver_model_name: str = "Qwen/Qwen3-32B"
    solver_model_ckpt_path: str | None = None

    # Worker pool.
    num_workers: int = 100
    launch_interval_s: float = 2.0

    # Task selection from the gh archive.
    num_tasks: slice = slice(0, 50)

    # Whether to wipe the wiki version at the start of the experiment.
    wiki_reset: bool = True

    # Experiment name gets timestamped at run time as `{prefix}_{timestamp}`.
    experiment_name_prefix: str = "study"


# All-Qwen run: Solver, Curator, Analyzer, Verifier all served via Tinker.
ALL_TINKER = StudyConfig(
    backend="tinker",
    library_name="numrs2",
)


# All-Gemini run: every agent uses Gemini through LiteLLM. Useful as a
# reference / sanity-check baseline.
ALL_GEMINI = StudyConfig(
    backend="gemini",
    library_name="numrs2",
)


CONFIG: StudyConfig = ALL_TINKER  # ← pick recipe here


# ---------------------------------------------------------------------------


async def main():
    setup_base_loglevel()
    cfg = CONFIG

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{cfg.experiment_name_prefix}_{timestamp}"

    # Shared support model reused by Curator, Analyzer, and Verifier.
    support_model = build_support_model(cfg.backend)
    qwen_no_think = cfg.backend == "tinker"

    ctx = await setup_experiment(
        experiment_name,
        reset=cfg.wiki_reset,
        num_tasks=cfg.num_tasks,
        curator_model=support_model,
        curator_qwen_no_think=qwen_no_think,
        library_name=cfg.library_name,
    )
    resources = await build_worker_resources(
        wiki_version=experiment_name,
        solver_kind=cfg.backend,
        solver_model_name=cfg.solver_model_name,
        solver_model_ckpt_path=cfg.solver_model_ckpt_path,
        analyzer_model=support_model,
        analyzer_qwen_no_think=qwen_no_think,
        verifier_model=support_model,
        verifier_qwen_no_think=qwen_no_think,
        library_name=cfg.library_name,
    )

    worker_tasks = []
    for i in range(cfg.num_workers):
        print(f"Launching worker {i + 1}/{cfg.num_workers}...")
        study_actor = build_study_actor(
            resources=resources,
            task_network=ctx.task_network,
            rl_db=ctx.rl_db,
            studier=ctx.studier,
            json_path=ctx.json_path,
        )
        worker_tasks.append(asyncio.create_task(study_actor.run()))
        if i < cfg.num_workers - 1:
            await asyncio.sleep(cfg.launch_interval_s)

    try:
        await asyncio.gather(*worker_tasks)
    finally:
        await teardown_experiment(ctx)
        await resources.prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
