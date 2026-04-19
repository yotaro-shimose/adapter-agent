import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma

from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.reflector import Reflector
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


DEFAULT_SOLVER_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_RUST_LIBDIR = Path("repositories/numrs")


@dataclass
class WorkerResources:
    """Per-worker resources that can be constructed independently in each process."""

    solver_model: TinkerModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    reflector: Reflector
    prisma: Prisma


async def build_worker_resources(
    wiki_version: str,
    solver_model_name: str = DEFAULT_SOLVER_MODEL_NAME,
    solver_model_ckpt_path: str | None = None,
    rust_libdir: Path = DEFAULT_RUST_LIBDIR,
) -> WorkerResources:
    """Construct per-process resources that StudyActor depends on.

    Each Ray worker process should call this on startup. DB connections,
    analyzers, and model clients are not picklable, so they must be built
    in the process that uses them.
    """
    service_client = tinker.ServiceClient()
    solver_model, _tokenizer, _renderer = setup_tinkermodel(
        service_client=service_client,
        path=solver_model_ckpt_path,
        model_name=solver_model_name,
    )

    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(rust_libdir)

    prisma = Prisma()
    await prisma.connect()
    wiki_manager = WikiManager(prisma, version=wiki_version)

    verifier_model = get_gemini()
    reflector = Reflector(model=verifier_model)

    return WorkerResources(
        solver_model=solver_model,
        verifier_model=verifier_model,
        rust_doc_analyzer=rust_doc_analyzer,
        wiki_manager=wiki_manager,
        reflector=reflector,
        prisma=prisma,
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
        reflector=resources.reflector,
    )


@dataclass
class StudyActor:
    task_network: TaskNetwork
    solver_model: TinkerModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    rl_db: RLDatabase
    studier: KnowledgeStudier
    json_path: Path
    reflector: Reflector

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

        ret = await ss_solve_verify(
            solver_model=self.solver_model,
            verifier_model=self.verifier_model,
            rust_doc_analyzer=self.rust_doc_analyzer,
            task=task.task,
            max_turns=10,
            qwen_no_think=True,
            runtime_settings=RuntimeSettings.docker_numrs2(),
            wiki_manager=self.wiki_manager,
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

            # Enqueue for distillation if successful (and let Studier handle uniqueness & formalization)
            if ret.reward > 0:
                print(f"Generating reflections for task {task.id}...")
                reflections = await self.reflector.reflect(ret.trials)
                await self.studier.enqueue_trajectory(
                    task_id=task.id,
                    instruction=task.task.instruction,
                    reflections=reflections,
                    trajectory=ret.trials,
                )

        # Mark task as completed IMMEDIATELY in TaskNetwork
        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
        else:
            try:
                analyzer = Analyzer(model=get_gemini())
                subtask = await analyzer.analyze_trajectory(ret.trials)
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
    reset: bool = True,
    num_tasks: int = 40,
    json_path: Path = Path("graphvis/public/data.json"),
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

    verifier_model = get_gemini()
    gh_tasks = load_gh_archive()
    task_network = TaskNetwork(tasks_pool=gh_tasks[:num_tasks])

    await rl_db.update_graph_json(task_network.to_dict())

    studier = KnowledgeStudier(
        verifier_model=verifier_model,
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


async def main():
    reset = True
    num_workers = 40
    launch_interval = 2
    setup_base_loglevel()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"study_{timestamp}"
    solver_model_ckpt_path = None
    # solver_model_ckpt_path = "tinker://77d12766-fb79-5995-ab03-108a8de53af1:train:0/sampler_weights/rl_0020"

    ctx = await setup_experiment(experiment_name, reset=reset)
    resources = await build_worker_resources(wiki_version=experiment_name, solver_model_ckpt_path=solver_model_ckpt_path)

    worker_tasks = []
    for i in range(num_workers):
        print(f"Launching worker {i + 1}/{num_workers}...")
        study_actor = build_study_actor(
            resources=resources,
            task_network=ctx.task_network,
            rl_db=ctx.rl_db,
            studier=ctx.studier,
            json_path=ctx.json_path,
        )
        worker_tasks.append(asyncio.create_task(study_actor.run()))
        if i < num_workers - 1:
            await asyncio.sleep(launch_interval)

    try:
        await asyncio.gather(*worker_tasks)
    finally:
        await teardown_experiment(ctx)
        await resources.prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
