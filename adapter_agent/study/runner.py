import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tinker
from agents.extensions.models.litellm_model import LitellmModel
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.analyzer import Analyzer
from adapter_agent.hierarchical.agent.reflector import Reflector
from adapter_agent.hierarchical.process.rewire import ss_solve_verify
from adapter_agent.internalize.studier import KnowledgeStudier
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
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
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.study.qra_budget import QRABudgetConfig, QRABudgetTracker
from adapter_agent.study.qra_distiller import QRADistiller
from adapter_agent.util.exception import AllTasksCompleted

logger = logging.getLogger(__name__)


DEFAULT_SOLVER_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_RUST_LIBDIR = Path("repositories/numrs")


@dataclass
class StudyWorkerResources:
    solver_model: TinkerModel | LitellmModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    reflector: Reflector
    qra_distiller: QRADistiller
    qra_verify_executor: InternalizeExecutor
    prisma: Prisma
    qwen_no_think: bool


async def build_study_worker_resources(
    wiki_version: str,
    solver_backend: Literal["qwen_tinker", "gemini"] = "qwen_tinker",
    solver_model_name: str = DEFAULT_SOLVER_MODEL_NAME,
    solver_model_ckpt_path: str | None = None,
    rust_libdir: Path = DEFAULT_RUST_LIBDIR,
    verify_runtime: Literal["docker", "cloudrun"] = "docker",
    verify_runtime_pool_size: int = 2,
) -> StudyWorkerResources:
    if solver_backend == "qwen_tinker":
        service_client = tinker.ServiceClient()
        solver_model, _tok, _renderer = setup_tinkermodel(
            service_client=service_client,
            path=solver_model_ckpt_path,
            model_name=solver_model_name,
        )
        qwen_no_think = True
    elif solver_backend == "gemini":
        solver_model = get_gemini()
        qwen_no_think = False
    else:
        raise ValueError(f"Unknown solver_backend: {solver_backend}")

    rust_doc_analyzer = await AsyncRustDocAnalyzer.create_from_libdir(rust_libdir)

    prisma = Prisma()
    await prisma.connect()
    wiki_manager = WikiManager(prisma, version=wiki_version)

    verifier_model = get_gemini()
    reflector = Reflector(model=verifier_model)
    qra_distiller = QRADistiller(model=verifier_model)

    if verify_runtime == "docker":
        verify_runtime_settings = RuntimeSettings.docker_numrs2()
    elif verify_runtime == "cloudrun":
        verify_runtime_settings = RuntimeSettings.cloudrun_numrs2()
    else:
        raise ValueError(f"Unknown verify_runtime: {verify_runtime}")
    qra_verify_executor = InternalizeExecutor(
        runtime_pool=RuntimePool(
            verify_runtime_settings, max_size=verify_runtime_pool_size
        ),
        verifier=Verifier(model=get_gemini_lite(), rust_doc_analyzer=rust_doc_analyzer),
    )

    return StudyWorkerResources(
        solver_model=solver_model,
        verifier_model=verifier_model,
        rust_doc_analyzer=rust_doc_analyzer,
        wiki_manager=wiki_manager,
        reflector=reflector,
        qra_distiller=qra_distiller,
        qra_verify_executor=qra_verify_executor,
        prisma=prisma,
        qwen_no_think=qwen_no_think,
    )


@dataclass
class StudyWorker:
    """One async worker that pulls from the shared TaskNetwork and runs study sessions.

    When the TaskNetwork is drained (AllTasksCompleted), the worker parks on
    `tasks_available` instead of exiting, since the upstream injector may push
    additional tasks later (e.g., from the RL pipeline's failed-task queue).
    """

    task_network: TaskNetwork
    solver_model: TinkerModel | LitellmModel
    verifier_model: AgentsSDKModel
    rust_doc_analyzer: AsyncRustDocAnalyzer
    wiki_manager: WikiManager
    rl_db: RLDatabase
    studier: KnowledgeStudier
    reflector: Reflector
    qra_distiller: QRADistiller
    qra_verify_executor: InternalizeExecutor
    qra_budget: QRABudgetTracker
    qra_out_queue: asyncio.Queue[tuple[str, QRA]]
    root_task_ids: set[str]
    json_path: Path
    tasks_available: asyncio.Event
    qwen_no_think: bool = True

    async def run(self) -> None:
        while True:
            try:
                async with await TaskContext.anext_task_from_network(
                    self.task_network
                ) as current:
                    if is_study(current):
                        await self._study(current)
            except AllTasksCompleted:
                self.tasks_available.clear()
                logger.info("StudyWorker: pool empty, waiting for more tasks.")
                await self.tasks_available.wait()
            except Exception:
                logger.exception("StudyWorker: iteration failed")
                await asyncio.sleep(1)

    async def _study(
        self, current: TaskResultContext[StudyTask, StudyTaskCompleted]
    ) -> None:
        await self._sync_graph()
        task = current.task

        ret = await ss_solve_verify(
            solver_model=self.solver_model,
            verifier_model=self.verifier_model,
            rust_doc_analyzer=self.rust_doc_analyzer,
            task=task.task,
            max_turns=10,
            qwen_no_think=self.qwen_no_think,
            runtime_settings=RuntimeSettings.docker_numrs2(),
            wiki_manager=self.wiki_manager,
        )

        if isinstance(ret, RewireSessionResultNormal):
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

            if ret.reward > 0:
                is_root = task.id in self.root_task_ids
                logger.info(
                    f"Task {task.id} solved (is_root={is_root}). "
                    "Reflecting" + (" + topping up QRA quota." if is_root else " only.")
                )
                # Reflection / wiki integration runs for both root and subtask
                # solves — wiki growth helps later root attempts. QRA emission,
                # however, is gated to root tasks only: subtask QRAs would land
                # under the wrong task_id and never trigger RL-side cache replay
                # for the task RL actually keeps failing.
                coros = [self._enqueue_reflections(task, ret.trials)]
                if is_root:
                    await self.qra_budget.add_trajectory(task.id, ret.trials)
                    coros.append(self._topup_qra_quota(task))
                await asyncio.gather(*coros)

        if not task.is_generation or not isinstance(ret, RewireSessionResultFailure):
            current.register_result(task.complete(ret))
        else:
            try:
                analyzer = Analyzer(model=get_gemini())
                subtask = await analyzer.analyze_trajectory(ret.trials)
                logger.info(f"Decomposed into subtask: {subtask.instruction}")
                current.register_result(task.complete(ret, new_task=subtask))
            except Exception:
                logger.exception("Subtask generation failed")
                current.register_result(task.complete(ret, new_task=None))

        await self._sync_graph()

    async def _enqueue_reflections(self, task: StudyTask, trials) -> None:
        reflections = await self.reflector.reflect(trials)
        await self.studier.enqueue_trajectory(
            task_id=task.id,
            instruction=task.task.instruction,
            reflections=reflections,
            trajectory=trials,
        )

    async def _topup_qra_quota(self, task: StudyTask) -> None:
        """Pick trajectories from the budget tracker and distill+verify until
        the per-task quota is met or the attempt budget is exhausted.
        """
        while True:
            picked = await self.qra_budget.pick(task.id)
            if picked is None:
                return
            try:
                qra = await self.qra_distiller.distill(
                    task.task.instruction, picked.trajectory
                )
                ok = qra is not None and await self._verify_qra(
                    task.task.instruction, qra
                )
            except Exception:
                logger.exception(f"Task {task.id}: distill/verify raised; counting as failure.")
                ok = False
                qra = None

            await self.qra_budget.record_result(task.id, success=ok)
            if ok and qra is not None:
                await self.qra_out_queue.put((task.id, qra))
                snapshot = await self.qra_budget.snapshot(task.id)
                logger.info(
                    f"Task {task.id} QRA emitted "
                    f"({snapshot['emitted']}/{snapshot['quota']}, "
                    f"attempts={snapshot['total_attempts']}/{snapshot['max_attempts']})."
                )

    async def _verify_qra(self, instruction: str, qra: QRA) -> bool:
        outcome = await self.qra_verify_executor.run_execution_and_verification(
            instruction, qra.reasoning, qra.answer
        )
        return outcome.success

    async def _sync_graph(self) -> None:
        await self.task_network.save_json(self.json_path)
        await self.rl_db.update_graph_json(self.task_network.to_dict())


@dataclass
class StudyRunner:
    """Drives N StudyWorkers fed by a shared input queue.

    Tasks arriving on `in_queue` are pushed into `task_network.tasks_pool`
    so that the existing progressive-widening scheduler in TaskNetwork picks
    them up. Successful solves emit distilled QRAs onto `qra_out_queue`.
    """

    experiment_name: str
    resources: StudyWorkerResources
    task_network: TaskNetwork
    rl_db: RLDatabase
    studier: KnowledgeStudier
    in_queue: asyncio.Queue[StudyTask]
    qra_out_queue: asyncio.Queue[tuple[str, QRA]]
    qra_budget: QRABudgetTracker
    num_workers: int = 4
    launch_interval_s: float = 2.0
    json_path: Path = field(default_factory=lambda: Path("graphvis/public/data.json"))

    async def run(self) -> None:
        tasks_available = asyncio.Event()
        # Tasks injected from the input queue are "root" tasks (the ones RL
        # actually owns and re-tries). Subtasks created by the Analyzer during
        # ss_solve_verify failures are children of those roots and must NOT
        # be treated as roots — their QRAs would be tagged with the wrong
        # task_id and cache replay would never trigger.
        root_task_ids: set[str] = set()
        injector = asyncio.create_task(
            self._injector_loop(tasks_available, root_task_ids),
            name="study-injector",
        )
        workers: list[asyncio.Task] = []
        for i in range(self.num_workers):
            worker = StudyWorker(
                task_network=self.task_network,
                solver_model=self.resources.solver_model,
                verifier_model=self.resources.verifier_model,
                rust_doc_analyzer=self.resources.rust_doc_analyzer,
                wiki_manager=self.resources.wiki_manager,
                rl_db=self.rl_db,
                studier=self.studier,
                reflector=self.resources.reflector,
                qra_distiller=self.resources.qra_distiller,
                qra_verify_executor=self.resources.qra_verify_executor,
                qra_budget=self.qra_budget,
                qra_out_queue=self.qra_out_queue,
                root_task_ids=root_task_ids,
                json_path=self.json_path,
                tasks_available=tasks_available,
                qwen_no_think=self.resources.qwen_no_think,
            )
            workers.append(
                asyncio.create_task(worker.run(), name=f"study-worker-{i}")
            )
            if i < self.num_workers - 1:
                await asyncio.sleep(self.launch_interval_s)

        try:
            await asyncio.gather(injector, *workers)
        except asyncio.CancelledError:
            pass
        finally:
            injector.cancel()
            for w in workers:
                w.cancel()
            await asyncio.gather(injector, *workers, return_exceptions=True)

    async def _injector_loop(
        self, tasks_available: asyncio.Event, root_task_ids: set[str]
    ) -> None:
        while True:
            study_task = await self.in_queue.get()
            self.task_network.tasks_pool.append(study_task.task)
            root_task_ids.add(study_task.id)
            tasks_available.set()
            logger.info(
                f"StudyRunner: injected root task {study_task.id} "
                f"(pool size now {len(self.task_network.tasks_pool)})."
            )
            self.in_queue.task_done()


async def setup_study_runner(
    experiment_name: str,
    resources: StudyWorkerResources,
    in_queue: asyncio.Queue[StudyTask],
    qra_out_queue: asyncio.Queue[tuple[str, QRA]],
    num_workers: int = 4,
    json_path: Path = Path("graphvis/public/data.json"),
    qra_budget_config: QRABudgetConfig | None = None,
) -> tuple[StudyRunner, RLDatabase]:
    """Initialize shared state and return an unstarted StudyRunner.

    Unlike scripts/study.py's `setup_experiment`, this does NOT reset the
    wiki — callers are expected to reuse a pre-built wiki version across
    RL iterations.
    """
    rl_db = RLDatabase()
    await rl_db.connect()
    await rl_db.register_experiment(experiment_name)
    logger.info(f"Study experiment registered: {experiment_name}")

    task_network = TaskNetwork(tasks_pool=[])
    await rl_db.update_graph_json(task_network.to_dict())

    studier = KnowledgeStudier(
        verifier_model=resources.verifier_model,
        wiki_manager=resources.wiki_manager,
        rl_db=rl_db,
        runtime_settings=RuntimeSettings.docker_numrs2(),
        task_network=task_network,
    )
    await studier.start()

    qra_budget = QRABudgetTracker(qra_budget_config or QRABudgetConfig())

    runner = StudyRunner(
        experiment_name=experiment_name,
        resources=resources,
        task_network=task_network,
        rl_db=rl_db,
        studier=studier,
        in_queue=in_queue,
        qra_out_queue=qra_out_queue,
        qra_budget=qra_budget,
        num_workers=num_workers,
        json_path=json_path,
    )
    return runner, rl_db


async def teardown_study_runner(
    runner: StudyRunner, rl_db: RLDatabase
) -> None:
    try:
        await asyncio.wait_for(runner.studier.stop(), timeout=600)
    except asyncio.TimeoutError:
        logger.warning(
            "teardown_study_runner: studier.stop() exceeded 600s timeout; "
            "continuing with DB teardown anyway."
        )
    await runner.resources.qra_verify_executor.runtime_pool.close_all()
    await rl_db.close()
    await runner.resources.prisma.disconnect()


def _now_experiment_name(prefix: str = "study_via_rl") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
