import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

from oai_utils.agent import AgentsSDKModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.reflector import Reflection
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.rl_database import RLDatabase
from adapter_agent.rl.task_net import TaskNetwork

from adapter_agent.internalize.wiki_integrator import WikiIntegrator

logger = logging.getLogger(__name__)


@dataclass
class DistillationRequest:
    task_id: str
    instruction: str
    reflections: List[Reflection]
    trajectory: List[TinkerMessage]
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeStudier:
    """
    Decoupled pipeline for studying successful trajectories and extracting knowledge.

    1. Receives successful trajectories.
    2. Sequentially checks reflections for uniqueness to avoid race conditions.
    3. Concurrently formalizes unique reflections (Cargo PoC + Markdown).
    4. Marks knowledge as 'Ready' in TaskNetwork.
    """

    def __init__(
        self,
        verifier_model: AgentsSDKModel,
        wiki_manager: WikiManager,
        rl_db: RLDatabase,
        runtime_settings: RuntimeSettings,
        task_network: TaskNetwork,  # Avoid circular import
        concurrency: int = 4,
    ):
        self.verifier_model = verifier_model
        self.wiki_manager = wiki_manager
        self.rl_db = rl_db
        self.runtime_settings = runtime_settings
        self.task_network = task_network
        self.concurrency = concurrency

        self.queue: asyncio.Queue[DistillationRequest] = asyncio.Queue()

        self._processed_tasks: Set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the background distillation worker."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._main_loop())
        logger.info("KnowledgeStudier pipeline started.")

    async def stop(self):
        """Gracefully stop the pipeline after draining the queue."""
        logger.info("Stopping KnowledgeStudier pipeline...")
        self._stop_event.set()
        if self._worker_task:
            await self._worker_task
        logger.info("KnowledgeStudier pipeline stopped.")

    async def enqueue_trajectory(
        self,
        task_id: str,
        instruction: str,
        reflections: List[Reflection],
        trajectory: List[TinkerMessage],
    ):
        """Add a successful trajectory to the distillation queue."""
        request = DistillationRequest(
            task_id=task_id,
            instruction=instruction,
            reflections=reflections,
            trajectory=trajectory,
        )
        await self.queue.put(request)

    async def _main_loop(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                try:
                    request = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Step 1: Deduplication (Task Layer)
                if request.task_id in self._processed_tasks:
                    logger.info(
                        f"Skipping redundant knowledge generation for task {request.task_id}"
                    )
                    self.queue.task_done()
                    continue

                # Step 2: Skip manual uniqueness check, lean on the Integrator Agent
                if not request.reflections:
                    logger.info(f"No reflections for task {request.task_id}")
                    await self.task_network.mark_knowledge_ready(request.task_id, [])
                    self._processed_tasks.add(request.task_id)
                    self.queue.task_done()
                    continue

                # Step 3: Formalize via WikiIntegrator (Globally Sequential)
                logger.info(f"Picked up task {request.task_id} for distillation.")
                await self._formalize_and_finalize(request, request.reflections)
                
                self._processed_tasks.add(request.task_id)
                self.queue.task_done()
                logger.info(f"Finished distillation for task {request.task_id}.")

            except Exception as e:
                logger.exception(f"Error in KnowledgeStudier main loop: {e}")

    async def _formalize_and_finalize(
        self, request: DistillationRequest, reflections: List[Reflection]
    ):
        """
        Spawns an autonomous Integrator Agent for each reflection to audit and merge into the Wiki.
        """

        logger.info(
            f"Starting synthetic integration for {len(reflections)} reflections (task: {request.task_id})"
        )

        integrator = WikiIntegrator(
            wiki_manager=self.wiki_manager, model=self.verifier_model
        )

        async with self.runtime_settings.build_runtime() as runtime:
            # Process each reflection sequentially (or with limited concurrency) to avoid Wiki conflicts
            for i, reflection in enumerate(reflections):
                try:
                    logger.info(
                        f"Integrating reflection {i + 1}/{len(reflections)} for {request.task_id}..."
                    )
                    await integrator.integrate(reflection, runtime=runtime)
                except Exception as e:
                    logger.error(
                        f"Failed to integrate reflection {i} for task {request.task_id}: {e}"
                    )

        # Notify TaskNetwork that knowledge is ready
        await self.task_network.mark_knowledge_ready(request.task_id, [])
        await self.rl_db.update_graph_json(self.task_network.to_dict())
        logger.info(f"Knowledge integration complete for task {request.task_id}")

