import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Set

from oai_utils.agent import AgentsSDKModel
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.hierarchical.agent.reflector import extract_submit_content
from adapter_agent.internalize.wiki_curator import WikiCurator
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.rl_database import RLDatabase
from adapter_agent.rl.task_net import TaskNetwork

logger = logging.getLogger(__name__)


@dataclass
class CurationRequest:
    task_id: str
    instruction: str
    trajectory: List[TinkerMessage]
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeStudier:
    """
    Decoupled pipeline for studying successful trajectories and curating
    them into the Wiki.

    Each enqueued trajectory is processed by a single WikiCurator run that
    mines the agent's verified-working answer for library knowledge and
    commits it. The Reflector is no longer used; submit extraction happens
    inline.
    """

    def __init__(
        self,
        curator_model: AgentsSDKModel,
        wiki_manager: WikiManager,
        rl_db: RLDatabase,
        runtime_settings: RuntimeSettings,
        task_network: TaskNetwork,  # Avoid circular import
        library_name: str,
        concurrency: int = 4,
        curator_qwen_no_think: bool = False,
    ):
        self.curator_model = curator_model
        self.curator_qwen_no_think = curator_qwen_no_think
        self.library_name = library_name
        self.wiki_manager = wiki_manager
        self.rl_db = rl_db
        self.runtime_settings = runtime_settings
        self.task_network = task_network
        self.concurrency = concurrency

        self.curator = WikiCurator(
            wiki_manager=self.wiki_manager,
            model=self.curator_model,
            library_name=self.library_name,
            qwen_no_think=self.curator_qwen_no_think,
        )

        self.queue: asyncio.Queue[CurationRequest] = asyncio.Queue()

        self._processed_tasks: Set[str] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the background curation worker."""
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
        trajectory: List[TinkerMessage],
    ):
        """Add a successful trajectory to the curation queue."""
        request = CurationRequest(
            task_id=task_id,
            instruction=instruction,
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

                if request.task_id in self._processed_tasks:
                    logger.info(
                        f"Skipping redundant curation for task {request.task_id}"
                    )
                    self.queue.task_done()
                    continue

                logger.info(f"Picked up task {request.task_id} for curation.")
                await self._curate(request)

                self._processed_tasks.add(request.task_id)
                self.queue.task_done()
                logger.info(f"Finished curation for task {request.task_id}.")

            except Exception as e:
                logger.exception(f"Error in KnowledgeStudier main loop: {e}")

    async def _curate(self, request: CurationRequest):
        """
        Run a single WikiCurator pass on the trajectory's task + final answer.
        """
        final_answer = extract_submit_content(request.trajectory)
        if final_answer is None:
            logger.warning(
                f"Skipping curation for task {request.task_id}: no <submit> block in trajectory."
            )
            await self.task_network.mark_knowledge_ready(request.task_id, [])
            return

        async with self.runtime_settings.build_runtime() as runtime:
            try:
                await self.curator.curate(
                    task_instruction=request.instruction,
                    final_answer=final_answer,
                    runtime=runtime,
                )
            except Exception as e:
                logger.error(
                    f"Curation failed for task {request.task_id}: {e}"
                )

        await self.task_network.mark_knowledge_ready(request.task_id, [])
        await self.rl_db.update_graph_json(self.task_network.to_dict())
        logger.info(f"Curation complete for task {request.task_id}")
