import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from prisma import Json
from prisma.types import TrajectoryWhereInput
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.rl.postgres_db import PostgresDB

logger = logging.getLogger(__name__)

class RLDatabase:
    """
    Unified database access layer for RL experiments.
    Handles experiments, trajectories, and knowledge.
    Includes internal locking to ensure sequential access.
    """
    def __init__(self, experiment_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self._db = PostgresDB()
        self._lock = asyncio.Lock()

    async def connect(self):
        """Explicitly connect to the database."""
        async with self._lock:
            await self._db.connect()

    async def close(self):
        """Explicitly close the database connection."""
        async with self._lock:
            await self._db.close()

    async def _get_client(self):
        # Internal helper, assumes lock is already held if called from public methods
        return await self._db.get_client()

    # --- Experiment Management ---

    async def register_experiment(self, experiment_name: str) -> str:
        async with self._lock:
            client = await self._get_client()
            exp = await client.experiment.upsert(
                where={"experiment_name": experiment_name},
                data={
                    "create": {"experiment_name": experiment_name},
                    "update": {"experiment_name": experiment_name}
                }
            )
            self.experiment_name = exp.experiment_name
            return exp.experiment_name

    async def update_graph_json(self, graph_json: dict):
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before updating graph")
        async with self._lock:
            client = await self._get_client()
            await client.experiment.update(
                where={"experiment_name": self.experiment_name},
                data={
                    "graph_json": Json(graph_json)
                }
            )

    # --- Knowledge Management ---

    async def create_knowledge(
        self,
        knowledge_id: str,
        task_id: str,
        instruction: str,
        title: str,
        content: str
    ) -> str:
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before creating knowledge")
        async with self._lock:
            client = await self._get_client()
            knowledge = await client.knowledge.create(
                data={
                    "id": knowledge_id,
                    "experiment_name": self.experiment_name,
                    "task_id": task_id,
                    "instruction": instruction,
                    "title": title,
                    "content": content,
                }
            )
            return knowledge.id

    async def get_knowledges(self) -> List[Any]:
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before getting knowledges")
        async with self._lock:
            client = await self._get_client()
            return await client.knowledge.find_many(
                where={"experiment_name": self.experiment_name}
            )

    # --- Trajectory Management ---

    async def add_trajectory(
        self,
        task_id: str,
        instruction: str | None,
        conclusion: str | None,
        reward: float | None,
        trajectory: List[TinkerMessage],
        knowledge_ids: List[str] | None = None,
        final_knowledge: str | None = None,
        final_knowledge_title: str | None = None,
    ) -> int:
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before adding trajectory")
        async with self._lock:
            client = await self._get_client()

            # 1. Create the main Trajectory record
            new_traj = await client.trajectory.create(
                data={
                    "experiment_name": self.experiment_name,
                    "task_id": task_id,
                    "knowledge_ids": knowledge_ids or [],
                    "instruction": instruction,
                    "conclusion": conclusion,
                    "reward": reward,
                    "trials_json": Json(
                        [
                            PydanticTinkerBaseMessage.model_validate(m).model_dump(
                                mode="json", exclude_none=True
                            )
                            for m in trajectory
                        ]
                    ),
                    "final_knowledge": final_knowledge,
                    "final_knowledge_title": final_knowledge_title,
                    "is_sft_candidate": (reward is not None and reward > 0),
                }
            )

            return new_traj.id

    async def get_batch(self, batch_size: int, max_reuse: int) -> List[Dict[str, Any]]:
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before getting batch")
        async with self._lock:
            client = await self._get_client()
            # Select oldest last_used_at first
            trajectories = await client.trajectory.find_many(
                where={
                    "experiment_name": self.experiment_name,
                    "is_sft_candidate": True,
                    "usage_count": {"lt": max_reuse},
                },
                order={"last_used_at": "asc"},
                take=batch_size,
            )

            if not trajectories:
                return []

            results = []
            now = datetime.now()

            for t in trajectories:
                new_usage = t.usage_count + 1
                await client.trajectory.update(
                    where={"id": t.id}, data={"usage_count": new_usage, "last_used_at": now}
                )

                raw_json = t.trials_json
                if isinstance(raw_json, list):
                    messages = [
                        PydanticTinkerBaseMessage.model_validate(m).to_tinker_message()
                        for m in raw_json
                    ]
                else:
                    messages = []

                results.append(
                    {
                        "task_id": t.task_id,
                        "knowledge_ids": t.knowledge_ids,
                        "messages": messages,
                        "usage_count": new_usage,
                    }
                )

            # Cleanup: Delete those that reached max_reuse
            await client.trajectory.delete_many(
                where={
                    "experiment_name": self.experiment_name,
                    "usage_count": {"gte": max_reuse},
                }
            )
            return results

    async def get_count(self, max_reuse: int | None = None) -> int:
        if self.experiment_name is None:
            raise ValueError("experiment_name must be set before getting count")
        async with self._lock:
            client = await self._get_client()
            where_clause: TrajectoryWhereInput = {
                "experiment_name": self.experiment_name,
                "is_sft_candidate": True,
            }
            if max_reuse is not None:
                where_clause["usage_count"] = {"lt": max_reuse}
            count = await client.trajectory.count(where=where_clause)
            return count or 0

# For backward compatibility / easier integration
async def create_rl_database(experiment_name: Optional[str] = None) -> RLDatabase:
    db = RLDatabase(experiment_name)
    return db
