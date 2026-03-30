from datetime import datetime
from typing import Any, Dict, List, Optional

from prisma import Json
from prisma.types import TrajectoryWhereInput
from tinker_cookbook.renderers.base import Message as TinkerMessage

from adapter_agent.data import PydanticTinkerBaseMessage
from adapter_agent.rl.postgres_db import PostgresDB, get_db


class TrajectoryDB:
    def __init__(self, experiment_id: int):
        self.experiment_id = experiment_id
        self._db: Optional[PostgresDB] = None

    async def _get_client(self):
        if self._db is None:
            self._db = await get_db()
        return await self._db.get_client()

    async def add_trajectory(
        self,
        task_id: str,
        instruction: str | None,
        conclusion: str | None,
        reward: float | None,
        trajectory: List[TinkerMessage],
        final_knowledge: str | None = None,
        final_knowledge_title: str | None = None,
        citations: List[Dict[str, Any]] | None = None,
    ) -> int:
        client = await self._get_client()

        # 1. Create the main Trajectory record
        new_traj = await client.trajectory.create(
            data={
                "experiment_id": self.experiment_id,
                "task_id": task_id,
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

        # 2. Create associated Citations if present
        if citations:
            for c in citations:
                k_id = c.get("knowledge_id")
                try:
                    k_id_int = int(k_id) if k_id is not None else None
                except (ValueError, TypeError):
                    k_id_int = None
                
                await client.citation.create(
                    data={
                        "trajectory_id": new_traj.id,
                        "knowledge_id": k_id_int,
                        "content": c.get("content"),
                        "title": c.get("title"),
                        "turn_index": c.get("turn_index", 0),
                    }
                )

        return new_traj.id

    async def get_batch(self, batch_size: int, max_reuse: int) -> List[Dict[str, Any]]:
        client = await self._get_client()
        # Select oldest last_used_at first
        trajectories = await client.trajectory.find_many(
            where={
                "experiment_id": self.experiment_id,
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
            # Messages is already a list/dict thanks to Prisma Json support
            # We restore them to proper TinkerMessage objects so that Renderers can handle tool_calls correctly.
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
                    "knowledge_id": t.knowledge_id,
                    "messages": messages,
                    "usage_count": new_usage,
                }
            )

        # Cleanup: Delete those that reached max_reuse
        await client.trajectory.delete_many(
            where={
                "experiment_id": self.experiment_id,
                "usage_count": {"gte": max_reuse},
            }
        )
        return results

    async def get_count(self, max_reuse: int | None = None) -> int:
        client = await self._get_client()
        where_clause: TrajectoryWhereInput = {
            "experiment_id": self.experiment_id,
            "is_sft_candidate": True,
        }
        if max_reuse is not None:
            where_clause["usage_count"] = {"lt": max_reuse}
        count = await client.trajectory.count(where=where_clause)
        return count or 0


async def create_trajectory_db(experiment_id: int) -> TrajectoryDB:
    db = TrajectoryDB(experiment_id)
    return db
