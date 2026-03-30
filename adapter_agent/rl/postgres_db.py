import asyncio
import logging
import os
from typing import Optional

from prisma import Prisma, Json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Prisma uses DATABASE_URL from .env or os.environ
if not os.getenv("DATABASE_URL"):
    # Fallback to default if not set
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/adapter_agent"

class PostgresDB:
    _instance: Optional['PostgresDB'] = None
    _prisma: Optional[Prisma] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostgresDB, cls).__new__(cls)
        return cls._instance

    async def connect(self):
        if self._prisma is None:
            logger.info("Connecting to PostgreSQL via Prisma...")
            self._prisma = Prisma(auto_register=True)
            
            # Retry logic for Docker startup
            for i in range(10):
                try:
                    await self._prisma.connect()
                    break
                except Exception as e:
                    if i == 9:
                        raise e
                    logger.warning(f"Failed to connect to PG via Prisma (attempt {i+1}/10), retrying in 2s...")
                    await asyncio.sleep(2)

    async def register_experiment(self, experiment_name: str) -> int:
        client = await self.get_client()
        # Upsert: Try to create, update if exists
        exp = await client.experiment.upsert(
            where={"experiment_name": experiment_name},
            data={
                "create": {"experiment_name": experiment_name},
                "update": {"experiment_name": experiment_name}
            }
        )
        return exp.id

    async def update_graph_json(self, experiment_id: int, graph_json: dict):
        client = await self.get_client()
        await client.experiment.update(
            where={"id": experiment_id},
            data={
                "graph_json": Json(graph_json)
            }
        )

    async def create_knowledge(
        self,
        experiment_id: int,
        task_id: str,
        instruction: str,
        title: str,
        content: str
    ) -> int:
        client = await self.get_client()
        knowledge = await client.knowledge.create(
            data={
                "experiment_id": experiment_id,
                "task_id": task_id,
                "instruction": instruction,
                "title": title,
                "content": content,
            }
        )
        return knowledge.id

    async def get_knowledges(self, experiment_id: int):
        client = await self.get_client()
        return await client.knowledge.find_many(
            where={"experiment_id": experiment_id}
        )

    async def get_client(self) -> Prisma:
        if self._prisma is None or not self._prisma.is_connected():
            await self.connect()
        assert self._prisma is not None
        return self._prisma

    async def close(self):
        if self._prisma and self._prisma.is_connected():
            await self._prisma.disconnect()
            self._prisma = None

# Global helper to get the DB instance
async def get_db() -> PostgresDB:
    db = PostgresDB()
    await db.connect()
    return db
