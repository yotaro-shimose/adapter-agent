import asyncio
import logging
import os
import random
from typing import Optional

from prisma import Prisma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Prisma uses DATABASE_URL from .env or os.environ
if not os.getenv("DATABASE_URL"):
    # Fallback to default if not set
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/adapter_agent"

class PostgresDB:
    """
    Low-level connection manager for PostgreSQL via Prisma.
    This class is NOT a singleton and should be managed by a higher-level 
    database class (like RLDatabase) to ensure single-instance access 
    within a process.
    """
    def __init__(self):
        self._prisma: Optional[Prisma] = None

    async def connect(self):
        if self._prisma is None:
            logger.info("Connecting to PostgreSQL via Prisma...")
            self._prisma = Prisma(auto_register=False)
            
            # Retry logic for Docker startup
            for i in range(10):
                try:
                    await self._prisma.connect()
                    break
                except Exception as e:
                    if i == 9:
                        raise e
                    # Exponential backoff with jitter for Docker/network startup
                    wait_time = min(2**i + random.uniform(0, 1), 16)
                    logger.warning(f"Failed to connect to PG via Prisma (attempt {i+1}/10), retrying in {wait_time:.2f}s... Error: {e}")
                    await asyncio.sleep(wait_time)

    async def get_client(self) -> Prisma:
        if self._prisma is None or not self._prisma.is_connected():
            await self.connect()
        assert self._prisma is not None
        return self._prisma

    async def close(self):
        if self._prisma and self._prisma.is_connected():
            await self._prisma.disconnect()
            self._prisma = None

# Deprecated: Use RLDatabase inside UniRLState instead
async def get_db() -> PostgresDB:
    db = PostgresDB()
    await db.connect()
    return db
