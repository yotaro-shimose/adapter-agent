import asyncio
import logging
import os
from prisma import Prisma
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def clean_db():
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/adapter_agent"
    
    prisma = Prisma()
    await prisma.connect()
    
    try:
        logger.info("Cleaning database...")
        # Deleting all experiments will cascade delete trajectories and citations
        count = await prisma.experiment.delete_many()
        logger.info(f"Deleted {count} experiments and all associated data.")
        
        # Double check Trajectory table just in case (though cascade should handle it)
        t_count = await prisma.trajectory.count()
        if t_count > 0:
            logger.warning(f"Found {t_count} orphaned trajectories, cleaning up...")
            await prisma.trajectory.delete_many()
            
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
    finally:
        await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(clean_db())
