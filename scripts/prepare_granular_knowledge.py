import argparse
import asyncio
import logging
from datetime import datetime

from prisma import Prisma

from adapter_agent.hierarchical.types import KnowledgeSeed
from adapter_agent.internalize.granularizer import WikiGranularizer
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.util.logger_util import setup_base_loglevel

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Prepare granular knowledge from Wiki articles")
    parser.add_argument("--version", type=str, default="lab_verification", help="Wiki version to load articles from")
    parser.add_argument("--simple-train-id", type=str, default=None, help="SimpleTrainRun ID to associate with")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency for granular knowledge generation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_base_loglevel()
    logger.setLevel(logging.INFO)

    db = Prisma()
    await db.connect()

    # 1. Setup Train Run ID
    train_id = args.simple_train_id or f"granular_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure SimpleTrainRun exists
    await db.simpletrainrun.upsert(
        where={"id": train_id},
        data={
            "create": {"id": train_id},
            "update": {}
        }
    )
    logger.info(f"Using SimpleTrainRun ID: {train_id}")

    # 2. Fetch Articles
    wiki_manager = WikiManager(db, version=args.version)
    article_titles = await wiki_manager.ls(path="api/")
    if not article_titles:
        logger.error(f"No articles found in Wiki version '{args.version}' with prefix 'api/'.")
        await db.disconnect()
        return

    logger.info(f"Found {len(article_titles)} articles to process.")

    model = get_gemini()
    granularizer = WikiGranularizer(model=model)
    runtime_settings = RuntimeSettings.docker_numrs2()

    # 3. Process each article to get seeds
    all_seeds_with_content = []
    for title in article_titles:
        content = await wiki_manager.read(title)
        if not content:
            continue
        
        seeds = await granularizer.identify_seeds(title, content)
        for seed in seeds:
            all_seeds_with_content.append((seed, title, content))

    logger.info(f"Directives identified: {len(all_seeds_with_content)} granular seeds across all articles.")

    # 4. Generate Knowledge for each seed concurrently
    sem = asyncio.Semaphore(args.concurrency)

    async def _process_seed(seed: KnowledgeSeed, source_title: str, source_content: str, runtime):
        async with sem:
            logger.info(f"Generating knowledge for seed: {seed.title}")
            knowledge = await granularizer.generate_knowledge(seed, source_title, source_content, runtime)
            if knowledge:
                # Save to GranularKnowledge table
                try:
                    await db.granularknowledge.create(
                        data={
                            "simple_train_id": train_id,
                            "title": knowledge.title,
                            "content": knowledge.content,
                        }
                    )
                    logger.info(f"Successfully saved granular knowledge: {knowledge.title}")
                except Exception as e:
                    logger.error(f"Failed to save granular knowledge '{knowledge.title}': {e}")

    async with runtime_settings.build_runtime() as runtime:
        tasks = [
            _process_seed(seed, source_title, source_content, runtime)
            for seed, source_title, source_content in all_seeds_with_content
        ]
        await asyncio.gather(*tasks)

    logger.info("Granular knowledge preparation complete.")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
