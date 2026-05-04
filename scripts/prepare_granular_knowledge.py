"""Granular knowledge preparation driver.

Named recipes; pick one via `CONFIG = ...` near the bottom — same pattern as
`run_continue_rl.py`. Each recipe pins the study experiment whose Wiki
articles to mine, the runtime image used to generate knowledge, and the
prefix of the resulting `granular_id`.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from agents import set_tracing_disabled
from prisma import Prisma

from adapter_agent.hierarchical.types import KnowledgeSeed
from adapter_agent.internalize.granularizer import WikiGranularizer
from adapter_agent.library.library_spec import LibrarySpec
from adapter_agent.library.wiki_manager import WikiManager
from adapter_agent.model_helper import get_gemini
from adapter_agent.util.logger_util import setup_base_loglevel

# Suppress Agents SDK tracing telemetry — see scripts/run_continue_rl.py for context.
set_tracing_disabled(True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepRecipe:
    """Named recipe for generating granular knowledge from a Wiki version.

    `granular_id` is derived at run time as
    `<granular_id_prefix>_<timestamp>` unless explicitly set (use the
    explicit form to resume / append to an existing granular run).

    `wiki_path_prefix` filters which articles to mine: e.g. ``"api/"``
    keeps only API reference pages, ``None`` includes everything in the
    version (api + concepts + MOC).
    """

    version: str
    granular_id_prefix: str
    concurrency: int
    library_spec: LibrarySpec
    wiki_path_prefix: str | None = "api/"
    granular_id: str | None = None


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

# numrs2 study run. `granular_prep` prefix matches the IDs that downstream
# pipelines (e.g. run_continue_rl.py's GRANULAR_ID lineage) consume.
NUMRS2_STUDY_PREP = PrepRecipe(
    version="study_20260418_233708",
    granular_id_prefix="granular_prep",
    concurrency=50,
    library_spec=LibrarySpec.numrs2(),
)

# Hisab study run (study.py succeeded with LibrarySpec.hisab() at this
# experiment id). Granular IDs are prefixed `granular_prep_hisab` to keep
# them clearly separated from numrs2 lineage. wiki_path_prefix=None to
# include `concepts/` articles in addition to `api/`.
HISAB_STUDY_PREP = PrepRecipe(
    version="study_20260504_070444",
    granular_id_prefix="granular_prep_hisab",
    concurrency=50,
    library_spec=LibrarySpec.hisab(),
    wiki_path_prefix=None,
)


CONFIG = HISAB_STUDY_PREP


# ---------------------------------------------------------------------------


async def main() -> None:
    cfg = CONFIG

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    setup_base_loglevel()
    logger.setLevel(logging.INFO)

    db = Prisma()
    await db.connect()

    granular_id = cfg.granular_id or (
        f"{cfg.granular_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    await db.simpletrainrun.upsert(
        where={"id": granular_id},
        data={"create": {"id": granular_id}, "update": {}},
    )
    logger.info(f"Recipe: version={cfg.version}, library={cfg.library_spec.name}")
    logger.info(f"Using granular ID: {granular_id}")

    wiki_manager = WikiManager(db, version=cfg.version)
    article_titles = await wiki_manager.ls(path=cfg.wiki_path_prefix)
    if not article_titles:
        prefix_repr = repr(cfg.wiki_path_prefix) if cfg.wiki_path_prefix else "<no filter>"
        logger.error(
            f"No articles found in Wiki version '{cfg.version}' (prefix={prefix_repr})."
        )
        await db.disconnect()
        return

    logger.info(f"Found {len(article_titles)} articles to process.")

    model = get_gemini()
    granularizer = WikiGranularizer(model=model)
    runtime_settings = cfg.library_spec.docker_runtime()

    all_seeds_with_content = []
    for title in article_titles:
        content = await wiki_manager.read(title)
        if not content:
            continue

        seeds = await granularizer.identify_seeds(title, content)
        for seed in seeds:
            all_seeds_with_content.append((seed, title, content))

    logger.info(
        f"Directives identified: {len(all_seeds_with_content)} granular seeds across all articles."
    )

    sem = asyncio.Semaphore(cfg.concurrency)

    async def _process_seed(
        seed: KnowledgeSeed, source_title: str, source_content: str, runtime
    ):
        async with sem:
            logger.info(f"Generating knowledge for seed: {seed.title}")
            knowledge = await granularizer.generate_knowledge(
                seed, source_title, source_content, runtime
            )
            if knowledge:
                try:
                    await db.granularknowledge.create(
                        data={
                            "simple_train_id": granular_id,
                            "title": knowledge.title,
                            "content": knowledge.content,
                        }
                    )
                    logger.info(
                        f"Successfully saved granular knowledge: {knowledge.title}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to save granular knowledge '{knowledge.title}': {e}"
                    )

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
