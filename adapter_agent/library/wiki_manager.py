import logging
from typing import Any, List, Optional, cast
from prisma import Prisma

logger = logging.getLogger(__name__)

class WikiManager:
    """
    Manages Wiki articles in a versioned PostgreSQL database via Prisma.
    Provides atomic updates and isolated version sandboxing.
    """
    def __init__(self, db: Prisma, version: str = "v1"):
        self.db = db
        self.version = version

    async def ls(self, path: Optional[str] = None) -> List[str]:
        """Lists article titles, optionally filtered by path prefix."""
        try:
            where: dict[str, Any] = {"version": self.version}
            if path:
                where["title"] = {"startswith": path}

            articles = await self.db.wikiarticle.find_many(
                where=cast(Any, where),
                order={"title": "asc"}
            )
            return [a.title for a in articles]
        except Exception as e:
            logger.error(f"Failed to list articles for version {self.version} (path: {path}): {e}")
            return []

    async def read(self, title: str) -> Optional[str]:
        """Fetches the content of an article by title."""
        try:
            article = await self.db.wikiarticle.find_unique(
                where={"version_title": {"version": self.version, "title": title}}
            )
            return article.content if article else None
        except Exception as e:
            logger.error(f"Failed to read article '{title}' in version {self.version}: {e}")
            return None

    async def write(self, title: str, content: str):
        """Creates or overwrites an article."""
        try:
            await self.db.wikiarticle.upsert(
                where={"version_title": {"version": self.version, "title": title}},
                data={
                    "create": {"version": self.version, "title": title, "content": content},
                    "update": {"content": content}
                }
            )
        except Exception as e:
            logger.error(f"Failed to write article '{title}' in version {self.version}: {e}")
            raise

    async def str_replace(self, title: str, old_str: str, new_str: str) -> bool:
        """
        Atomically replaces text in an article.
        Returns True if the update was successful (rows affected > 0).
        """
        try:
            # Atomic SQL REPLACE scoping by version and title
            # Uses standard PostgreSQL REPLACE(string, from, to)
            count = await self.db.execute_raw(
                'UPDATE "wiki_articles" SET content = REPLACE(content, $1, $2), updated_at = NOW() '
                'WHERE version = $3 AND title = $4 AND content LIKE $5',
                old_str, new_str, self.version, title, f'%{old_str}%'
            )
            return count > 0
        except Exception as e:
            logger.error(f"Atomic replacement failed for '{title}': {e}")
            return False

    async def search(self, query: str, limit: int = 5) -> List[dict]:
        """Simple keyword search across titles and content in the current version."""
        try:
            # Case-insensitive partial match search
            articles = await self.db.wikiarticle.find_many(
                where={
                    "version": self.version,
                    "OR": [
                        {"title": {"contains": query, "mode": "insensitive"}},
                        {"content": {"contains": query, "mode": "insensitive"}}
                    ]
                },
                take=limit
            )
            return [{"title": a.title, "content": a.content} for a in articles]
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def reset(self):
        """Deletes all articles in the current version."""
        try:
            count = await self.db.wikiarticle.delete_many(where={"version": self.version})
            logger.info(f"Reset version '{self.version}': deleted {count} articles.")
        except Exception as e:
            logger.error(f"Failed to reset version {self.version}: {e}")
            raise
