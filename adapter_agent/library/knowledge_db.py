import logging
from typing import TypedDict

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

class KnowledgeDoc(TypedDict):
    query: str
    content: str

class KnowledgeDB:
    def __init__(self, host: str = "http://localhost:9200", index_name: str = "knowledge_box"):
        self.client = AsyncElasticsearch(hosts=[host])
        self.index_name = index_name

    async def initialize(self) -> None:
        """Create the index with BM25 settings if it doesn't exist."""
        exists = await self.client.indices.exists(index=self.index_name)
        if not exists:
            logger.info(f"Creating Elasticsearch index '{self.index_name}'...")
            # BM25 is the default similarity algorithm in recent versions of Elasticsearch.
            await self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "default": {
                                    "type": "standard"
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "query": {"type": "text", "similarity": "BM25"},
                            "content": {"type": "text", "similarity": "BM25"},
                        }
                    }
                }
            )

    async def add_knowledge(self, query: str, content: str) -> str:
        """Add a new knowledge entry and return the document ID."""
        logger.info(f"Adding knowledge to '{self.index_name}' (query: {query})")
        res = await self.client.index(
            index=self.index_name,
            document={
                "query": query,
                "content": content,
            }
        )
        # Refresh the index to make the new document immediately searchable
        await self.client.indices.refresh(index=self.index_name)
        return res["_id"]
        
    async def search(self, query: str, limit: int = 5) -> list[KnowledgeDoc]:
        """Search the knowledge base using BM25 across query and content fields."""
        if not await self.client.indices.exists(index=self.index_name):
            return []

        response = await self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["query^2", "content"] # Give more weight to query field matching
                    }
                },
                "size": limit
            }
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append(hit["_source"])
            
        return results

    async def clear(self) -> None:
        """Delete the index (useful for resetting between learning runs)."""
        if await self.client.indices.exists(index=self.index_name):
            logger.info(f"Deleting Elasticsearch index '{self.index_name}'...")
            await self.client.indices.delete(index=self.index_name)
            
    async def close(self) -> None:
        await self.client.close()
