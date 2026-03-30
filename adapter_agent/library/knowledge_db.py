import logging
from typing import TypedDict, cast

from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)

class KnowledgeDoc(TypedDict):
    id: str
    query: str
    title: str
    content: str


class KnowledgeDB:
    def __init__(self, host: str = "http://localhost:9200", index_name: str = "knowledge_box"):
        self.client = AsyncElasticsearch(hosts=[host])
        self.index_name = index_name

    @classmethod
    def for_experiment(cls, experiment_id: int, host: str = "http://localhost:9200") -> "KnowledgeDB":
        """Create a KnowledgeDB instance with an index name scoped to the experiment ID."""
        return cls(host=host, index_name=f"knowledge_box_{experiment_id}")

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
                            "title": {"type": "text", "similarity": "BM25"},
                            "content": {"type": "text", "similarity": "BM25"},
                        }
                    }
                }
            )

    async def add_knowledge(self, id: int, query: str, title: str, content: str) -> str:
        """Add a new knowledge entry using a stable Postgres ID and return the ES document ID."""
        logger.info(f"Adding knowledge to '{self.index_name}' (id: {id}, title: {title}, query: {query})")
        res = await self.client.index(
            index=self.index_name,
            id=str(id), # Use Postgres ID as the stable ES document ID
            document={
                "query": query,
                "title": title,
                "content": content,
            }
        )
        # Refresh the index to make the new document immediately searchable
        await self.client.indices.refresh(index=self.index_name)
        return res["_id"]

    async def get_knowledge_by_id(self, doc_id: str) -> KnowledgeDoc | None:
        """Retrieve a specific knowledge entry by its document ID."""
        try:
            res = await self.client.get(index=self.index_name, id=doc_id)
            doc = cast(KnowledgeDoc, res["_source"])
            doc["id"] = res["_id"]
            return doc
        except Exception as e:
            logger.error(f"Failed to get knowledge by id {doc_id}: {e}")
            return None

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
                        "fields": ["title^3", "query^2", "content"] # Give more weight to title and query field matching
                    }
                },
                "size": limit
            }
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = cast(KnowledgeDoc, hit["_source"])
            doc["id"] = hit["_id"]
            results.append(doc)
            
        return results

    async def clear(self) -> None:
        """Delete the index (useful for resetting between learning runs)."""
        if await self.client.indices.exists(index=self.index_name):
            logger.info(f"Deleting Elasticsearch index '{self.index_name}'...")
            await self.client.indices.delete(index=self.index_name)
            
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            # Mark it as closed/None if possible, though AsyncElasticsearch doesn't 
            # strictly require it if we don't reuse it.
            self.client = None # type: ignore
