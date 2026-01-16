import json
from pathlib import Path

from pydantic import BaseModel, Field


class Topic(BaseModel):
    id: str
    title: str
    source_file: str
    related_apis: list[str] = Field(default_factory=list)
    description: str


class Exercise(BaseModel):
    id: str
    topic_id: str
    question: str
    answer: str


class TopicDatabase:
    def __init__(self, db_path: str = "topics.json"):
        self.db_path = Path(db_path)
        self.topics: list[Topic] = []
        self.load()

    def load(self):
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.topics = [Topic(**t) for t in data]
            except Exception as e:
                print(f"Error loading database: {e}")
                self.topics = []

    def save(self):
        data = [t.model_dump() for t in self.topics]
        self.db_path.write_text(json.dumps(data, indent=2))

    def add_topic(self, topic: Topic):
        # Check if ID exists? For now, just append or replace
        existing = next((t for t in self.topics if t.id == topic.id), None)
        if existing:
            self.topics.remove(existing)
        self.topics.append(topic)
        self.save()

    def remove_topic(self, topic_id: str):
        self.topics = [t for t in self.topics if t.id != topic_id]
        self.save()

    def search_topics(self, query: str) -> list[Topic]:
        query = query.lower()
        return [
            t
            for t in self.topics
            if query in t.title.lower() or query in t.description.lower()
        ]

    def get_topics_by_file(self, filename: str) -> list[Topic]:
        return [t for t in self.topics if t.source_file == filename]

    def get_topics_by_api(self, api_name: str) -> list[Topic]:
        return [t for t in self.topics if api_name in t.related_apis]

    def get_topic(self, topic_id: str) -> Topic | None:
        return next((t for t in self.topics if t.id == topic_id), None)
