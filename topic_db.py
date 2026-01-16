import asyncio
from abc import ABC
import json
from pathlib import Path
from typing import Generic, TypeVar, List, Optional, Type

from pydantic import BaseModel, Field


class IdentifiableModel(BaseModel):
    id: str


class Topic(IdentifiableModel):
    title: str
    source_file: str
    related_apis: list[str] = Field(default_factory=list)
    description: str


class Exercise(IdentifiableModel):
    topic_id: str
    question: str
    answer: str


T = TypeVar("T", bound=IdentifiableModel)


class BaseDatabase(Generic[T], ABC):
    def __init__(self, item_type: Type[T], db_path: str):
        self.item_type = item_type
        self.db_path = Path(db_path)
        self.items: List[T] = []
        self._lock = asyncio.Lock()
        self.load()

    def load(self):
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text())
                self.items = [self.item_type(**t) for t in data]
            except Exception as e:
                print(f"Error loading database {self.db_path}: {e}")
                self.items = []

    def save(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump() for t in self.items]
        self.db_path.write_text(json.dumps(data, indent=2))

    async def add_item(self, item: T):
        async with self._lock:
            existing = next((t for t in self.items if t.id == item.id), None)
            if existing:
                self.items.remove(existing)
            self.items.append(item)
            self.save()

    async def remove_item(self, item_id: str):
        async with self._lock:
            self.items = [t for t in self.items if t.id != item_id]
            self.save()

    def get_item(self, item_id: str) -> Optional[T]:
        return next((t for t in self.items if t.id == item_id), None)


class TopicDatabase(BaseDatabase[Topic]):
    def __init__(self, db_path: str = "topics.json"):
        super().__init__(Topic, db_path)

    @property
    def topics(self):
        return self.items

    async def add_topic(self, topic: Topic):
        await self.add_item(topic)

    async def remove_topic(self, topic_id: str):
        await self.remove_item(topic_id)

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
        return self.get_item(topic_id)


class ExerciseDatabase(BaseDatabase[Exercise]):
    def __init__(self, db_path: str = "exercises.json"):
        super().__init__(Exercise, db_path)

    @property
    def exercises(self):
        return self.items

    async def add_exercise(self, exercise: Exercise):
        await self.add_item(exercise)

    async def remove_exercise(self, exercise_id: str):
        await self.remove_item(exercise_id)

    def get_exercises_by_topic(self, topic_id: str) -> list[Exercise]:
        return [e for e in self.exercises if e.topic_id == topic_id]

    def search_exercises(self, query: str) -> list[Exercise]:
        query = query.lower()
        return [
            e
            for e in self.exercises
            if query in e.question.lower() or query in e.answer.lower()
        ]
