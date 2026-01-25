import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA


class TaskPool(BaseModel):
    tasks: dict[str, Task]

    def save(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    def register(self, task: Task) -> None:
        # Avoid circular dependency or simply print?
        # Original had print statements.
        print(f"Details: Registering task: {task.instruction}")
        self.tasks[task.id] = task

    def get(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def delete(self, task_id: str) -> None:
        if task_id in self.tasks:
            del self.tasks[task_id]

    def pop_random(self) -> Task | None:
        if not self.tasks:
            return None
        task_id = random.choice(list(self.tasks.keys()))
        task = self.tasks[task_id]
        self.delete(task_id)
        return task


class SFTDataset(BaseModel):
    items: list[QA] = []

    def save(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    def register(self, qra: QA) -> None:
        print(f"Details: Registering QA: {qra.question}")
        self.items.append(qra)


class TaskList(BaseModel):
    tasks: list[Task]
