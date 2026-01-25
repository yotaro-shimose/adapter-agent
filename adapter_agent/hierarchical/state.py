import asyncio
from pydantic import PrivateAttr
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA


class TaskPool(BaseModel):
    tasks: dict[str, Task]

    # Private attributes for concurrency control
    _lock: asyncio.Lock = PrivateAttr(default=None)
    _condition: asyncio.Condition = PrivateAttr(default=None)
    _active_workers: int = PrivateAttr(default=0)

    def __init__(self, **data):
        super().__init__(**data)
        # We can't init asyncio primitives here easily if loop isn't running or depends on context
        # We will lazy init or expect explicit setup if needed.
        # But actually, if main() runs asyncio.run(), the loop is active.
        # Use simple lazy property or check in methods.

    def _ensure_primitives(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._condition = asyncio.Condition(lock=self._lock)

    async def register(self, task: Task) -> None:
        self._ensure_primitives()
        async with self._lock:
            print(f"Details: Registering task: {task.instruction}")
            self.tasks[task.id] = task
            self._condition.notify_all()

    async def pop_task(self) -> Optional[Task]:
        """
        Wait for a task to be available.
        Returns None if no tasks are left AND no workers are active (shutdown).
        """
        self._ensure_primitives()
        async with self._lock:
            while True:
                if self.tasks:
                    # Atomic pop
                    task_id = next(iter(self.tasks))
                    task = self.tasks.pop(task_id)
                    self._active_workers += 1
                    return task

                if self._active_workers == 0:
                    # No tasks and no one working on tasks -> We are done.
                    return None

                # Wait for something to happen (new task registered or a worker finishing)
                await self._condition.wait()

    async def finish_task(self, task: Task) -> None:
        """
        Call this when a worker finishes processing a task.
        """
        self._ensure_primitives()
        async with self._lock:
            self._active_workers -= 1
            # Notify waiters (pop_task might return None if active_workers drops to 0)
            self._condition.notify_all()

    def save(self, path: Path) -> None:
        # Note: This is synchronous and might be unsafe if called concurrently without lock.
        # But for saving at the end, it is fine.
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))


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
