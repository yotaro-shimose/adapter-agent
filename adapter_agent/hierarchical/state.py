import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self

import polars as pl
from pydantic import BaseModel, PrivateAttr

from adapter_agent.hierarchical.types import Task
from adapter_agent.qra import QA


logger = logging.getLogger(__name__)


class TaskPool(BaseModel):
    tasks: dict[str, Task]

    # Private attributes for concurrency control
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _condition: asyncio.Condition = PrivateAttr(default_factory=asyncio.Condition)
    _active_workers: int = PrivateAttr(default=0)

    def model_post_init(self, context: Any):
        self._condition = asyncio.Condition(lock=self._lock)

    async def register(self, task: Task) -> None:
        async with self._lock:
            logger.info(f"Details: Registering task: {task.instruction}")
            self.tasks[task.id] = task
            self._condition.notify_all()

    async def pop_task(self) -> Optional[Task]:
        """
        Wait for a task to be available.
        Returns None if no tasks are left AND no workers are active (shutdown).
        """
        async with self._lock:
            while True:
                if self.tasks:
                    # Atomic pop (LIFO: newest task first)
                    _, task = self.tasks.popitem()
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
        async with self._lock:
            self._active_workers -= 1
            # Notify waiters (pop_task might return None if active_workers drops to 0)
            self._condition.notify_all()

    def save(self, path: Path) -> None:
        # Note: This is synchronous and might be unsafe if called concurrently without lock.
        # But for saving at the end, it is fine.
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_benchmark_csv(cls, path: Path) -> Self:
        benchmark_df = pl.read_csv(path).filter(pl.col("appropriate"))
        tasks = {}
        for row in benchmark_df.iter_rows(named=True):
            task = Task.from_instruction(row["problem_statement"])
            tasks[task.id] = task
        return cls(tasks=tasks)


class SFTDataset(BaseModel):
    items: list[QA] = []

    def save(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json(indent=2))

    async def register(self, item: QA) -> None:
        logger.info(f"Details: Registering QA: {item.question}")
        self.items.append(item)


@dataclass
class SFTPool:
    dataset: SFTDataset
    queue: asyncio.Queue[QA]

    async def register(self, item: QA) -> None:
        await self.queue.put(item)
        self.dataset.items.append(item)

    def get_batch(self, batch_size: int) -> list[QA]:
        batch = []
        for _ in range(batch_size):
            batch.append(self.queue.get_nowait())
        return batch

    @classmethod
    def from_sft_dataset(cls, dataset: SFTDataset) -> Self:
        return cls(dataset=dataset, queue=asyncio.Queue())

    @classmethod
    def new(cls) -> Self:
        return cls(dataset=SFTDataset(), queue=asyncio.Queue())


class TaskList(BaseModel):
    tasks: list[Task]
