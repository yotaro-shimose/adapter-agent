import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self

import polars as pl
from pydantic import BaseModel, PrivateAttr
from tinker_cookbook.rl.types import Trajectory

from adapter_agent.data import QA, QASFTDataset
from adapter_agent.hierarchical.types import Task

logger = logging.getLogger(__name__)




class RolloutSample(BaseModel):
    """Per-sample audit metadata for one rollout in an RLGroup.

    Stored as `RLGroup.samples`. `reward` lives separately in `RLGroup.rewards`
    so the GRPO hot path keeps its tight `list[float]` representation.
    """

    answer: str
    reasoning: str = ""
    parsed: bool
    success: bool
    execution_output: str
    verification_output: str


class RLGroup(BaseModel):
    trajectories: list[Trajectory]
    rewards: list[float]
    sampling_client_version: int = -1
    # Audit metadata — optional, populated by RLWorkerPool when DB persistence
    # of rollouts is desired. None on the Ray path (TODO: plumb through).
    suite_name: str | None = None
    task_id: str | None = None
    instruction: str | None = None
    samples: list[RolloutSample] | None = None


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
    def from_benchmark_csv(cls, path: Path, num_tasks: int | None = None) -> Self:
        benchmark_df = pl.read_csv(path).filter(pl.col("appropriate"))
        if num_tasks is not None:
            benchmark_df = benchmark_df.head(num_tasks)
        tasks = {}
        for row in benchmark_df.iter_rows(named=True):
            task = Task.from_instruction(row["problem_statement"])
            tasks[task.id] = task
        return cls(tasks=tasks)


@dataclass
class SFTPool:
    dataset: QASFTDataset
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
    def from_sft_dataset(cls, dataset: QASFTDataset) -> Self:
        return cls(dataset=dataset, queue=asyncio.Queue())

    @classmethod
    def new(cls) -> Self:
        return cls(dataset=QASFTDataset(), queue=asyncio.Queue())


@dataclass
class RLPool:
    queue: asyncio.Queue[RLGroup]

    async def register(self, group: RLGroup) -> None:
        await self.queue.put(group)

    def get_batch(self, batch_size: int) -> list[RLGroup]:
        batch = []
        for _ in range(batch_size):
            if self.queue.empty():
                break
            batch.append(self.queue.get_nowait())
        return batch

    @classmethod
    def new(cls) -> Self:
        return cls(queue=asyncio.Queue())

    def save(self, path: Path) -> None:
        """
        Save the current state of the RLPool to a JSON file.
        Note: This inspects the internal queue without consuming items.
        """
        # Access internal deque to get all items without popping
        # items is a list of RLGroup
        items = list(self.queue._queue)  # type: ignore

        # Serialize items
        serialized_items = [item.model_dump(mode="json") for item in items]

        with path.open("w") as f:
            # We can use json.dump since we already converted pydantic models to dicts/json-safe types
            import json

            json.dump(serialized_items, f, indent=2)


class TaskList(BaseModel):
    tasks: list[Task]
