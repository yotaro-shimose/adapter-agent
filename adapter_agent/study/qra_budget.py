"""Per-task QRA generation budget tracker.

Each RL-failed task has a quota of verified QRAs we want to harvest
(`target_qras_per_task`). Trajectories for the same task arrive online (we
don't know how many will succeed), so we distribute distillation across
whatever is available at the moment, always preferring the least-used
trajectory for diversity.

Concurrency: many StudyWorkers may be distilling for the same task in
parallel. To avoid over-emitting, `pick` reserves a slot via an `in_flight`
counter under a per-task lock; `record_result` releases it. I/O (Gemini,
cargo) happens outside the lock.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from tinker_cookbook.renderers.base import Message as TinkerMessage

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class QRABudgetConfig:
    target_qras_per_task: int = 8
    max_attempts_multiplier: float = 2.0

    @property
    def max_attempts_per_task(self) -> int:
        return max(
            self.target_qras_per_task,
            math.ceil(self.target_qras_per_task * self.max_attempts_multiplier),
        )


@dataclass
class TaskState:
    trajectories: list[list[TinkerMessage]] = field(default_factory=list)
    attempts_per_traj: list[int] = field(default_factory=list)
    emitted: int = 0
    in_flight: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts_per_traj)


@dataclass
class PickedTrajectory:
    index: int
    trajectory: list[TinkerMessage]


class QRABudgetTracker:
    def __init__(self, config: QRABudgetConfig) -> None:
        self._config = config
        self._states: dict[str, TaskState] = {}
        self._global_lock = asyncio.Lock()

    async def _get_state(self, task_id: str) -> TaskState:
        async with self._global_lock:
            return self._states.setdefault(task_id, TaskState())

    async def add_trajectory(
        self, task_id: str, trajectory: list[TinkerMessage]
    ) -> None:
        state = await self._get_state(task_id)
        async with state.lock:
            state.trajectories.append(trajectory)
            state.attempts_per_traj.append(0)

    async def pick(self, task_id: str) -> PickedTrajectory | None:
        """Reserve a distillation slot against the quota.

        Returns the least-used trajectory for the task, or None if the quota
        is already met (or about to be met via in-flight attempts) or the
        attempt budget is exhausted.
        """
        state = await self._get_state(task_id)
        async with state.lock:
            if not state.trajectories:
                return None
            if state.emitted + state.in_flight >= self._config.target_qras_per_task:
                return None
            if state.total_attempts >= self._config.max_attempts_per_task:
                return None
            idx = min(
                range(len(state.trajectories)),
                key=lambda i: state.attempts_per_traj[i],
            )
            state.attempts_per_traj[idx] += 1
            state.in_flight += 1
            return PickedTrajectory(index=idx, trajectory=state.trajectories[idx])

    async def record_result(self, task_id: str, success: bool) -> None:
        state = await self._get_state(task_id)
        async with state.lock:
            state.in_flight = max(0, state.in_flight - 1)
            if success:
                state.emitted += 1

    async def snapshot(self, task_id: str) -> dict[str, Any]:
        state = await self._get_state(task_id)
        async with state.lock:
            return {
                "emitted": state.emitted,
                "in_flight": state.in_flight,
                "trajectories": len(state.trajectories),
                "total_attempts": state.total_attempts,
                "quota": self._config.target_qras_per_task,
                "max_attempts": self._config.max_attempts_per_task,
            }
