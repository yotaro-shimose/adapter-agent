from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from inspect import isawaitable
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Self

from pydantic import BaseModel
from pyvis.network import Network

from adapter_agent.hierarchical.process.rewire_session import (
    RewireSessionResult,
    RewireSessionResultNormal,
    RewireSessionResultSuccess,
)
from adapter_agent.hierarchical.types import Task


@dataclass
class Attempt:
    timestamp: datetime
    result: RewireSessionResult


@dataclass
class TaskWithMeta:
    item: Task
    sft_conclusions: list[Attempt] = field(default_factory=list)
    PSEUDO_ROOT_ID: ClassVar[str] = "pseudo_root"

    def register_result(self, result: RewireSessionResult):
        if isinstance(result, RewireSessionResultNormal):
            self.sft_conclusions.append(
                Attempt(
                    result=result,
                    timestamp=datetime.now(),
                )
            )

    @classmethod
    def pseudo_root(cls) -> Self:
        return cls(
            item=Task(
                id=cls.PSEUDO_ROOT_ID,
                instruction="pseudo_root",
            ),
            sft_conclusions=[],
        )

    def is_pseudo_root(self) -> bool:
        return self.item.id == self.PSEUDO_ROOT_ID

    @classmethod
    def from_task(cls, task: Task) -> Self:
        return cls(
            item=task,
            sft_conclusions=[],
        )

    @property
    def sft_attempts_count(self) -> int:
        return len(self.sft_conclusions)

    @property
    def is_sft_solved(self) -> bool:
        return any(attempt.result.is_successful() for attempt in self.sft_conclusions)

    @property
    def latest_attempt(self) -> Attempt | None:
        if self.sft_conclusions:
            latest_sft = self.sft_conclusions[-1]
        else:
            latest_sft = None
        return latest_sft


@dataclass(kw_only=True)
class StudyTask:
    task: Task
    knowledges: list[str] = field(default_factory=list)
    is_generation: bool

    @property
    def id(self) -> str:
        return self.task.id

    def complete(
        self, result: RewireSessionResult, new_task: Task | None = None
    ) -> StudyTaskCompleted:
        return StudyTaskCompleted(
            task=self.task,
            knowledges=self.knowledges,
            is_generation=self.is_generation,
            result=result,
            new_task=new_task,
        )


@dataclass(kw_only=True)
class StudyTaskCompleted(StudyTask):
    result: RewireSessionResult
    new_task: Task | None = None


class PWParams(BaseModel):
    """Progressive Widening parameters"""

    k: float = 1.0
    alpha: float = 0.5


@dataclass
class TaskNetwork:
    tasks_pool: list[Task]
    nodes: dict[str, TaskWithMeta] = field(default_factory=dict)
    children_map: dict[str, list[str]] = field(default_factory=dict)
    executing_tasks: dict[str, int] = field(default_factory=dict)
    executing_generations: dict[str, int] = field(default_factory=dict)
    pd_params: PWParams = field(default_factory=PWParams)
    n_mandatory_parent_trials: int = 3
    root_id: str = field(init=False)

    def __post_init__(self):
        pseudo_root = TaskWithMeta.pseudo_root()
        self.nodes[pseudo_root.item.id] = pseudo_root
        self.root_id = pseudo_root.item.id

    def add_branch_root(self) -> Task | None:
        if not self.tasks_pool:
            return None
        next_task = self.tasks_pool.pop()
        self._add_child_node(self.root_id, next_task)
        return next_task

    def _add_edge(self, parent_id: str, child_id: str):
        self.children_map.setdefault(parent_id, []).append(child_id)

    def _add_child_node(self, parent_id: str, new_task: Task):
        child = TaskWithMeta.from_task(new_task)
        self.nodes[child.item.id] = child
        self._add_edge(parent_id, child.item.id)

    def get_next_study_task(self) -> StudyTask:
        curr_id = self.root_id

        # Pre-calculate subtree trials N(s) for all nodes
        subtree_trials: dict[str, int] = {}

        def get_subtree_trials(node_id: str) -> int:
            if node_id in subtree_trials:
                return subtree_trials[node_id]
            node = self.nodes[node_id]
            # N(s) = T(s) + in_flight_tasks(s) + in_flight_generations(s) + sum(N(c))
            count = len(node.sft_conclusions)
            count += self.executing_tasks.get(node_id, 0)
            for child_id in self.children_map.get(node_id, []):
                count += get_subtree_trials(child_id)
            subtree_trials[node_id] = count
            return count

        # Populate the trials map
        get_subtree_trials(curr_id)

        while True:
            node = self.nodes[curr_id]
            children = self.children_map.get(curr_id, [])
            generation = self._should_generate_subtask(curr_id)

            if generation:
                if node.is_pseudo_root():
                    new_task = self.add_branch_root()
                    if new_task is None:
                        raise ValueError("No more tasks in the pool.")
                    study_task = StudyTask(
                        task=new_task,
                        knowledges=[],
                        is_generation=False,
                    )
                else:
                    study_task = StudyTask(
                        task=node.item,
                        knowledges=self._get_solved_subtask_knowledges(curr_id),
                        is_generation=True,
                    )
                break

            if self._should_prioritize_parent(curr_id):
                study_task = StudyTask(
                    task=node.item,
                    knowledges=self._get_solved_subtask_knowledges(curr_id),
                    is_generation=False,
                )
                break

            # Filter out solved children
            unsolved_children = [
                c_id for c_id in children if not self.nodes[c_id].is_sft_solved
            ]

            if not unsolved_children:
                if node.is_pseudo_root():
                    new_task = self.add_branch_root()
                    if new_task is None:
                        raise ValueError("No more tasks in the pool.")
                    study_task = StudyTask(
                        task=new_task,
                        knowledges=[],
                        is_generation=False,
                    )
                else:
                    study_task = StudyTask(
                        task=node.item,
                        knowledges=self._get_solved_subtask_knowledges(curr_id),
                        is_generation=False,
                    )
                break

            # Traverse to the child with minimum subtree trials among unsolved children
            curr_id = min(
                unsolved_children, key=lambda c_id: subtree_trials.get(c_id, 0)
            )
        return study_task

    def get_and_setup_next_study_task(self) -> StudyTask:
        study_task = self.get_next_study_task()
        self.study_task_setup(study_task)
        return study_task

    def study_task_setup(self, study_task: StudyTask):
        self._execution_countup(study_task)

    def study_task_teardown(self, completed: StudyTaskCompleted):
        self._execution_countdown(completed)
        self._process_study_task_result(completed)

    def _execution_countup(self, study_task: StudyTask):
        self.executing_tasks[study_task.id] = (
            self.executing_tasks.get(study_task.id, 0) + 1
        )
        if study_task.is_generation:
            self.executing_generations[study_task.id] = (
                self.executing_generations.get(study_task.id, 0) + 1
            )

    def _execution_countdown(self, completed: StudyTaskCompleted):
        if completed.id not in self.executing_tasks:
            raise ValueError(f"Task {completed.id} is not in executing tasks")
        self.executing_tasks[completed.id] = self.executing_tasks[completed.id] - 1
        if self.executing_tasks[completed.id] == 0:
            del self.executing_tasks[completed.id]
        if completed.is_generation:
            if completed.id not in self.executing_generations:
                raise ValueError(f"Task {completed.id} is not in executing generations")
            self.executing_generations[completed.id] = (
                self.executing_generations[completed.id] - 1
            )
            if self.executing_generations[completed.id] == 0:
                del self.executing_generations[completed.id]

    def _process_study_task_result(self, completed: StudyTaskCompleted):
        self.nodes[completed.id].register_result(completed.result)
        if completed.new_task is not None:
            if not completed.is_generation:
                raise ValueError("Task is not generation task but result has new task")
            self._add_child_node(completed.id, completed.new_task)

    def _get_latest_child_success_ts(self, node_id: str) -> datetime | None:
        latest_ts: datetime | None = None
        children = self.children_map.get(node_id, [])
        for child_id in children:
            child_node = self.nodes[child_id]
            for attempt in child_node.sft_conclusions:
                if attempt.result.is_successful():
                    if latest_ts is None or attempt.timestamp > latest_ts:
                        latest_ts = attempt.timestamp
        return latest_ts

    def _should_prioritize_parent(self, node_id: str) -> bool:
        if node_id == self.root_id:
            return False
        latest_child_success = self._get_latest_child_success_ts(node_id)
        if latest_child_success is None:
            return False

        node = self.nodes[node_id]
        trials_after_success = sum(
            1
            for attempt in node.sft_conclusions
            if attempt.timestamp > latest_child_success
        )
        return trials_after_success < self.n_mandatory_parent_trials

    def _get_ns(self, node_id: str) -> int:
        node = self.nodes[node_id]
        return node.sft_attempts_count + self.executing_tasks.get(node_id, 0)

    def _should_generate_subtask(self, node_id: str) -> bool:
        children = self.children_map.get(node_id, [])
        # Progressive widening condition with virtual nodes
        # |A(s)|_effective = |children| + executing_generations
        node = self.nodes[node_id]

        if node.is_pseudo_root():
            if len(self.tasks_pool) == 0:
                return False
            n_s = sum(self._get_ns(child_id) for child_id in children)
            effective_children_count = len(
                [c for c in children if not self.nodes[c].is_sft_solved]
            )
        else:
            n_s = self._get_ns(node_id)
            effective_children_count = len(children) + self.executing_generations.get(
                node_id, 0
            )
        return (
            effective_children_count
            <= self.pd_params.k * (n_s**self.pd_params.alpha) - 1
        )

    def _get_solved_subtask_knowledges(self, node_id: str) -> list[str]:
        knowledges = []
        for child_id in self.children_map.get(node_id, []):
            child_node = self.nodes[child_id]
            successful_attempts = [
                attempt
                for attempt in child_node.sft_conclusions
                if isinstance(attempt.result, RewireSessionResultSuccess)
            ]
            if len(successful_attempts) > 0:
                attempt = random.choice(successful_attempts)
                assert isinstance(attempt.result, RewireSessionResultSuccess)
                knowledges.append(attempt.result.knowledge)

            for attempt in child_node.sft_conclusions:
                if isinstance(attempt.result, RewireSessionResultSuccess):
                    knowledges.append(attempt.result.knowledge)
        return knowledges

    def to_pyvis(
        self, node_ids: set[str] | None = None, recent_ids: set[str] | None = None
    ):
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            directed=True,
        )

        def wrap_text(text, width=30, max_lines=3):
            import textwrap

            lines = textwrap.wrap(text, width=width)
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] += "..."
            return "\n".join(lines)

        nodes_to_include = (
            self.nodes.values()
            if node_ids is None
            else [self.nodes[id] for id in node_ids if id in self.nodes]
        )

        for node in nodes_to_include:
            instruction_label = wrap_text(node.item.instruction)

            # Success rate calculation
            success_count = sum(
                1 for c in node.sft_conclusions if c.result.is_successful()
            )
            total_count = len(node.sft_conclusions)

            # Show rate for all nodes if recent_ids is None,
            # or only for nodes in the recent set if recent_ids is provided.
            show_rate = False
            if recent_ids is None:
                show_rate = True
            elif node.item.id in recent_ids:
                show_rate = True

            if node.item.id in self.executing_tasks:
                instruction_label = f"⚡ {instruction_label}"

            gen_count = self.executing_generations.get(node.item.id, 0)
            if gen_count > 0:
                label = f"{instruction_label}\n({success_count}/{total_count}) 📂({gen_count})"
            elif show_rate and total_count > 0:
                label = f"{instruction_label}\n({success_count}/{total_count})"
            else:
                label = instruction_label

            attempts = node.sft_attempts_count
            lightness = max(5, 80 - attempts * 15)

            if node.is_sft_solved:
                color = "#28a745"
                font_color = "white"
            else:
                color = f"hsl(0, 0%, {lightness}%)"
                font_color = "white"

            node_opts: dict[str, Any] = {
                "label": label,
                "title": node.item.instruction,
                "shape": "box",
                "color": color,
                "font": {"color": font_color},
            }

            if node.item.id in self.executing_tasks:
                node_opts["borderWidth"] = 4

            if gen_count > 0:
                node_opts["shapeProperties"] = {"borderDashes": [5, 5]}

            net.add_node(node.item.id, **node_opts)
        for parent_id, children in self.children_map.items():
            for child_id in children:
                if node_ids is None or (parent_id in node_ids and child_id in node_ids):
                    net.add_edge(parent_id, child_id)
        return net

    def recent_graph(self, n: int = 10):
        # Collect all attempts with their task IDs and timestamps
        all_attempts: list[tuple[datetime, str]] = []
        for node_id, node in self.nodes.items():
            for att in node.sft_conclusions:
                all_attempts.append((att.timestamp, node_id))

        # Sort by timestamp to find the latest
        all_attempts.sort(key=lambda x: x[0])

        # Take the task IDs associated with the last n attempts
        recent_attempts = all_attempts[-n:]
        recent_ids_set = {task_id for _, task_id in recent_attempts}

        # Include all parents of these recent nodes for context
        child_to_parent = {
            child_id: parent_id
            for parent_id, children in self.children_map.items()
            for child_id in children
        }

        to_include = set(recent_ids_set)
        for node_id in recent_ids_set:
            curr = node_id
            while curr in child_to_parent:
                parent = child_to_parent[curr]
                to_include.add(parent)
                curr = parent

        return self.to_pyvis(node_ids=to_include, recent_ids=recent_ids_set)

    def save_visualization(self, path: Path, recent_n: int | None = None):
        if recent_n is not None:
            pyvis_net = self.recent_graph(recent_n)
        else:
            pyvis_net = self.to_pyvis()
        pyvis_net.save_graph(str(path))

    def node_count(self) -> int:
        return len(self.nodes)


@dataclass
class StudyTaskContext:
    task: StudyTask
    register: (
        Callable[[StudyTaskCompleted], Awaitable[None]]
        | Callable[[StudyTaskCompleted], None]
    )
    completed: StudyTaskCompleted | None = None

    @classmethod
    def next_task_from_network(cls, task_network: TaskNetwork):
        task = task_network.get_and_setup_next_study_task()
        return cls(task=task, register=task_network.study_task_teardown)

    def register_result(self, result: RewireSessionResult, new_task: Task | None):
        self.completed = self.task.complete(result=result, new_task=new_task)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.completed is None:
            raise ValueError("No result registered for task.")
        ret = self.register(self.completed)
        if isawaitable(ret):
            await ret
