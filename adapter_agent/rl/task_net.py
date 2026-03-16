import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Self

from pydantic import BaseModel
from pyvis.network import Network

from adapter_agent.data import QRA
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
    level: int  # 0 = root
    sft_conclusions: list[Attempt] = field(default_factory=list)

    def register_result(self, result: RewireSessionResult):
        if isinstance(result, RewireSessionResultNormal):
            self.sft_conclusions.append(
                Attempt(
                    result=result,
                    timestamp=datetime.now(),
                )
            )

    @classmethod
    def from_task(cls, task: Task, level: int = 0) -> Self:
        return cls(
            item=task,
            level=level,
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


@dataclass
class StudyTaskResult:
    result: RewireSessionResult
    new_task: Task | None = None


@dataclass
class StudyTaskBase:
    task: TaskWithMeta
    knowledges: list[QRA] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.task.item.id


@dataclass
class StudyNoGenerationTask(StudyTaskBase):
    result: RewireSessionResult | None = None

    def register_result(self, result: RewireSessionResult):
        if self.result is not None:
            raise ValueError("Study task result is already set")
        self.result = result


@dataclass
class StudyGenerationTask(StudyTaskBase):
    result: StudyTaskResult | None = None

    def register_result(
        self, result: RewireSessionResult, new_task: Task | None = None
    ):
        if self.result is not None:
            raise ValueError("Study task result is already set")
        self.result = StudyTaskResult(result=result, new_task=new_task)


type StudyTask = StudyNoGenerationTask | StudyGenerationTask


class PDParams(BaseModel):
    k: float = 1.0
    alpha: float = 0.5


@dataclass
class TaskNetwork:
    nodes: dict[str, TaskWithMeta] = field(default_factory=dict)
    root_id: str | None = None
    children_map: dict[str, list[str]] = field(default_factory=dict)
    executing_tasks: dict[str, int] = field(default_factory=dict)
    executing_generations: dict[str, int] = field(default_factory=dict)
    n_mandatory_parent_trials: int = 3
    pd_params: PDParams = field(default_factory=PDParams)

    def add_root(self, task: Task):
        self.nodes[task.id] = TaskWithMeta.from_task(task)
        self.root_id = task.id

    def _add_edge(self, parent_id: str, child_id: str):
        self.children_map.setdefault(parent_id, []).append(child_id)

    def _add_child_node(self, parent_id: str, new_task: Task):
        child = TaskWithMeta.from_task(new_task, level=self.nodes[parent_id].level + 1)
        self.nodes[child.item.id] = child
        self._add_edge(parent_id, child.item.id)

    @contextmanager
    def get_next_study_task(self) -> Generator[StudyTask, None, None]:
        """
        Selects the next task to study using a progressive widening-like approach.
        """
        if self.root_id is None:
            raise ValueError("TaskNetwork has no root node.")

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
                study_task = StudyGenerationTask(
                    task=node,
                    knowledges=self._get_solved_subtask_qras(curr_id),
                )
                break

            if self._should_prioritize_parent(curr_id):
                study_task = StudyNoGenerationTask(
                    task=node,
                    knowledges=self._get_solved_subtask_qras(curr_id),
                )
                break

            # Filter out solved children
            unsolved_children = [
                c_id for c_id in children if not self.nodes[c_id].is_sft_solved
            ]

            if not unsolved_children:
                # If there are no unsolved children, treat as leaf node
                study_task = StudyNoGenerationTask(
                    task=node,
                    knowledges=self._get_solved_subtask_qras(curr_id),
                )
                break

            # Traverse to the child with minimum subtree trials among unsolved children
            curr_id = min(
                unsolved_children, key=lambda c_id: subtree_trials.get(c_id, 0)
            )

        self._execution_countup(study_task)
        yield study_task
        self._execution_countdown(study_task)
        self._process_study_task_result(study_task)

    def _execution_countup(self, study_task: StudyTask):
        self.executing_tasks[study_task.id] = (
            self.executing_tasks.get(study_task.id, 0) + 1
        )
        if isinstance(study_task, StudyGenerationTask):
            self.executing_generations[study_task.id] = (
                self.executing_generations.get(study_task.id, 0) + 1
            )

    def _execution_countdown(self, study_task: StudyTask):
        if study_task.id not in self.executing_tasks:
            raise ValueError(f"Task {study_task.id} is not in executing tasks")
        self.executing_tasks[study_task.id] = self.executing_tasks[study_task.id] - 1
        if self.executing_tasks[study_task.id] == 0:
            del self.executing_tasks[study_task.id]
        if isinstance(study_task, StudyGenerationTask):
            if study_task.id not in self.executing_generations:
                raise ValueError(
                    f"Task {study_task.id} is not in executing generations"
                )
            self.executing_generations[study_task.id] = (
                self.executing_generations[study_task.id] - 1
            )
            if self.executing_generations[study_task.id] == 0:
                del self.executing_generations[study_task.id]

    def _process_study_task_result(self, study_task: StudyTask):
        if isinstance(study_task, StudyGenerationTask):
            if study_task.result is None:
                raise ValueError("Study task result is None")
            self.nodes[study_task.id].register_result(study_task.result.result)
            if study_task.result.new_task is not None:
                self._add_child_node(study_task.id, study_task.result.new_task)
        elif isinstance(study_task, StudyNoGenerationTask):
            if study_task.result is None:
                raise ValueError("Study task result is None")
            self.nodes[study_task.id].register_result(study_task.result)
        else:
            raise ValueError(f"Unknown study task type: {type(study_task)}")

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

    def _should_generate_subtask(self, node_id: str) -> bool:
        children = self.children_map.get(node_id, [])
        # Progressive widening condition with virtual nodes
        # |A(s)|_effective = |children| + executing_generations
        node = self.nodes[node_id]
        n_s = node.sft_attempts_count + self.executing_tasks.get(node_id, 0)
        effective_children_count = len(children) + self.executing_generations.get(
            node_id, 0
        )
        return (
            effective_children_count
            <= self.pd_params.k * (n_s**self.pd_params.alpha) - 1
        )

    def _get_solved_subtask_qras(self, node_id: str) -> list[QRA]:
        qras = []
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
                qras.append(attempt.result.qra)

            for attempt in child_node.sft_conclusions:
                if isinstance(attempt.result, RewireSessionResultSuccess):
                    qras.append(attempt.result.qra)
        return qras

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
