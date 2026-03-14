from datetime import datetime
from typing import Self

from pydantic import BaseModel
from pyvis.network import Network

from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.env.conclusion import SSConclusion


class Attempt(BaseModel):
    conclusion: SSConclusion
    timestamp: datetime


class TaskWithMeta(BaseModel):
    item: Task
    level: int  # 0 = root
    sft_conclusions: list[Attempt]
    practice_conclusions: list[Attempt]

    def add_sft_conclusion(self, conclusion: SSConclusion):
        self.sft_conclusions.append(Attempt(conclusion=conclusion, timestamp=datetime.now()))

    def add_practice_conclusion(self, conclusion: SSConclusion):
        self.practice_conclusions.append(Attempt(conclusion=conclusion, timestamp=datetime.now()))

    @classmethod
    def from_task(cls, task: Task, level: int = 0) -> Self:
        return cls(
            item=task,
            level=level,
            sft_conclusions=[],
            practice_conclusions=[],
        )

    @property
    def sft_attempts_count(self) -> int:
        return len(self.sft_conclusions)

    @property
    def is_sft_solved(self) -> bool:
        return any(conclusion.conclusion == "success" for conclusion in self.sft_conclusions)

    @property
    def latest_attempt(self) -> Attempt | None:
        if self.sft_conclusions:
            latest_sft = self.sft_conclusions[-1]
        else:
            latest_sft = None
        return latest_sft


class TaskNetwork(BaseModel):
    nodes: dict[str, TaskWithMeta] = {}
    edges: list[tuple[str, str]] = []

    def add_edge(self, parent_id: str, child_id: str):
        self.edges.append((parent_id, child_id))

    def add_root(self, task: Task):
        self.nodes[task.id] = TaskWithMeta.from_task(task)

    def add_child_node(self, parent_id: str, new_task: Task):
        child = TaskWithMeta.from_task(new_task, level=self.nodes[parent_id].level + 1)
        self.nodes[child.item.id] = child
        self.add_edge(parent_id, child.item.id)

    def add_sft_conclusion(self, task_id: str, conclusion: SSConclusion):
        self.nodes[task_id].add_sft_conclusion(conclusion)

    def add_practice_conclusion(self, task_id: str, conclusion: SSConclusion):
        self.nodes[task_id].add_practice_conclusion(conclusion)

    def to_pyvis(self, node_ids: set[str] | None = None, recent_ids: set[str] | None = None):
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
            success_count = sum(1 for c in node.sft_conclusions if c.conclusion == "success")
            total_count = len(node.sft_conclusions)
            
            # Show rate for all nodes if recent_ids is None, 
            # or only for nodes in the recent set if recent_ids is provided.
            show_rate = False
            if recent_ids is None:
                show_rate = True
            elif node.item.id in recent_ids:
                show_rate = True
            
            if show_rate and total_count > 0:
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

            net.add_node(
                node.item.id,
                label=label,
                title=node.item.instruction,
                shape="box",
                color=color,
                font={"color": font_color},
            )
        for edge in self.edges:
            if node_ids is None or (edge[0] in node_ids and edge[1] in node_ids):
                net.add_edge(edge[0], edge[1])
        return net

    def recent_graph(self, n: int = 10):
        # Collect all attempts with their task IDs and timestamps
        all_attempts: list[tuple[datetime, str]] = []
        for node_id, node in self.nodes.items():
            for att in node.sft_conclusions:
                all_attempts.append((att.timestamp, node_id))
            for att in node.practice_conclusions:
                all_attempts.append((att.timestamp, node_id))
        
        # Sort by timestamp to find the latest
        all_attempts.sort(key=lambda x: x[0])
        
        # Take the task IDs associated with the last n attempts
        recent_attempts = all_attempts[-n:]
        recent_ids_set = {task_id for _, task_id in recent_attempts}

        # Include all parents of these recent nodes for context
        child_to_parent = {edge[1]: edge[0] for edge in self.edges}

        to_include = set(recent_ids_set)
        for node_id in recent_ids_set:
            curr = node_id
            while curr in child_to_parent:
                parent = child_to_parent[curr]
                to_include.add(parent)
                curr = parent

        return self.to_pyvis(node_ids=to_include, recent_ids=recent_ids_set)

    def save_visualization(self, path: str, recent_n: int | None = None):
        if recent_n is not None:
            pyvis_net = self.recent_graph(recent_n)
        else:
            pyvis_net = self.to_pyvis()
        pyvis_net.save_graph(path)
