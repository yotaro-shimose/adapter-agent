import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Self

from tinker import SamplingClient

from adapter_agent.data import QRA
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.internalize.types import (
    GroupRolloutResult,
    InternalizationQRA,
    InternalizationTask,
    Mastery,
    MasteryConfig,
    RLTask,
    SingleRolloutResult,
)
from adapter_agent.rl.shared_sampling_client import (
    IndexedSamplingClient,
    SharedSamplingClient,
)


@dataclass
class UsedGroupRolloutResult:
    task_id: str
    trajectories: list[SingleRolloutResult]
    used: bool

    @classmethod
    def from_group_result(cls, result: GroupRolloutResult) -> Self:
        return cls(task_id=result.task_id, trajectories=result.trajectories, used=False)

    def to_rlgroup(self) -> RLGroup:
        """Transform this rollout group into an RLGroup for GRPO training."""
        if self.used:
            raise ValueError("This rollout group has already been used.")

        trajectories = [res.trajectory for res in self.trajectories]
        # Binary rewards based on success
        rewards = [1.0 if res.success else 0.0 for res in self.trajectories]
        return RLGroup(trajectories=trajectories, rewards=rewards)

    @property
    def success_ratio(self) -> float:
        return sum(1.0 if res.success else 0.0 for res in self.trajectories) / len(
            self.trajectories
        )


@dataclass
class InternalizationKnowledge:
    knowledge: Knowledge
    qras: list[InternalizationQRA]
    groups: list[UsedGroupRolloutResult]  # sampling_version -> list[GroupRolloutResult]
    running_qra_generation: int
    running_sft: dict[str, InternalizationQRA]  # qra_id -> InternalizationQRA
    running_rl: dict[str, RLTask]  # task_id -> RLTask
    mastery_config: MasteryConfig

    @property
    def mastery(self) -> Mastery:
        if self.latest_success_ratio >= self.mastery_config.success_threshold:
            return "mastered"
        elif self.latest_success_group_ratio >= self.mastery_config.studying_threshold:
            return "practicing"
        else:
            return "studying"

    @property
    def qra_target(self) -> int:
        if self.mastery == "mastered":
            return 0
        if self.mastery == "practicing":
            return int(self.mastery_config.k_rl * self.mastery_config.overgen_factor)
        return int(
            (self.mastery_config.k_sft + self.mastery_config.k_rl)
            * self.mastery_config.overgen_factor
        )

    def get_required_qra(self) -> int:
        return (
            self.qra_target
            - self.running_qra_generation
            - len(self.running_rl)
            - len(self.running_sft)
            - len(self.qras)
        )

    def increment_running_qra(self) -> None:
        self.running_qra_generation += 1

    def decrement_running_qra(self) -> None:
        self.running_qra_generation -= 1

    @property
    def latest_success_group_ratio(self) -> float:
        groups_of_interest = self.groups[-self.mastery_config.k_rl :]
        if not groups_of_interest:
            return 0.0
        success_group_count = sum(
            1.0 for g in groups_of_interest if g.success_ratio > 0
        )
        return success_group_count / len(groups_of_interest)

    @property
    def latest_success_ratio(self) -> float:
        groups_of_interest = self.groups[-self.mastery_config.k_rl :]
        if not groups_of_interest:
            return 0.0

        success_count = sum(
            1.0 for g in groups_of_interest for res in g.trajectories if res.success
        )
        total_count = sum(len(g.trajectories) for g in groups_of_interest)
        return success_count / total_count

    def pop_sft_qras(self) -> list[InternalizationQRA]:
        if self.sft_ready_count() == 0:
            return []
        pulled = self.qras[: self.mastery_config.k_sft]
        self.qras = self.qras[self.mastery_config.k_sft :]
        running_sft = {qra.id: qra for qra in pulled}
        self.running_sft.update(running_sft)
        return pulled

    def sft_ready_count(self) -> int:
        if self.mastery != "studying":
            return 0
        if self.running_sft:
            return 0
        if len(self.qras) < self.mastery_config.k_sft:
            return 0
        return self.mastery_config.k_sft

    def rl_ready_count(self) -> int:
        return len([g for g in self.groups if not g.used])

    def pop_rl_groups(self) -> list[RLGroup]:
        groups = [g.to_rlgroup() for g in self.groups if not g.used]
        for g in self.groups:
            g.used = True
        return groups

    def rollout_ready_count(self) -> int:
        num_running_rl = len(self.running_rl)
        if num_running_rl >= self.mastery_config.k_rl:
            return 0
        required_rollout_count = self.mastery_config.k_rl - num_running_rl
        return min(required_rollout_count, len(self.qras))

    def pop_rollout_qra(self) -> InternalizationQRA:
        qra = self.qras.pop(0)
        task = RLTask(
            knowledge_id=self.knowledge.id,
            instruction=qra.question,
        )
        self.running_rl[task.id] = task
        return qra

    def push_rollout_result(self, result: GroupRolloutResult) -> None:
        self.running_rl.pop(result.task_id)
        new_group = UsedGroupRolloutResult.from_group_result(result)
        self.groups.append(new_group)

    def push_qra(self, qra: QRA) -> None:
        internalization_qra = InternalizationQRA(
            question=qra.question,
            reasoning=qra.reasoning,
            answer=qra.answer,
            knowledge_id=self.knowledge.id,
        )
        self.qras.append(internalization_qra)


@dataclass
class KnowledgeMasteryManager:
    knowledges: dict[str, InternalizationKnowledge]
    mastery_config: MasteryConfig

    def get_replenishment_plan(self) -> list[tuple[Knowledge, int]]:
        """Calculate how many QRAs to generate for each underperforming knowledge."""
        return [(k.knowledge, k.get_required_qra()) for k in self.knowledges.values()]

    def get_status_summary(self) -> dict[str, float]:
        """Return a mapping of knowledge_id to current success ratio."""
        return {
            k.knowledge.id: k.latest_success_group_ratio
            for k in self.knowledges.values()
        }

    def get_available_sft_qras(self) -> int:
        return sum(k.sft_ready_count() for k in self.knowledges.values())

    def pop_sft_batch(self, min_batch_size: int) -> list[InternalizationQRA] | None:
        """
        Returns a combined batch of QRAs if:
        1. Total available QRAs meets min_batch_size.
        2. At least one underperforming knowledge (success < threshold) has enough data.
        Returns None if conditions aren't met, ensuring no QRAs are popped prematurely.
        """
        total_available = self.get_available_sft_qras()
        if total_available < min_batch_size:
            return None

        studying_knowledges = [
            k for k in self.knowledges.values() if k.mastery == "studying"
        ]

        non_studying_knowledges = [
            k for k in self.knowledges.values() if k.mastery != "studying"
        ]

        if not studying_knowledges:
            return None

        # At this point, we are guaranteed that total_available >= min_batch_size,
        # so we can safely pop qras without losing data.
        batch: list[InternalizationQRA] = []
        for k in studying_knowledges:
            batch.extend(k.pop_sft_qras())
            if len(batch) >= min_batch_size:
                break

        if len(batch) >= min_batch_size:
            return batch

        for k in non_studying_knowledges:
            batch.extend(k.pop_sft_qras())
            if len(batch) >= min_batch_size:
                break

        return batch

    def report_sft_results(self, qras: list[InternalizationQRA]) -> None:
        """Remove qras from running sft"""
        for qra in qras:
            knowledge = self.knowledges[qra.knowledge_id]
            knowledge.running_sft.pop(qra.id)

    def pop_rollout_task(self) -> InternalizationTask | None:
        counts = {
            k.knowledge.id: k.rollout_ready_count()
            for k in self.knowledges.values()
            if k.rollout_ready_count() > 0
        }
        if not counts:
            return None
        knowledge_id = random.choice(list(counts.keys()))
        knowledge = self.knowledges[knowledge_id]
        qra = knowledge.pop_rollout_qra()

        return InternalizationTask(
            knowledge_id=knowledge_id,
            instruction=qra.question,
        )

    def push_rollout_result(self, result: GroupRolloutResult) -> None:
        knowledge = self.knowledges[result.knowledge_id]
        knowledge.push_rollout_result(result)

    def pop_rl_batch(self, min_batch_size: int) -> list[RLGroup] | None:
        available_group_count = sum(
            k.rl_ready_count() for k in self.knowledges.values()
        )
        if available_group_count < min_batch_size:
            return None

        return list(
            chain.from_iterable(
                [
                    k.pop_rl_groups()
                    for k in self.knowledges.values()
                    if k.rl_ready_count() > 0
                ]
            )
        )

    def push_qra(self, knowledge_id: str, qra: QRA) -> None:
        self.knowledges[knowledge_id].push_qra(qra)

    def report_qra_generation_start(self, knowledge_id: str) -> None:
        self.knowledges[knowledge_id].increment_running_qra()

    def report_qra_generation_end(self, knowledge_id: str) -> None:
        self.knowledges[knowledge_id].decrement_running_qra()


@dataclass
class GlobalState:
    sampling_client: SharedSamplingClient
    knowledge_manager: KnowledgeMasteryManager

    async def update_sampling_client(self, sampling_client: SamplingClient) -> None:
        await self.sampling_client.update_client(sampling_client)

    async def get_sampling_client(self) -> IndexedSamplingClient:
        return await self.sampling_client.get_client()

    async def get_replenishment_plan(self) -> list[tuple[Knowledge, int]]:
        """Calculate how many QRAs to generate for each underperforming knowledge."""
        return self.knowledge_manager.get_replenishment_plan()

    async def get_status_summary(self) -> dict[str, float]:
        """Return a mapping of knowledge_id to current success ratio."""
        return self.knowledge_manager.get_status_summary()

    async def report_qra_generation_result(
        self, knowledge_id: str, qra: Optional[QRA] = None
    ) -> None:
        """Report results of a generation task and end its lifecycle."""
        if qra:
            self.knowledge_manager.push_qra(knowledge_id, qra)
        self.knowledge_manager.report_qra_generation_end(knowledge_id)

    async def report_qra_generation_start(self, knowledge_id: str) -> None:
        self.knowledge_manager.report_qra_generation_start(knowledge_id)

    # --- Rollout Interfaces ---
    async def pop_rollout_task(self) -> InternalizationTask | None:
        """Get a task from the queue. Called by RemoteWorkers."""
        return self.knowledge_manager.pop_rollout_task()

    async def pop_rl_batch(self, min_batch_size: int) -> list[RLGroup] | None:
        """Returns results only if the queue size exceeds min_batch_size."""
        return self.knowledge_manager.pop_rl_batch(min_batch_size)

    async def pop_sft_batch(
        self, min_batch_size: int
    ) -> list[InternalizationQRA] | None:
        return self.knowledge_manager.pop_sft_batch(min_batch_size)

    async def report_sft_results(self, qras: list[InternalizationQRA]) -> None:
        self.knowledge_manager.report_sft_results(qras)

    async def push_rollout_result(self, result: GroupRolloutResult) -> None:
        self.knowledge_manager.push_rollout_result(result)
