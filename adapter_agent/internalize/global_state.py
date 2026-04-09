import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Self

import ray
from tinker import SamplingClient
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

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
from adapter_agent.util.logger_util import setup_base_loglevel


@dataclass
class UsedGroupRolloutResult:
    task_id: str
    trajectories: list[SingleRolloutResult]
    used: bool
    sampling_version: int

    @classmethod
    def from_group_result(cls, result: GroupRolloutResult) -> Self:
        return cls(
            task_id=result.task_id,
            trajectories=result.trajectories,
            used=False,
            sampling_version=result.current_sampling_version,
        )

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

    def is_learnable(self) -> bool:
        """A group is learnable if there is at least one success and one failure."""
        if not self.trajectories:
            return False
        first_success = self.trajectories[0].success
        return any(res.success != first_success for res in self.trajectories)


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
        return len([g for g in self.groups if not g.used and g.is_learnable()])

    def pop_rl_groups(self) -> list[RLGroup]:
        learnable_groups = []
        for g in self.groups:
            if not g.used:
                if g.is_learnable():
                    learnable_groups.append(g.to_rlgroup())
                g.used = True
        return learnable_groups

    def rollout_ready_count(self) -> int:
        num_running_rl = len(self.running_rl)
        if num_running_rl >= self.mastery_config.k_rl:
            return 0
        required_rollout_count = self.mastery_config.k_rl - num_running_rl
        return min(required_rollout_count, len(self.qras))

    def pop_rollout_qra(self) -> tuple[InternalizationQRA, str]:
        qra = self.qras.pop(0)
        task = RLTask(
            knowledge_id=self.knowledge.id,
            instruction=qra.question,
        )
        self.running_rl[task.id] = task
        return qra, task.id

    def push_rollout_result(self, result: GroupRolloutResult) -> None:
        self.running_rl.pop(result.task_id)
        new_group = UsedGroupRolloutResult.from_group_result(result)
        self.groups.append(new_group)

    def report_rollout_failure(self, task_id: str) -> None:
        if task_id in self.running_rl:
            self.running_rl.pop(task_id)

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
    last_reported_version: int = -1
    total_rollout_groups_completed: int = 0
    total_rollout_groups_successful: int = 0

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
        qra, task_id = knowledge.pop_rollout_qra()

        return InternalizationTask(
            id=task_id,
            knowledge_id=knowledge_id,
            instruction=qra.question,
        )

    def push_rollout_result(self, result: GroupRolloutResult) -> None:
        self.total_rollout_groups_completed += 1
        if any(r.success for r in result.trajectories):
            self.total_rollout_groups_successful += 1

        knowledge = self.knowledges[result.knowledge_id]
        knowledge.push_rollout_result(result)

    def report_rollout_failure(self, knowledge_id: str, task_id: str) -> None:
        if knowledge_id in self.knowledges:
            self.knowledges[knowledge_id].report_rollout_failure(task_id)

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

    def get_new_version_stats(
        self, current_version: int
    ) -> dict[int, dict[str, float]]:
        """
        Calculate metrics for all versions between last_reported_version and current_version.
        """
        version_to_groups: dict[int, list[UsedGroupRolloutResult]] = {}
        for k in self.knowledges.values():
            for g in k.groups:
                v = g.sampling_version
                if self.last_reported_version < v < current_version:
                    if v not in version_to_groups:
                        version_to_groups[v] = []
                    version_to_groups[v].append(g)

        stats: dict[int, dict[str, float]] = {}
        for v, groups in version_to_groups.items():
            if not groups:
                continue

            # Ratio of tasks with at least one success
            success_tasks = sum(
                1.0 for g in groups if any(r.success for r in g.trajectories)
            )
            task_success_ratio = success_tasks / len(groups)

            # Total success rate across all rollouts
            total_trajectories = sum(len(g.trajectories) for g in groups)
            total_successes = sum(
                1.0 for g in groups for r in g.trajectories if r.success
            )
            total_success_rate = (
                total_successes / total_trajectories if total_trajectories > 0 else 0.0
            )

            stats[v] = {
                "overall/task_success_ratio": task_success_ratio,
                "overall/total_success_rate": total_success_rate,
            }

        if stats:
            self.last_reported_version = max(stats.keys())

        return stats

    def get_detailed_status(self) -> dict[str, float]:
        """
        Calculate and return total status metrics.
        """
        active_qras = sum(k.running_qra_generation for k in self.knowledges.values())
        active_rollouts = sum(len(k.running_rl) for k in self.knowledges.values())
        sft_ready = sum(k.sft_ready_count() for k in self.knowledges.values())
        rl_ready = sum(k.rl_ready_count() for k in self.knowledges.values())
        qra_plan = sum(k.get_required_qra() for k in self.knowledges.values())

        mastery_counts = {"studying": 0, "practicing": 0, "mastered": 0}
        for k in self.knowledges.values():
            m = k.mastery
            if m in mastery_counts:
                mastery_counts[m] += 1

        return {
            "status/active_qra_generations": float(active_qras),
            "status/active_rollouts": float(active_rollouts),
            "status/sft_ready_qras": float(sft_ready),
            "status/rl_ready_groups": float(rl_ready),
            "status/qra_generation_plan_total": float(qra_plan),
            "status/total_rollout_groups_completed": float(
                self.total_rollout_groups_completed
            ),
            "status/total_rollout_groups_successful": float(
                self.total_rollout_groups_successful
            ),
            "status/studying_count": float(mastery_counts["studying"]),
            "status/practicing_count": float(mastery_counts["practicing"]),
            "status/mastered_count": float(mastery_counts["mastered"]),
        }


@dataclass
class GlobalState:
    sampling_client: SharedSamplingClient
    knowledge_manager: KnowledgeMasteryManager
    ml_logger: MLLogger = None  # type: ignore

    def update_sampling_client(self, sampling_client: SamplingClient) -> None:
        self.sampling_client.update_client(sampling_client)
        # Automatically log stats for the versions completed after this update
        self.log_version_stats()

    async def setup_logging(
        self, log_dir: str, wandb_project: Optional[str], config: Optional[dict]
    ) -> None:
        setup_base_loglevel()
        self.ml_logger = ml_log.setup_logging(
            log_dir=log_dir,
            wandb_project=wandb_project,
            config=config,
        )

    async def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        self.ml_logger.log_metrics(metrics, step=step)

    async def log_hparams(self, config: dict) -> None:
        self.ml_logger.log_hparams(config)

    @ray.method
    def get_current_version(self) -> int:
        return self.sampling_client.version

    def log_version_stats(self) -> None:
        stats = self.get_new_version_stats()
        for version, s in stats.items():
            self.ml_logger.log_metrics(s, step=version)

    def report_detailed_status(self) -> None:
        if self.ml_logger:
            version = self.sampling_client.version
            metrics = self.knowledge_manager.get_detailed_status()
            self.ml_logger.log_metrics(metrics, step=version)

    def __repr__(self) -> str:
        return f"GlobalState(v={self.sampling_client.version})"

    def get_sampling_client(self) -> IndexedSamplingClient:
        return self.sampling_client.get_client()

    def get_replenishment_plan(self) -> list[tuple[Knowledge, int]]:
        """Calculate how many QRAs to generate for each underperforming knowledge."""
        return self.knowledge_manager.get_replenishment_plan()

    def get_status_summary(self) -> dict[str, float]:
        """Return a mapping of knowledge_id to current success ratio."""
        return self.knowledge_manager.get_status_summary()

    def get_new_version_stats(self) -> dict[int, dict[str, float]]:
        current_version = self.sampling_client.version
        return self.knowledge_manager.get_new_version_stats(current_version)

    def report_qra_generation_result(
        self, knowledge_id: str, qra: Optional[QRA] = None
    ) -> None:
        """Report results of a generation task and end its lifecycle."""
        if qra:
            self.knowledge_manager.push_qra(knowledge_id, qra)
        self.knowledge_manager.report_qra_generation_end(knowledge_id)

    def report_qra_generation_start(self, knowledge_id: str) -> None:
        self.knowledge_manager.report_qra_generation_start(knowledge_id)

    # --- Rollout Interfaces ---
    def pop_rollout_task(self) -> InternalizationTask | None:
        """Get a task from the queue. Called by RemoteWorkers."""
        return self.knowledge_manager.pop_rollout_task()

    def pop_rl_batch(self, min_batch_size: int) -> list[RLGroup] | None:
        """Returns results only if the queue size exceeds min_batch_size."""
        return self.knowledge_manager.pop_rl_batch(min_batch_size)

    def pop_sft_batch(self, min_batch_size: int) -> list[InternalizationQRA] | None:
        return self.knowledge_manager.pop_sft_batch(min_batch_size)

    async def report_sft_results(self, qras: list[InternalizationQRA]) -> None:
        self.knowledge_manager.report_sft_results(qras)

    async def push_rollout_result(self, result: GroupRolloutResult) -> None:
        self.knowledge_manager.push_rollout_result(result)

    @ray.method
    def report_rollout_failure(self, knowledge_id: str, task_id: str) -> None:
        self.knowledge_manager.report_rollout_failure(knowledge_id, task_id)
