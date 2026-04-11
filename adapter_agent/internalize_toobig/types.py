from dataclasses import dataclass
from typing import Literal, Self

from tinker_cookbook.rl import Trajectory

from adapter_agent.data import QRA
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Entity, Task

type Mastery = Literal["studying", "practicing", "mastered"]


class InternalizationTask(Task):
    knowledge_id: str


class InternalizationQRA(QRA, Entity):
    knowledge_id: str


@dataclass
class SingleRolloutResult:
    question: str
    reasoning: str
    answer: str
    execution_output: str
    main_rs_content: str
    success: bool
    verification_reasoning: str
    trajectory: Trajectory

    @classmethod
    def parse_failed(cls, question: str, trajectory: Trajectory) -> Self:
        return cls(
            question=question,
            reasoning="Parse Error",
            answer="N/A",
            execution_output="The model output could not be parsed as a valid message.",
            main_rs_content="",
            success=False,
            verification_reasoning="Parse failure.",
            trajectory=trajectory,
        )


@dataclass(kw_only=True)
class GroupRolloutResult:
    knowledge_id: str
    task_id: str
    trajectories: list[SingleRolloutResult]
    current_sampling_version: int

    def to_rlgroup(self) -> RLGroup:
        """Transform this rollout group into an RLGroup for GRPO training."""
        trajectories = [res.trajectory for res in self.trajectories]
        # Binary rewards based on success
        rewards = [1.0 if res.success else 0.0 for res in self.trajectories]
        return RLGroup(trajectories=trajectories, rewards=rewards)


class RLTask(Entity):
    knowledge_id: str
    instruction: str

    def as_task(self) -> Task:
        return Task(id=self.id, instruction=self.instruction)


@dataclass
class MasteryConfig:
    studying_threshold: float
    success_threshold: float
    overgen_factor: float
    k_sft: int  # number of SFT samples in a batch
    k_rl: int  # number of RL tasks in a batch (The resulting batch will include samples k_rl * rollout_per_tasks)
