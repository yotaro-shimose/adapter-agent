from typing import Literal

import torch
from tinker_cookbook.rl.data_processing import Trajectory, TrajectoryGroup

type AdvantageRegularizer = float | Literal["output_token"]


def get_traj_output_token_count(trajectory: Trajectory) -> int:
    return sum(len(transition.ac.tokens) for transition in trajectory.transitions)


def compute_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    regularization: AdvantageRegularizer = "output_token",
) -> list[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        traj_token_counts = torch.tensor(
            [get_traj_output_token_count(traj) for traj in traj_group.trajectories_G]
        )
        if regularization == "output_token":
            # Normalize by the mean token count of the group to avoid length bias
            regularizer = traj_token_counts
        elif isinstance(regularization, float):
            regularizer = torch.tensor(regularization)
        else:
            raise ValueError(f"Invalid regularization: {regularization}")

        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = (rewards_G - rewards_G.mean()) / regularizer
        advantages_P.append(advantages_G)

    return advantages_P
