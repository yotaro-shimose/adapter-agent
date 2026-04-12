from typing import Literal

import torch
from tinker_cookbook.rl.data_processing import Trajectory, TrajectoryGroup

type AdvantageRegularizer = float | Literal["output_token", "group_std"]


def get_traj_output_token_count(trajectory: Trajectory) -> int:
    return sum(len(transition.ac.tokens) for transition in trajectory.transitions)


def compute_advantages(
    trajectory_groups_P: list[TrajectoryGroup],
    regularization: AdvantageRegularizer = "group_std",
) -> list[torch.Tensor]:
    """Compute advantages for each trajectory, centered and standardized within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards(), dtype=torch.float32)

        if regularization == "group_std":
            # Center and standardize within the group (standard GRPO)
            mean_R = rewards_G.mean()
            std_R = rewards_G.std()
            advantages_G = (rewards_G - mean_R) / (std_R + 1e-8)
        elif regularization == "output_token":
            # Backward compatibility or manual length bias correction
            traj_token_counts = torch.tensor(
                [get_traj_output_token_count(traj) for traj in traj_group.trajectories_G],
                dtype=torch.float32,
            )
            advantages_G = (rewards_G - rewards_G.mean()) / (traj_token_counts + 1e-8)
        elif isinstance(regularization, float):
            advantages_G = (rewards_G - rewards_G.mean()) / regularization
        else:
            raise ValueError(f"Invalid regularization: {regularization}")

        advantages_P.append(advantages_G)

    return advantages_P
