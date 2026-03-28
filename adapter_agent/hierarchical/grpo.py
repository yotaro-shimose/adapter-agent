import logging

import tinker
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.types import TrajectoryGroup

from adapter_agent.hierarchical.state import RLGroup

logger = logging.getLogger(__name__)


async def compute_grpo_loss(
    groups: list[RLGroup], training_client: tinker.TrainingClient
):
    """
    Computes GRPO-like loss (using importance_sampling on server) for a list of RLGroups.
    Prepares data using tinker_cookbook utilities.
    """
    if not groups:
        return 0.0

    # Convert RLGroup to TrajectoryGroup
    traj_groups = []
    for g in groups:
        # RLGroup has trajectories and rewards.
        # TrajectoryGroup expects metrics_G as well.
        metrics_G = [{} for _ in g.rewards]
        traj_groups.append(
            TrajectoryGroup(
                trajectories_G=g.trajectories,
                final_rewards_G=g.rewards,
                metrics_G=metrics_G,
            )
        )

    # Use tinker_cookbook to compute advantages and assemble data
    # This ensures compatibility with "importance_sampling" loss on server
    advantages = compute_advantages(traj_groups)
    data_D, _ = assemble_training_data(traj_groups, advantages)

    if not data_D:
        logger.warning("No valid training data assembled for GRPO step.")
        return 0.0

    logger.info(f"Computing GRPO (importance_sampling) step with {len(data_D)} datums.")

    # Call training client with importance_sampling
    fwd_bwd_future = await training_client.forward_backward_async(
        data=data_D, loss_fn="importance_sampling"
    )
    result = await fwd_bwd_future.result_async()
    return result
