import logging

import tinker
from tinker_cookbook.rl.types import TrajectoryGroup

from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.rl.config import OptimizerParams
from adapter_agent.rl.trajectory import prepare_minibatch_simplified

logger = logging.getLogger(__name__)


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    """
    Remove the 'mask' field from loss_fn_inputs as it often causes 400 errors
    with certain server-side loss functions like 'importance_sampling' or 'ppo'.
    """
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


async def compute_grpo_loss(
    groups: list[RLGroup],
    training_client: tinker.TrainingClient,
    optimizer_params: OptimizerParams,
    kl_reference_client: tinker.SamplingClient | None = None,
):
    """
    Computes GRPO-like loss for a list of RLGroups.
    Uses native adapter_agent trajectory processing for consistency.
    """
    if not groups:
        return None

    # Convert RLGroup to TrajectoryGroup
    traj_groups = []
    for g in groups:
        metrics_G = [{} for _ in g.rewards]
        traj_groups.append(
            TrajectoryGroup(
                trajectories_G=g.trajectories,
                final_rewards_G=g.rewards,
                metrics_G=metrics_G,
            )
        )

    # Use native prepare_minibatch_simplified
    # It handles internal advantage calculation and data assembly.
    data_D, _metrics = await prepare_minibatch_simplified(
        trajectory_groups=traj_groups,
        regularization=optimizer_params.advantage_regularizer,
        kl_reference_client=kl_reference_client,
        kl_penalty_coef=optimizer_params.kl_penalty_coef,
        kl_discount_factor=optimizer_params.kl_discount_factor,
    )

    if not data_D:
        logger.warning("No valid training data assembled for GRPO step.")
        return None

    logger.info(
        f"Computing GRPO step with {len(data_D)} datums using loss_fn={optimizer_params.loss_fn}."
    )

    # Strip mask to avoid 400 Bad Request on the server
    clean_data_D = [_remove_mask(d) for d in data_D]

    # Call training client
    fwd_bwd_future = await training_client.forward_backward_async(
        data=clean_data_D, loss_fn=optimizer_params.loss_fn
    )
    result = await fwd_bwd_future.result_async()

    # Merge KL metrics into the result
    if _metrics:
        result.metrics.update(_metrics)

    return result
