import tinker
import torch
from tinker import TensorData
from tinker_cookbook.rl.data_processing import (
    FlatObElem,
    _flat_ob_to_model_input,
    _flat_ob_token_len,
    _flatten_chunks,
    _is_prefix,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import safezip

from adapter_agent.rl.advantage import AdvantageRegularizer, compute_advantages


def trajectory_to_data(
    traj: Trajectory,
    traj_advantage: float,
) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = (
            create_rightshifted_model_input_and_leftshifted_targets(
                list(all_tokens_T.chunks)
            )
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        # Normalize advantages by the number of output tokens
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(
                    torch.tensor(advantages_T) / 1000.0
                ),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend(
            [0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens)
        )

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    # TODO: fixthis

    return data[-1:]


def assemble_training_data(
    trajectory_groups_P: list[TrajectoryGroup],
    advantages_P: list[torch.Tensor],
) -> tuple[list[tinker.Datum], list[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory and normalize by length
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend(
                [dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data]
            )

    return data_D, metadata_D


def prepare_minibatch_simplified(
    trajectory_groups: list[TrajectoryGroup],
    regularization: AdvantageRegularizer = "output_token",
):
    trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if not trajectory_groups:
        return []

    advantages_G = compute_advantages(trajectory_groups, regularization)

    # Assert any of the advantages are non-zero
    all_advantages = torch.cat(advantages_G)
    assert torch.any(all_advantages != 0), "All advantages are zero!"

    data_D, _metadata_D = assemble_training_data(trajectory_groups, advantages_G)
    return data_D
