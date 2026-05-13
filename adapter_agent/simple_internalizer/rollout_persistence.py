"""Shared rollout-audit helpers.

Two pieces of glue used by both the RL training loop and the evaluation
worker:

  - `rollout_batch_to_rl_group`: RolloutBatch (engine output) → RLGroup
    (pipeline-internal record with audit metadata + GRPO-ready trajectories).
    Same shape on both code paths so `simple_rl_rollouts` rows are uniform.

  - `persist_rl_groups`: bulk-write the audit rows for one batch of RLGroups.
    Wrapped in try/except so a DB failure logs a warning but never kills
    training. Used identically from the GRPO update path and the eval worker.
"""

from __future__ import annotations

import logging

from prisma import Prisma
from tinker_cookbook.rl.types import TokensWithLogprobs, Trajectory, Transition

from adapter_agent.hierarchical.state import RLGroup, RolloutSample

from .rollout_engine import RolloutBatch

logger = logging.getLogger(__name__)


def rollout_batch_to_rl_group(
    batch: RolloutBatch,
    *,
    suite_name: str,
    task_id: str,
    instruction: str,
    sampling_client_version: int,
) -> RLGroup:
    """Convert one engine RolloutBatch into an RLGroup with audit metadata.

    The reward shape mirrors the RL pool's mapping (1.0 on success, 0.0
    otherwise). For eval-only callers the rewards never hit GRPO; they're
    only persisted as audit info.
    """
    trajectories: list[Trajectory] = []
    rewards: list[float] = []
    samples: list[RolloutSample] = []
    for o in batch.outcomes:
        ac = TokensWithLogprobs(tokens=o.tokens, maybe_logprobs=o.logprobs)
        trajectories.append(
            Trajectory(
                transitions=[
                    Transition(
                        ob=batch.prompt, ac=ac, reward=0.0, episode_done=True
                    )
                ],
                final_ob=batch.prompt,
            )
        )
        rewards.append(1.0 if o.success else 0.0)
        samples.append(
            RolloutSample(
                answer=o.answer,
                reasoning=o.reasoning,
                parsed=o.parsed,
                success=o.success,
                execution_output=o.execution_output,
                verification_output=o.verification_output,
            )
        )
    return RLGroup(
        trajectories=trajectories,
        rewards=rewards,
        sampling_client_version=sampling_client_version,
        suite_name=suite_name,
        task_id=task_id,
        instruction=instruction,
        samples=samples,
    )


async def persist_rl_groups(
    prisma_client: Prisma,
    *,
    simple_train_id: str,
    rl_step: int,
    groups: list[RLGroup],
) -> None:
    """Bulk-write rollouts to the `simple_rl_rollouts` table for later audit.

    One `create_many` call per batch. Wrapped in try/except so a DB failure
    logs a warning but never kills training. Skips groups that lack audit
    metadata (e.g. Ray rollouts — TODO).
    """
    records: list[dict] = []
    for gi, g in enumerate(groups):
        if (
            g.samples is None
            or g.instruction is None
            or g.suite_name is None
            or g.task_id is None
        ):
            continue
        for si, (sample, reward) in enumerate(zip(g.samples, g.rewards)):
            records.append(
                {
                    "simple_train_id": simple_train_id,
                    "rl_step": rl_step,
                    "suite_name": g.suite_name,
                    "task_id": g.task_id,
                    "group_idx": gi,
                    "sample_idx": si,
                    "num_samples": len(g.samples),
                    "instruction": g.instruction,
                    "answer": sample.answer,
                    "reasoning": sample.reasoning,
                    "parsed": sample.parsed,
                    "success": sample.success,
                    "reward": float(reward),
                    "execution_output": sample.execution_output,
                    "verification_output": sample.verification_output,
                    "sampling_client_version": g.sampling_client_version,
                }
            )
    if not records:
        return
    try:
        await prisma_client.simplerlrollout.create_many(data=records)
    except Exception as e:
        logger.warning(
            f"Failed to persist {len(records)} RL rollouts at step {rl_step}: {e}"
        )
