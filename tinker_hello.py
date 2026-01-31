"""
Implements RL on general MDPs
"""

import asyncio
import io
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence, TypeVar

import chz
import numpy as np
import tinker
import torch
from agents import ModelSettings
from oai_utils.agent import AgentWrapper
from oai_utils.tinker.litellm_model import TinkerLLM, result_to_trajectory
from oai_utils.tinker.model_with_logprob import LogprobLitellmModel
from tinker.types import LossFnType
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import (
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
)
from tinker_cookbook.recipes.math_rl.math_env import (
    Gsm8kDataset,
    Gsm8kDatasetBuilder,
    MathEnv,
)
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import (
    RLTestSetEvaluator,
    compute_trajectory_metrics,
)
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import logtree, ml_log
from tinker_cookbook.utils.misc_utils import all_same, safezip, split_list, timed
from tinker_cookbook.utils.trace import scope, trace_init, update_scope_context

logger = logging.getLogger(__name__)

T = TypeVar("T")


@chz.chz
class KLReferenceConfig:
    """Configuration for the KL penalty reference model.

    If not specified in Config, the training model's base model is used.
    """

    base_model: str
    load_checkpoint_path: str | None = None


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    """
    Creates a context manager; all log inside this context will be logged under the section `scope_name`.
    It will create a file with the path of log_path/f_name.html
    If num_groups_to_log is 0, it will disable logging (but note that this function does not actually implement the logic for logging itself!)
    """
    if log_path is not None and num_groups_to_log > 0:
        logtree_path = os.path.join(log_path, f"{f_name}.html")
        with logtree.init_trace(scope_name, path=logtree_path):
            yield
    else:
        yield


@scope
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


@scope
def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    """
    Print a subset of the trajectory group to the console.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(
            traj_group.get_total_rewards(), max_trajs_to_print
        )
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    @scope
    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in fwd_bwd_result.loss_fn_outputs]


@scope
async def train_step(
    data_D: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """Train the model on collected trajectories.

    Pipelines forward_backward and optim_step so they land on the same clock cycle.
    """
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []

    adam_params = tinker.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    training_logprobs_D: list[torch.Tensor] = []
    optim_result: tinker.OptimStepResponse | None = None

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]],
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
        optim_result = await optim_future.result_async()
        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
            optim_future = next_optim_future

    if metrics is not None and optim_result is not None and optim_result.metrics:
        metrics.update(optim_result.metrics)

    return training_logprobs_D


@chz.chz
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    temperature: float = 1.0  # Changing sampling temperature is not generally recommended; does not currently play well with KL penalty
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(
        default_factory=list
    )
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    kl_reference_config: KLReferenceConfig | None = (
        None  # If None, uses base model_name
    )

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    eval_every: int = 20  # 0 = disabled
    save_every: int = 20  # 0 = disabled
    ttl_seconds: int = 604800  # 7 days
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = None

    # Logtree configuration
    num_groups_to_log: int = (
        4  # Number of groups to log per iteration (0 = disable logging)
    )


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements async off-policy training, capped at K steps off policy."""
    assert cfg.async_config is not None

    shutdown_event = asyncio.Event()
    # We will have groups_per_batch worker generating rollouts, so cap the
    # queue size to be groups_per_batch.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None](
        maxsize=cfg.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()

    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{start_batch:06d}",
        log_path=cfg.log_path,
        loop_state={"batch": start_batch},
        kind="both",
        ttl_seconds=cfg.ttl_seconds,
    )

    # This will be updated by the training loop
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    sampling_client_step = start_batch
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    @scope
    def shutdown_loops():
        """Trigger all loops to shutdown"""
        shutdown_event.set()
        assert cfg.async_config is not None
        for _ in range(cfg.async_config.groups_per_batch):
            env_group_builders_queue.put_nowait(None)
        sampling_client_updated_event.set()

    @scope
    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        i_batch = start_batch
        while not shutdown_event.is_set() and i_batch < end_batch:
            env_group_builders_P = dataset.get_batch(i_batch)
            for env_group_builder in env_group_builders_P:
                await env_group_builders_queue.put(env_group_builder)
            i_batch += 1

    @scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        while not shutdown_event.is_set():
            env_group_builder = await env_group_builders_queue.get()
            if env_group_builder is None:
                break

            metrics = {}
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step

            trajectory_group = await do_agent_rollout_and_filter_constant_reward(
                env_group_builder,
                tokenizer,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                model_name=cfg.model_name,
                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
            )
            if trajectory_group is None:
                trajectory_groups_queue.put_nowait(None)
            else:
                metrics["time/trajectory_group_worker_loop/total"] = (
                    time.time() - t_start
                )
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=metrics,
                    )
                )

    @scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None

        i_batch = start_batch
        wrapped_trajectory_groups = []
        while i_batch < end_batch:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if wrapped_trajectory_group is None:
                continue

            @scope
            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                if wrapped_trajectory_group is None:
                    return False

                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > cfg.async_config.max_steps_off_policy
                ):
                    logger.info(
                        f"[training_loop] Step {i_batch}: Samples are too stale, skipping"
                    )
                    asyncio.create_task(
                        env_group_builders_queue.put(
                            wrapped_trajectory_group.env_group_builder
                        ),
                        name="requeue_stale_sample_task",
                    )
                    return False
                return True

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            nonlocal sampling_client
            nonlocal sampling_client_step

            if not filter_stale_trajectory_group(wrapped_trajectory_group):
                continue

            # Dynamic sampling: Wait for enough trajectories to accumulate to
            # ensure all batch sizes are the same size. This avoids needing to adjust
            # the learning rate for different batch sizes.
            wrapped_trajectory_groups.append(wrapped_trajectory_group)
            if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                continue
            logger.info(
                f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Compute sampling client metrics, as samples may have been generated with
            # different sampler versions
            metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

            # TODO: For proper checkpointing, we also need to save dataloader state and
            # all queued trajectory groups that haven't been trained on yet
            (
                sampling_client,
                train_step_metrics,
            ) = await do_train_step_and_get_sampling_client(
                cfg,
                i_batch,
                training_client,
                service_client,
                tokenizer,
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
            )
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/training_loop/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            i_batch += 1
            wrapped_trajectory_groups = []

        shutdown_loops()

    @scope
    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or cfg.eval_every == 0:
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if cfg.eval_every > 0 and sampling_client_eval_step % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client_eval)
                        metrics.update(
                            {f"test/{k}": v for k, v in eval_metrics.items()}
                        )
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    await asyncio.gather(
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(cfg.async_config.groups_per_batch)
        ],
        asyncio.create_task(training_loop(), name="training_loop"),
        asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
    )


@scope
async def do_agent_rollout_and_filter_constant_reward(
    env_group_builder: EnvGroupBuilder,
    tokenizer: Tokenizer,
    max_tokens: int,
    temperature: float,
    model_name: str,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
) -> TrajectoryGroup | None:
    # We use a dummy model name because the actual routing is handled by litellm + TinkerLLM custom provider
    litellm_model_name = f"agl-tinker/{model_name}"
    model = LogprobLitellmModel(model=litellm_model_name)

    async def run_single_agent(env: Env) -> Trajectory:
        ob, _ = await env.initial_observation()

        prompt_text = ""
        if isinstance(ob, tinker.ModelInput):
            prompt_text = tokenizer.decode(ob.to_ints())
        else:
            prompt_text = tokenizer.decode(ob)

        agent_wrapper = AgentWrapper[str].create(
            name="RolloutAgent",
            instructions="You are a helpful assistant.",
            model=model,
            model_settings=ModelSettings(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        result = await agent_wrapper.run(prompt_text)
        return result_to_trajectory(result)

    with logtree.optional_enable_logging(enable_logging):
        envs_G = await env_group_builder.make_envs()

        trajectories_G = await asyncio.gather(
            *[run_single_agent(env) for env in envs_G]
        )

        rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
            trajectories_G, envs_G
        )
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

        trajectory_group = TrajectoryGroup(
            list(trajectories_G), list(rewards_G), list(metrics_G)
        )

    if do_remove_constant_reward_groups and all_same(
        trajectory_group.get_total_rewards()
    ):
        return None
    else:
        return trajectory_group


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
    ttl_seconds: int | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
                ttl_seconds=ttl_seconds,
            )
            return training_client.create_sampling_client(
                path_dict["sampler_path"]
            ), metrics
        else:
            return (
                await training_client.save_weights_and_get_sampling_client_async(),
                metrics,
            )


def create_kl_reference_client(
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_reference_config: KLReferenceConfig | None,
) -> tinker.SamplingClient:
    """Create a sampling client for KL penalty computation.

    If kl_reference_config is None, uses the base model_name.
    If kl_reference_config is provided, uses its base_model and optionally load_checkpoint_path.
    """
    if kl_reference_config is None:
        return service_client.create_sampling_client(base_model=model_name)

    if kl_reference_config.load_checkpoint_path is not None:
        return service_client.create_sampling_client(
            base_model=kl_reference_config.base_model,
            model_path=kl_reference_config.load_checkpoint_path,
        )
    else:
        return service_client.create_sampling_client(
            base_model=kl_reference_config.base_model
        )


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    kl_reference_config: KLReferenceConfig | None = None,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    metrics = {}
    taglist_P = [
        env_group_builder.logging_tags() for env_group_builder in env_group_builders_P
    ]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_reference_client = create_kl_reference_client(
                service_client, model_name, kl_reference_config
            )
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                kl_reference_client,
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    log_path: str,
    save_every: int,
    do_compute_post_kl: bool,
    ttl_seconds: int | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.

    The reason we return a sampling client is that if do_compute_post_kl is True,
    we need to create a sampling client from the post-update policy.
    """
    metrics = {}

    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client, i_batch, log_path, save_every, ttl_seconds=ttl_seconds
    )
    metrics.update(checkpoint_metrics)

    if do_compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        kl_reference_config=cfg.kl_reference_config,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=metrics,
        )

    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
        cfg.ttl_seconds,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(
            f"Tracing is enabled. Trace events will be saved to {trace_events_path}"
        )
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    # Need to register TinkerLLM provider for AgentWrapper
    # We create a dummy TinkerLLM just to register the provider.
    # The actual sampling client is passed per-request.
    tokenizer = training_client.get_tokenizer()

    try:
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        base_sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name
        )

        tinker_llm = TinkerLLM(
            model_name=cfg.model_name,
            renderer=renderer,
            tokenizer=tokenizer,
            sampling_client=base_sampling_client,
            max_tokens=cfg.max_tokens,
        )
        tinker_llm.rewrite_litellm_custom_providers()
        logger.info("Registered TinkerLLM provider for AgentWrapper")
    except Exception as e:
        logger.error(f"Failed to register TinkerLLM provider: {e}")

        pass

    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(
            RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens)
        )

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_async_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=cfg.ttl_seconds,
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


# --- Reward Calculation Implementation ---
@dataclass(frozen=True)
class ComputingProblemGroupBuilder(ProblemGroupBuilder):
    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        results = []
        for trajectory, env in safezip(trajectory_group, env_group):
            # We assume the env is a ProblemEnv which has a renderer with a tokenizer
            assert isinstance(env, ProblemEnv)
            tokenizer = env.renderer.tokenizer

            # Extract the final answer from the trajectory
            model_text = ""
            if trajectory.final_ob.chunks:
                last_chunk = trajectory.final_ob.chunks[-1]
                if isinstance(last_chunk, tinker.types.EncodedTextChunk):
                    model_text = tokenizer.decode(list(last_chunk.tokens))

            # Calculate reward
            correct_format = float(env.check_format(model_text))
            correct_answer = float(env.check_answer(model_text))
            total_reward = env.format_coef * (correct_format - 1) + correct_answer

            results.append(
                (total_reward, {"format": correct_format, "correct": correct_answer})
            )

        return results


@chz.chz
class ComputingGsm8kDatasetBuilder(Gsm8kDatasetBuilder):
    # Redeclare fields to ensure chz handles them correctly if inheritance is wonky
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[Gsm8kDataset, Gsm8kDataset]:
        # Local definition to override _make_env_group_builder
        class ComputingGsm8kDataset(Gsm8kDataset):
            def _make_env_group_builder(
                self, x: dict[str, str], group_size: int
            ) -> ProblemGroupBuilder | None:
                builder = super()._make_env_group_builder(x, group_size)
                if builder is None:
                    return None
                # Return our new builder with the same params
                return ComputingProblemGroupBuilder(
                    env_thunk=builder.env_thunk,
                    num_envs=builder.num_envs,
                    dataset_name=builder.dataset_name,
                )

        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        datasets = [
            ComputingGsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


def build_config_blueprint() -> chz.Blueprint[Config]:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    except Exception:
        # Fallback if not found in model_info
        renderer_name = "llama3"

    builder = ComputingGsm8kDatasetBuilder(
        batch_size=128,
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )

    return chz.Blueprint(
        Config
    ).apply(
        {
            "model_name": model_name,
            "log_path": "/tmp/tinker-examples/rl_basic",
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 256,
            "eval_every": 0,
            "async_config": AsyncConfig(
                max_steps_off_policy=100,
                groups_per_batch=4,  # Matches wrapped_trajectory_groups check in training_loop
            ),
            "save_every": 0,
            "wandb_project": "tinker_hello",
        }
    )


if __name__ == "__main__":
    import os
    import sys

    blueprint = build_config_blueprint()
    # Allow overriding from CLI
    blueprint.make_from_argv(sys.argv[1:])
    config = blueprint.make()

    log_path = os.path.expanduser(config.log_path)
    if os.path.exists(log_path):
        print(f"Warning: Log directory {log_path} already exists.")
    os.makedirs(log_path, exist_ok=True)

    asyncio.run(main(config))
