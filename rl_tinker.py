import weave  # noqa: F401
import asyncio
import csv
import logging
import threading
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
import tinker
import tinker_cookbook.checkpoint_utils
import torch
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import LogprobLitellmModel, setup_tinkermodel
from tinker import SamplingClient, TrainingClient
from tinker.types.loss_fn_type import LossFnType
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger
from tinker_cookbook.utils.trace import scope

from adapter_agent.hierarchical.agent.simplified_solver import SimplifiedSolver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.gh import Library
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.state import SFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from adapter_agent.rl.config import (
    EnvParams,
    ExperimentSettings,
    ModelLoadingSettings,
    OptimizerParams,
    RLConfig,
    RolloutParams,
)
from adapter_agent.rl.trajectory import prepare_minibatch_simplified
from adapter_agent.util.logger_util import setup_base_loglevel

litellm.add_function_to_prompt

logger = logging.getLogger(__name__)


# Filter out specific openai.agents warning
class OpenAITracingFilter(logging.Filter):
    def filter(self, record):
        if (
            record.name == "openai.agents"
            and "Tracing client error 400" in record.getMessage()
        ):
            return False
        return True


# logging.getLogger("openai.agents").addFilter(OpenAITracingFilter())


class SharedSamplingClient:
    def __init__(self, client: SamplingClient):
        self._client = client
        self._lock = asyncio.Lock()

    async def get_client(self) -> SamplingClient:
        async with self._lock:
            return self._client

    async def update_client(self, new_client: SamplingClient) -> None:
        async with self._lock:
            self._client = new_client


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


@dataclass
class ResultRecord:
    original_question: str
    original_answer: str
    generated_question: str
    generated_answer: str
    cause: str
    reward: float
    has_trajectory: bool
    sampler_id: int


@dataclass
class RLState:
    queue_questions: asyncio.Queue[QA]
    queue_trajectories: asyncio.Queue[TrajectoryGroup]
    sampling_client_manager: SharedSamplingClient
    all_results: list[ResultRecord]
    results_lock: threading.Lock
    csv_output_path: Path
    rust_doc_analyzer: RustDocAnalyzer
    litellm_model_name: str
    training_client: TrainingClient
    step_counter: int = 0

    def get_latest_model(self) -> LogprobLitellmModel:
        current_model = LogprobLitellmModel(
            model=self.litellm_model_name,
            sampling_client=self.sampling_client_manager._client,
        )
        return current_model

    def get_latest_solver(self) -> SimplifiedSolver[LogprobLitellmModel]:
        return SimplifiedSolver[LogprobLitellmModel](
            model=self.get_latest_model(),
            rust_doc_analyzer=self.rust_doc_analyzer,
        )


def _save_results_csv(results: list[ResultRecord], path: Path) -> None:
    """Save results to CSV, overwriting the file each time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ResultRecord)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in results:
            row = {fn: getattr(record, fn) for fn in fieldnames}
            writer.writerow(row)


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in fwd_bwd_result.loss_fn_outputs]


@scope
async def train_step(
    data_D: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    adam_params: tinker.AdamParams,
    num_steps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, int | float] | None = None,
) -> list[torch.Tensor]:
    """Train the model on collected trajectories.

    Uses the whole batch and updates the model num_steps times.
    Pipelines forward_backward and optim_step so they land on the same clock cycle.
    """
    if not data_D:
        return []

    training_logprobs_D: list[torch.Tensor] = []
    optim_result: tinker.OptimStepResponse | None = None
    fwd_bwd_result: tinker.ForwardBackwardOutput | None = None

    # Enqueue first step
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in data_D],
        loss_fn=loss_fn,
        loss_fn_config=loss_fn_config,
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(num_steps):
        # Enqueue next step before consuming current results (to stay on same clock cycle)
        if i + 1 < num_steps:
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in data_D],
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
    if metrics is not None and fwd_bwd_result is not None and fwd_bwd_result.metrics:
        metrics.update(fwd_bwd_result.metrics)

    return training_logprobs_D


def uniform_reward(rewards: list[float]) -> bool:
    return all(r == rewards[0] for r in rewards)


def compute_batch_metrics(
    unfiltered_traj_groups: list[TrajectoryGroup],
) -> dict[str, Any]:
    all_rewards = []
    for group in unfiltered_traj_groups:
        all_rewards.extend(group.final_rewards_G)

    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    success_count = sum(1 for r in all_rewards if r > 0)
    success_ratio = success_count / len(all_rewards) if all_rewards else 0.0

    return {
        "train/mean_reward": mean_reward,
        "train/success_ratio": success_ratio,
        "train/num_groups": len(unfiltered_traj_groups),
    }


async def rollout_worker(
    worker_id: int,
    rl_state: RLState,
    verifier: Verifier,
    cfg: RLConfig,
):
    logger.info(f"Rollout Worker {worker_id} started.")
    while True:
        try:
            qa_item = rl_state.queue_questions.get_nowait()
        except asyncio.QueueEmpty:
            break

        question_text = qa_item.question

        # Create new model and solver for this run
        solver = rl_state.get_latest_solver()
        task = Task.from_instruction(question_text)
        results = await gather_with_semaphore(
            [
                solve_verify(
                    solver=solver,
                    verifier=verifier,
                    task=task,
                    image_name=cfg.env_params.image_name,
                    library_name=cfg.env_params.library.name,
                    max_turns=cfg.env_params.max_turns,
                    collect_trajectory=True,
                    use_search=False,
                )
                for _ in range(cfg.rollout_params.rollouts_per_question)
            ],
            max_concurrent=cfg.rollout_params.per_group_concurrency,
        )

        trajectories: list[Trajectory] = []
        final_rewards: list[float] = []
        metrics_G_worker: list[dict[str, Any]] = []

        for res in results:
            # Record every result for CSV
            if res.verification_result and res.verification_result.success:
                normalized_turns = res.turns / cfg.env_params.max_turns
                reward = 1.0 - (1.0 - cfg.env_params.r_min) * normalized_turns
                reward = max(reward, cfg.env_params.r_min)
            else:
                reward = 0.0
            record = ResultRecord(
                original_question=qa_item.question,
                original_answer=qa_item.answer,
                generated_question=res.qa.question if res.qa else "",
                generated_answer=res.qa.answer if res.qa else "",
                cause=res.cause or "",
                reward=reward,
                has_trajectory=res.trajectory is not None,
                sampler_id=id(solver.model.sampling_client),
            )
            with rl_state.results_lock:
                rl_state.all_results.append(record)

            if res.trajectory is not None:
                trajectories.append(res.trajectory)
                final_rewards.append(reward)
                metrics_G_worker.append({"turns": res.turns})
            else:
                logger.warning("Trajectory is None.")

        # Save CSV after each question
        with rl_state.results_lock:
            _save_results_csv(rl_state.all_results, rl_state.csv_output_path)
        logger.info(
            f"Worker {worker_id} saved CSV ({len(rl_state.all_results)} records) to {rl_state.csv_output_path}"
        )

        if not trajectories:
            logger.info(f"Worker {worker_id} produced NO trajectories for question.")
            rl_state.queue_questions.task_done()
            continue
        elif len(trajectories) == 1:
            logger.info("Only one trajectory produced. Skipping this question.")
            rl_state.queue_questions.task_done()
            continue

        tg = TrajectoryGroup(
            trajectories_G=trajectories,
            final_rewards_G=final_rewards,
            metrics_G=metrics_G_worker,
        )
        await rl_state.queue_trajectories.put(tg)
        rl_state.queue_questions.task_done()
        logger.info(
            f"Worker {worker_id} produced TrajectoryGroup (mean reward: {sum(final_rewards) / len(final_rewards):.2f})"
        )

    logger.info(f"Rollout Worker {worker_id} finished.")


def remove_uniform_traj_groups(
    traj_groups: list[TrajectoryGroup],
) -> list[TrajectoryGroup]:
    return [
        traj_group
        for traj_group in traj_groups
        if not uniform_reward(traj_group.final_rewards_G)
    ]


async def train_worker(
    worker_id: int,
    rl_state: RLState,
    cfg: RLConfig,
    kl_reference_client: SamplingClient | None,
    ml_logger: MLLogger,
):
    logger.info(f"[Train Worker {worker_id}] started.")
    logger.warning("[CAUTION!] Temporary train worker does not do anything.")
    batch_trajectory_groups: list[TrajectoryGroup] = []
    step_counter = 0

    while True:
        tg = await rl_state.queue_trajectories.get()

        batch_trajectory_groups.append(tg)
        rl_state.queue_trajectories.task_done()

        nonuniform_traj_groups = remove_uniform_traj_groups(batch_trajectory_groups)

        if len(nonuniform_traj_groups) >= cfg.rollout_params.num_groups_per_batch:
            step_counter += 1
            logger.info(
                f"[Train Worker {worker_id}] Training Batch {step_counter} with {len(nonuniform_traj_groups)} groups..."
            )

            data_D, prepare_minibatch_metrics = await prepare_minibatch_simplified(
                nonuniform_traj_groups,
                regularization=cfg.optimizer_params.advantage_regularizer,
                kl_reference_client=kl_reference_client,
                kl_penalty_coef=cfg.optimizer_params.kl_penalty_coef,
                kl_discount_factor=cfg.optimizer_params.kl_discount_factor,
            )

            metrics_step = {}

            if len(data_D) > 0:
                await train_step(
                    data_D=data_D,
                    training_client=rl_state.training_client,
                    adam_params=cfg.optimizer_params.adam_params,
                    num_steps=cfg.optimizer_params.num_steps,
                    loss_fn=cfg.optimizer_params.loss_fn,
                    metrics=metrics_step,
                )
                rl_state.step_counter += 1

                logger.info(
                    f"[Train Worker {worker_id}]   -> Update completed. Metrics: {metrics_step}"
                )

                # Update sampling client
                new_client = await rl_state.training_client.save_weights_and_get_sampling_client_async()
                await rl_state.sampling_client_manager.update_client(new_client)
                logger.info(
                    f"[Train Worker {worker_id}]   -> Latest sampling client updated. New ID: {id(new_client)}"
                )

            # Calculate metrics (always log environmental rewards)
            metrics = compute_batch_metrics(batch_trajectory_groups)
            if metrics_step:
                metrics.update(metrics_step)
            if prepare_minibatch_metrics:
                metrics.update(prepare_minibatch_metrics)
            metrics["train/batch_size"] = len(data_D)
            metrics["train/learning_rate"] = (
                cfg.optimizer_params.adam_params.learning_rate
            )
            ml_logger.log_metrics(metrics)

            batch_trajectory_groups = []

    logger.info(f"[Train Worker {worker_id}] finished.")


async def main():

    cfg = RLConfig(
        experiment_setting=ExperimentSettings(
            wandb_project="Adapter Agent",
            experiment_name=f"Adapter Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        optimizer_params=OptimizerParams(
            adam_params=tinker.AdamParams(
                learning_rate=3e-5,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
            ),
            loss_fn="ppo",
            advantage_regularizer="output_token",
            num_steps=1,
            # kl_penalty_coef=1e-8,
        ),
        rollout_params=RolloutParams(
            num_groups_per_batch=8,
            num_rollout_workers=8,
            rollouts_per_question=8,
            per_group_concurrency=8,
            temperature=0.7,
        ),
        env_params=EnvParams(
            max_turns=5,
            r_min=0.5,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
        ),
        model_loading_settings=ModelLoadingSettings(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            # model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
            lora_rank=32,
            resume_sampler_path="tinker://1bc694fc-033d-5b20-877e-519888a70c1d:train:0/sampler_weights/000030",
            resume_trainer_path="tinker://1bc694fc-033d-5b20-877e-519888a70c1d:train:0/weights/000030",
            # resume_sampler_path=(
            #     "tinker://15349118-db43-54bd-88b5-6f187dc6ba28:train:0/sampler_weights/000030"
            # ),
            # resume_trainer_path=(
            #     "tinker://15349118-db43-54bd-88b5-6f187dc6ba28:train:0/weights/000030"
            # ),
        ),
    )

    # Setup logging
    log_root = cfg.experiment_setting.log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    setup_base_loglevel()
    ml_logger = ml_log.setup_logging(
        log_dir=str(log_root),
        wandb_project=cfg.experiment_setting.wandb_project,
        config=cfg,
    )
    logger.setLevel(logging.DEBUG)

    logging.getLogger("adapter_agent.hierarchical.agent.solver").setLevel(logging.DEBUG)
    logging.getLogger("adapter_agent.hierarchical.agent.simplified_solver").setLevel(
        logging.DEBUG
    )

    # Setup agents
    service_client = tinker.ServiceClient()
    logger.info(
        f"Setting up model {cfg.model_loading_settings.model_name} from {cfg.model_loading_settings.resume_sampler_path}..."
    )
    model, tokenizer, _renderer = setup_tinkermodel(
        service_client,
        cfg.model_loading_settings.model_name,
        cfg.model_loading_settings.resume_sampler_path,
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.env_params.library.local_path)
    logger.info("RustDocAnalyzer setup successful.")

    # Capture initial sampling client and model name
    sampling_client_manager = SharedSamplingClient(model.sampling_client)

    # Verifier
    logger.info("Initializing Verifier...")
    verifier_model = get_gemini()
    verifier = Verifier(model=verifier_model, rust_doc_analyzer=rust_doc_analyzer)
    if cfg.optimizer_params.kl_penalty_coef > 0:
        logger.info("Initializing KL reference client...")
        # Create a sampling client for the reference model
        kl_reference_client = service_client.create_sampling_client(
            base_model=cfg.model_loading_settings.model_name,
            model_path=cfg.model_loading_settings.resume_sampler_path,
        )
        logger.info(
            f"KL reference client initialized for model: {cfg.model_loading_settings.model_name} with path: {cfg.model_loading_settings.resume_sampler_path}"
        )
    else:
        kl_reference_client = None
    # Load questions
    logger.info("Loading questions from generated_qas.json...")
    qas_path = Path("generated_qas.json")
    qas_data_raw = SFTDataset.model_validate_json(qas_path.read_text())

    # Parse into QA objects
    qas_data = qas_data_raw.items

    logger.info(f"Loaded {len(qas_data)} questions.")

    # Initialize Training Client
    logger.info("Initializing Training Client...")
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_loading_settings.model_name, rank=cfg.model_loading_settings.lora_rank
    )
    if cfg.model_loading_settings.resume_trainer_path:
        logger.info(
            f"Loading trainer from {cfg.model_loading_settings.resume_trainer_path}..."
        )
        await training_client.load_state_async(
            path=cfg.model_loading_settings.resume_trainer_path
        )
    else:
        logger.info("No trainer path provided, starting from scratch.")

    # Shared results list for CSV export
    rl_state = RLState(
        queue_questions=asyncio.Queue(),
        queue_trajectories=asyncio.Queue(),
        sampling_client_manager=sampling_client_manager,
        all_results=[],
        results_lock=threading.Lock(),
        csv_output_path=log_root / "rl_results.csv",
        rust_doc_analyzer=rust_doc_analyzer,
        litellm_model_name=model.model,
        training_client=training_client,
    )

    # TODO: remove this
    for qa in qas_data * 3:
        rl_state.queue_questions.put_nowait(qa)

    # Start Workers
    train_worker_task = asyncio.create_task(
        train_worker(
            worker_id=0,
            rl_state=rl_state,
            cfg=cfg,
            kl_reference_client=kl_reference_client,
            ml_logger=ml_logger,
        )
    )
    rollout_workers_tasks = [
        asyncio.create_task(
            rollout_worker(
                worker_id=i,
                rl_state=rl_state,
                verifier=verifier,
                cfg=cfg,
            )
        )
        for i in range(cfg.rollout_params.num_rollout_workers)
    ]

    # Wait for producers to finish, or trainer to crash
    rollouts_task = asyncio.gather(*rollout_workers_tasks)
    done, pending = await asyncio.wait(
        {rollouts_task, train_worker_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if train_worker_task in done:
        # Train worker finished early (likely crashed)
        # Cancel rollouts
        rollouts_task.cancel()
        try:
            await rollouts_task
        except asyncio.CancelledError:
            pass
        # Propagate trainer exception
        await train_worker_task

    # Rollouts finished successfully
    # Wait for consumer to finish processing all items, or trainer to crash
    queue_task = asyncio.create_task(rl_state.queue_trajectories.join())
    done, pending = await asyncio.wait(
        {queue_task, train_worker_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if train_worker_task in done:
        # Train worker finished early (likely crashed) during queue drain
        queue_task.cancel()
        # Propagate trainer exception
        await train_worker_task

    # Queue is joined. Now cancel trainer.
    train_worker_task.cancel()
    try:
        await train_worker_task
    except asyncio.CancelledError:
        pass

    _ = await tinker_cookbook.checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=str(cfg.experiment_setting.log_root()),
        loop_state={"batch": rl_state.step_counter},
        kind="both",
        ttl_seconds=cfg.experiment_setting.ttl_seconds,
    )


if __name__ == "__main__":
    asyncio.run(main())
