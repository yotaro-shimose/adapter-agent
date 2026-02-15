from pydantic import Field
from datetime import datetime
from adapter_agent.hierarchical.gh import Library
import asyncio
import csv
import logging
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import litellm
import tinker
import torch
import wandb
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import setup_tinkermodel
from tinker.types.loss_fn_type import LossFnType
from tinker_cookbook.rl.data_processing import (
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.trace import scope

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.state import SFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from adapter_agent.rl.advantage import compute_advantages, get_traj_output_token_count
from adapter_agent.rl.config import RLConfig
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


def _save_results_csv(results: list[ResultRecord], path: str) -> None:
    """Save results to CSV, overwriting the file each time."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ResultRecord)]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
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
    metrics: dict[str, Any] | None = None,
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


def create_wandb_samples_table(
    data_D: list[tinker.Datum],
    tokenizer: Any,
    step_counter: int,
) -> wandb.Table | None:
    if not data_D:
        return None
    try:
        table = wandb.Table(columns=["step", "full_text", "mean_advantage"])
        for datum in data_D:
            # Decode text
            tokens = []
            for chunk in datum.model_input.chunks:
                if isinstance(chunk, tinker.EncodedTextChunk):
                    tokens.extend(chunk.tokens)
            full_text = tokenizer.decode(tokens)

            # Extract advantage
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mean_adv = advantages.mean().item()

            table.add_data(step_counter, full_text, mean_adv)

        return table
    except Exception as e:
        logger.warning(f"Failed to log wandb table: {e}")
        return None


def _compute_batch_metrics(
    batch_trajectory_groups: list[TrajectoryGroup],
    data_D: list[tinker.Datum],
) -> dict[str, Any]:
    all_rewards = []
    for group in batch_trajectory_groups:
        all_rewards.extend(group.final_rewards_G)

    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    return {
        "train/num_samples": len(data_D),
        "train/mean_reward": mean_reward,
        "train/num_groups": len(batch_trajectory_groups),
    }


def _log_metrics_and_table(
    metrics: dict[str, Any],
    data_D: list[tinker.Datum],
    step_counter: int,
    tokenizer: Any,
    ml_logger: Any,
    cfg: RLConfig,
) -> None:
    ml_logger.log_metrics(metrics, step=step_counter)

    if cfg.wandb_project is not None:
        if (
            table := create_wandb_samples_table(data_D, tokenizer, step_counter)
        ) is not None:
            wandb.log({"train/samples": table}, step=step_counter)


def _log_correlation_data(
    batch_trajectory_groups: list[TrajectoryGroup],
    cfg: RLConfig,
) -> None:
    """Save correlation data before training step."""
    filtered_groups = remove_constant_reward_groups(batch_trajectory_groups)
    if not filtered_groups:
        return

    advantages_P = compute_advantages(filtered_groups, cfg.advantage_regularizer)
    correlation_csv = Path(cfg.correlation_csv_path)
    correlation_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = correlation_csv.exists()

    with open(correlation_csv, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tokens", "turns", "reward", "advantage"]
        )
        if not file_exists:
            writer.writeheader()

        for tg, adv_G in zip(filtered_groups, advantages_P):
            for traj, reward, adv, metrics in zip(
                tg.trajectories_G,
                tg.get_total_rewards(),
                adv_G,
                tg.metrics_G,
            ):
                writer.writerow(
                    {
                        "tokens": get_traj_output_token_count(traj),
                        "turns": metrics.get("turns", 0),
                        "reward": float(reward),
                        "advantage": float(adv),
                    }
                )


async def main():
    setup_base_loglevel()
    # Add trace processor for printing behavior
    # add_trace_processor(AgentContentPrinter())

    cfg = RLConfig(
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        base_url=None,
        lora_rank=32,
        log_path="logs/hoge_rl",
        wandb_project="Hoge RL",
        experiment_name=Field(
            default_factory=lambda: (
                f"Hoge_RL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
        ),
        adam_params=tinker.AdamParams(
            learning_rate=1e-8,
            beta1=0.9,
            beta2=0.95,
            eps=1e-12,
        ),
        loss_fn="dro",
        advantage_regularizer="output_token",
        num_groups_per_batch=2,  # TODO: Increase this
        num_rollout_workers=1,  # TODO: Increase this
        num_steps=1,
        rollouts_per_question=4,
        per_group_concurrency=1,
        max_turns=10,
        image_name="coder-mcp-numrs2:latest",
        library=Library(name="numrs2", local_path=Path("repositories/numrs")),
        r_min=0.5,
        csv_output_path="logs/rl_results.csv",
        correlation_csv_path="logs/rl_correlation.csv",
    )
    cfg.log_path = f"{cfg.log_path}/{cfg.experiment_name}"

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
    )
    logger.setLevel(logging.DEBUG)

    # Ensure output also goes to logs/temp/logs.log
    temp_log_path = Path("logs/temp/logs.log")
    temp_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(temp_log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("adapter_agent.hierarchical.agent.solver").setLevel(logging.DEBUG)

    service_client = tinker.ServiceClient()
    path = (
        "tinker://c25c1802-91fa-5e5d-badf-d1ef12a28c32:train:0/sampler_weights/000030"
    )

    logger.info(f"Setting up model {cfg.model_name} from {path}...")
    model, tokenizer, _renderer = setup_tinkermodel(
        service_client, cfg.model_name, path
    )

    rust_doc_analyzer = RustDocAnalyzer.from_libdir(cfg.library.local_path)
    logger.info("RustDocAnalyzer setup successful.")

    # Solver
    logger.info("Initializing Solver and Verifier...")
    solver = Solver(model=model, rust_doc_analyzer=rust_doc_analyzer, memory=None)

    # Verifier
    verifier_model = get_gemini()
    verifier = Verifier(
        model=verifier_model, rust_doc_analyzer=rust_doc_analyzer, memory=None
    )

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
        cfg.model_name, rank=cfg.lora_rank
    )

    # queues
    queue_questions: asyncio.Queue[QA] = asyncio.Queue()
    queue_trajectories: asyncio.Queue[TrajectoryGroup] = asyncio.Queue()

    # Temporarily use only the first question for all workers
    first_qa = qas_data[1]
    for _ in range(cfg.num_rollout_workers * 12):
        queue_questions.put_nowait(first_qa)

    # Shared results list for CSV export
    all_results: list[ResultRecord] = []
    results_lock = threading.Lock()

    async def solve_for_question(question_text: str):
        task = Task.from_instruction(question_text)
        results = await gather_with_semaphore(
            [
                solve_verify(
                    solver=solver,
                    verifier=verifier,
                    task=task,
                    image_name=cfg.image_name,
                    library_name=cfg.library.name,
                    max_turns=cfg.max_turns,
                    collect_trajectory=True,
                    use_search=False,
                )
                for _ in range(cfg.rollouts_per_question)
            ],
            max_concurrent=cfg.per_group_concurrency,
        )
        return results

    async def rollout_worker(worker_id: int):
        logger.info(f"Rollout Worker {worker_id} started.")
        while True:
            try:
                qa_item = queue_questions.get_nowait()
            except asyncio.QueueEmpty:
                break

            question_text = qa_item.question
            sampler_id = id(solver.model.sampling_client)
            results = await solve_for_question(question_text)

            trajectories: list[Trajectory] = []
            final_rewards: list[float] = []
            metrics_G_worker: list[dict[str, Any]] = []

            for idx, res in enumerate(results):
                # Record every result for CSV
                if res.verification_result and res.verification_result.success:
                    normalized_turns = res.turns / cfg.max_turns
                    reward = 1.0 - (1.0 - cfg.r_min) * normalized_turns
                    reward = max(reward, cfg.r_min)
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
                    sampler_id=sampler_id,
                )
                with results_lock:
                    all_results.append(record)

                if res.trajectory is not None:
                    trajectories.append(res.trajectory)
                    final_rewards.append(reward)
                    metrics_G_worker.append({"turns": res.turns})
                else:
                    logger.warning(f"Trajectory {idx} is None.")

            # Save CSV after each question
            with results_lock:
                _save_results_csv(all_results, cfg.csv_output_path)
            logger.info(
                f"Worker {worker_id} saved CSV ({len(all_results)} records) to {cfg.csv_output_path}"
            )

            if not trajectories:
                logger.info(
                    f"Worker {worker_id} produced NO trajectories for question."
                )
                queue_questions.task_done()
                continue
            elif len(trajectories) == 1:
                logger.info("Only one trajectory produced. Skipping this question.")
                queue_questions.task_done()
                continue
            elif uniform_reward(final_rewards):
                logger.info("Uniform reward. Skipping this question.")
                queue_questions.task_done()
                continue

            tg = TrajectoryGroup(
                trajectories_G=trajectories,
                final_rewards_G=final_rewards,
                metrics_G=metrics_G_worker,
            )
            await queue_trajectories.put(tg)
            queue_questions.task_done()
            logger.info(
                f"Worker {worker_id} produced TrajectoryGroup (mean reward: {sum(final_rewards) / len(final_rewards):.2f})"
            )

        logger.info(f"Rollout Worker {worker_id} finished.")

    async def train_worker(worker_id: int):
        logger.info(f"[Train Worker {worker_id}] started.")
        batch_trajectory_groups = []
        step_counter = 0

        while True:
            try:
                tg = await queue_trajectories.get()
            except asyncio.CancelledError:
                break

            batch_trajectory_groups.append(tg)
            queue_trajectories.task_done()

            if len(batch_trajectory_groups) >= cfg.num_groups_per_batch:
                step_counter += 1
                logger.info(
                    f"[Train Worker {worker_id}] Training Batch {step_counter} with {len(batch_trajectory_groups)} groups..."
                )

                data_D = prepare_minibatch_simplified(
                    batch_trajectory_groups, cfg.advantage_regularizer
                )

                # Save correlation data before training step
                _log_correlation_data(batch_trajectory_groups, cfg)

                metrics_step = {}

                if len(data_D) > 0:
                    training_logprobs = await train_step(
                        data_D=data_D,
                        training_client=training_client,
                        adam_params=cfg.adam_params,
                        num_steps=cfg.num_steps,
                        loss_fn=cfg.loss_fn,
                        metrics=metrics_step,
                    )
                    logger.info(
                        f"[Train Worker {worker_id}]   -> Update completed. Metrics: {metrics_step}"
                    )

                    # Update sampling client
                    new_client = await training_client.save_weights_and_get_sampling_client_async()
                    solver.model.update_sampling_client(new_client)
                    logger.info(
                        f"[Train Worker {worker_id}]   -> Solver sampling client updated."
                    )

                # Calculate metrics (always log environmental rewards)
                metrics = _compute_batch_metrics(batch_trajectory_groups, data_D)
                if metrics_step:
                    metrics.update(metrics_step)

                _log_metrics_and_table(
                    metrics,
                    data_D,
                    step_counter,
                    tokenizer,
                    ml_logger,
                    cfg,
                )

                batch_trajectory_groups = []

        logger.info(f"[Train Worker {worker_id}] finished.")

    # Start Workers
    train_worker_task = asyncio.create_task(train_worker(0))
    rollout_workers_tasks = [
        asyncio.create_task(rollout_worker(i)) for i in range(cfg.num_rollout_workers)
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
    queue_task = asyncio.create_task(queue_trajectories.join())
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


if __name__ == "__main__":
    asyncio.run(main())
