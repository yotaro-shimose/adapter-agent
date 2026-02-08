from adapter_agent.hierarchical.gh import Library
import asyncio
import logging
from pathlib import Path
from typing import Any

import tinker
from pydantic import BaseModel, Field
from datetime import datetime

import torch
from oai_utils.async_utils import gather_with_semaphore
from oai_utils.tinker import setup_tinkermodel
from tinker.types.loss_fn_type import LossFnType
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import split_list
from tinker_cookbook.utils.trace import scope

from adapter_agent.hierarchical.agent.solver import Solver
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.process.solve_verify import solve_verify
from adapter_agent.hierarchical.state import SFTDataset
from adapter_agent.hierarchical.types import Task
from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer
from adapter_agent.model_helper import get_gemini
from adapter_agent.qra import QA
from adapter_agent.util.logger_util import setup_base_loglevel


class RLConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    base_url: str | None = None
    lora_rank: int = 32
    log_path: str = "logs/hoge_rl"
    wandb_project: str | None = "Hoge RL"
    experiment_name: str = Field(
        default_factory=lambda: f"Hoge_RL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    adam_params: tinker.AdamParams = tinker.AdamParams(
        learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8
    )
    num_substeps: int = 4
    loss_fn: str = "importance_sampling"
    num_groups_per_batch: int = 8  # TODO: Increase this
    num_train_updates: int = 3
    num_rollout_workers: int = 32  # TODO: Increase this
    rollouts_per_question: int = 4
    max_turns: int = 15
    library: Library = Library(name="numrs2", local_path=Path("repositories/numrs"))


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


logging.getLogger("openai.agents").addFilter(OpenAITracingFilter())


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
    adam_params: tinker.AdamParams,
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

    training_logprobs_D: list[torch.Tensor] = []
    optim_result: tinker.OptimStepResponse | None = None
    fwd_bwd_result: tinker.ForwardBackwardOutput | None = None

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
    if metrics is not None and fwd_bwd_result is not None and fwd_bwd_result.metrics:
        metrics.update(fwd_bwd_result.metrics)

    return training_logprobs_D


def uniform_reward(rewards: list[float]) -> bool:
    return all(r == rewards[0] for r in rewards)


def prepare_minibatch_simplified(trajectory_groups: list[TrajectoryGroup]):
    trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if not trajectory_groups:
        return []

    advantages_G = compute_advantages(trajectory_groups)

    # Assert any of the advantages are non-zero
    all_advantages = torch.cat(advantages_G)
    assert torch.any(all_advantages != 0), "All advantages are zero!"

    data_D, _metadata_D = assemble_training_data(trajectory_groups, advantages_G)
    return data_D


async def main():
    setup_base_loglevel()
    # Add trace processor for printing behavior
    # add_trace_processor(AgentContentPrinter())

    cfg = RLConfig()
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

    for item in qas_data:
        queue_questions.put_nowait(item)

    async def solve_for_question(question_text: str):
        task = Task.from_instruction(question_text)
        exclude = ["target", ".git"]
        workspace_template = Path("templates/rust_template")
        library_name = "numrs2"
        max_turns = cfg.max_turns
        collect_trajectory = True

        results = await gather_with_semaphore(
            [
                solve_verify(
                    solver=solver,
                    verifier=verifier,
                    task=task,
                    workspace_template=workspace_template,
                    library_name=library_name,
                    max_turns=max_turns,
                    collect_trajectory=collect_trajectory,
                    use_search=False,
                    exclude=exclude,
                )
                for _ in range(cfg.rollouts_per_question)
            ],
            max_concurrent=cfg.rollouts_per_question,
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
            # logger.info(f"Worker {worker_id} processing: {question_text[:30]}...")

            results = await solve_for_question(question_text)

            trajectories: list[Trajectory] = []
            final_rewards: list[float] = []

            for idx, res in enumerate(results):
                if res.trajectory is not None:
                    trajectories.append(res.trajectory)
                    if res.verification_result and res.verification_result.success:
                        final_rewards.append(1.0)
                    else:
                        final_rewards.append(0.0)
                else:
                    logger.warning(f"Trajectory {idx} is None.")

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

            metrics_G = [{} for _ in range(len(trajectories))]
            tg = TrajectoryGroup(
                trajectories_G=trajectories,
                final_rewards_G=final_rewards,
                metrics_G=metrics_G,
            )
            await queue_trajectories.put(tg)
            queue_questions.task_done()
            logger.info(
                f"Worker {worker_id} produced TrajectoryGroup (mean reward: {sum(final_rewards) / len(final_rewards):.2f})"
            )

        logger.info(f"Rollout Worker {worker_id} finished.")

    async def train_worker():
        logger.info("Train Worker started.")
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
                    f"Training Batch {step_counter} with {len(batch_trajectory_groups)} groups..."
                )

                data_D = prepare_minibatch_simplified(batch_trajectory_groups)

                if len(data_D) > 0:
                    accumulated_metrics = {}
                    # Update multiple times
                    for update_i in range(cfg.num_train_updates):
                        metrics_step = {}
                        training_logprobs = await train_step(
                            data_D=data_D,
                            training_client=training_client,
                            adam_params=cfg.adam_params,
                            num_substeps=cfg.num_substeps,
                            loss_fn=cfg.loss_fn,
                            metrics=metrics_step,
                        )
                        logger.info(
                            f"  -> Update {update_i + 1}/{cfg.num_train_updates} completed. Metrics: {metrics_step}"
                        )
                        for k, v in metrics_step.items():
                            if isinstance(v, (int, float)):
                                if k not in accumulated_metrics:
                                    accumulated_metrics[k] = []
                                accumulated_metrics[k].append(v)

                    # Update sampling client
                    new_client = await training_client.save_weights_and_get_sampling_client_async()
                    solver.model.update_sampling_client(new_client)
                    logger.info("  -> Solver sampling client updated.")

                    # Calculate metrics
                    all_rewards = []
                    for group in batch_trajectory_groups:
                        all_rewards.extend(group.final_rewards_G)

                    mean_reward = (
                        sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
                    )

                    metrics = {
                        "train/num_samples": len(data_D),
                        "train/mean_reward": mean_reward,
                        "train/num_groups": len(batch_trajectory_groups),
                    }

                    # Add mean of accumulated metrics
                    for k, v_list in accumulated_metrics.items():
                        if v_list:
                            metrics[f"train/{k}"] = sum(v_list) / len(v_list)

                    ml_logger.log_metrics(metrics, step=step_counter)

                batch_trajectory_groups = []

        logger.info("Train Worker finished.")

    # Start Workers
    rollout_workers_tasks = [
        asyncio.create_task(rollout_worker(i)) for i in range(cfg.num_rollout_workers)
    ]
    train_worker_task = asyncio.create_task(train_worker())

    # Wait for producers to finish
    await asyncio.gather(*rollout_workers_tasks)

    # Wait for consumer to finish processing all items
    await queue_trajectories.join()

    # Cancel consumer
    train_worker_task.cancel()
    try:
        await train_worker_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
