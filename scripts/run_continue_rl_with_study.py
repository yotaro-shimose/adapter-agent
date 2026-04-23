"""Run continuation RL in parallel with a StudyRunner that consumes failed tasks.

Flow:
1. RL samples rollouts for a task; if all rollouts fail, SimplePipeline pushes
   the task onto `study_task_queue`.
2. StudyRunner drains `study_task_queue`, injects each task into its own
   TaskNetwork, and N StudyWorkers solve them multi-turn.
3. On success, each worker distills the trajectory into a QRA via Gemini and
   pushes it onto `qra_out_queue`. (RL-side consumption of these QRAs is a
   follow-up; this launcher just logs what the queue produces.)

Wiki is reused (no reset) — edit `STUDY_WIKI_VERSION` to point at an existing one.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tinker
from dotenv import load_dotenv
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.task_net import StudyTask
from adapter_agent.simple_internalizer import PipelineConfig, SimplePipeline
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.study.qra_budget import QRABudgetConfig
from adapter_agent.study.runner import (
    StudyRunner,
    build_study_worker_resources,
    setup_study_runner,
    teardown_study_runner,
)

# --- General (RL pipeline) ---
RL_TASK_SLICE = slice(0, 40)
ADDITIONAL_RL_SLICE: slice | None = slice(60, 300)
EVAL_SLICE = slice(40, 60)
VERIFIER_MODEL: Literal["gemini", "gemini_lite"] = "gemini_lite"

# --- Study side (ignored when STUDY_ENABLED is False) ---
STUDY_ENABLED = False
STUDY_WIKI_VERSION = "study_20260419_041136"
STUDY_WORKERS = 4
STUDY_SOLVER_CKPT: str | None = None
STUDY_SOLVER_BACKEND = "qwen_tinker"
STUDY_VERIFY_RUNTIME = "docker"
STUDY_VERIFY_RUNTIME_POOL_SIZE = 2
STUDY_DRAIN_TIMEOUT_S = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"


class ClockCycleFilteredLogger(MLLogger):
    def __init__(self, base: MLLogger) -> None:
        self._base = base

    def log_hparams(self, config: Any) -> None:
        self._base.log_hparams(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        filtered = {k: v for k, v in metrics.items() if "clock_cycle" not in k}
        self._base.log_metrics(filtered, step)

    def log_long_text(self, key: str, text: str) -> None:
        self._base.log_long_text(key, text)

    def close(self) -> None:
        self._base.close()


async def main() -> None:
    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")
    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    simple_train_id = f"continue_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_experiment = f"study_{simple_train_id}"

    study_rl_suite = load_gh_archive_suite(
        name="gh_archive_study_suite",
        task_slice=RL_TASK_SLICE,
        for_rl=True,
        for_eval=False,
    )
    seed_suites = [study_rl_suite]
    if ADDITIONAL_RL_SLICE is not None:
        additional_rl_suite = load_gh_archive_suite(
            name="gh_archive_study_suite",
            task_slice=ADDITIONAL_RL_SLICE,
            for_rl=True,
            for_eval=False,
        )
        seed_suites.append(additional_rl_suite)
    eval_suite = load_gh_archive_suite(
        name="gh_archive_eval",
        task_slice=EVAL_SLICE,
        for_rl=False,
        for_eval=True,
    )
    seed_suites.append(eval_suite)

    continue_rl_checkpoint = ModelLoadingSettings(
        model_name="Qwen/Qwen3-8B",
        resume_trainer_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/weights/rl_0020",
        resume_sampler_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020",
        lora_rank=32,
    )

    verifier_model = get_gemini() if VERIFIER_MODEL == "gemini" else get_gemini_lite()

    config = PipelineConfig(
        runtime_pool_size=200,
        rl_worker_count=50,
        eval_concurrency=48,
        generation_concurrency=400,
        simple_train_id=simple_train_id,
        library_name="numrs2",
        runtime_settings=runtime_settings,
        model_loading_settings=continue_rl_checkpoint,
        sft=None,
        eval_rollout=4,
        eval_interval=5,
        rl_rollout=16,
        max_iterations=200,
        rl_checkpoint_interval=10,
        rl_batch_size=96,
        rl_update_epochs=1,
        rl_adam_params=tinker.AdamParams(learning_rate=2e-4),
        rl_loss_fn="cispo",
        verifier_model=verifier_model,
    )

    study_task_queue: asyncio.Queue[StudyTask] | None = (
        asyncio.Queue() if STUDY_ENABLED else None
    )
    qra_out_queue: asyncio.Queue[tuple[str, QRA]] | None = (
        asyncio.Queue() if STUDY_ENABLED else None
    )

    pipeline = await SimplePipeline.create(
        config=config,
        rust_doc_analyzer=analyzer,
        knowledge_list=[],
        seed_suites=seed_suites,
        study_task_queue=study_task_queue,
        qra_in_queue=qra_out_queue,
    )
    filtered_logger = ClockCycleFilteredLogger(pipeline.ml_logger)
    pipeline.ml_logger = filtered_logger
    pipeline.evaluate_worker.ml_logger = filtered_logger

    async def _drain_study_until_idle(
        runner: StudyRunner,
        in_queue: asyncio.Queue[StudyTask],
        grace_s: int,
    ) -> None:
        """After RL finishes, give StudyRunner a window to finish in-flight work.

        Idle = (a) input queue empty, (b) TaskNetwork has nothing to pop or
        executing, and (c) the KnowledgeStudier's reflection-integration queue
        is drained. (c) matters because reflection integration is often the
        longest tail — if we tear down while it's still running, studier.stop()
        will block beyond its timeout. Require 5 consecutive idle seconds to
        avoid racing a worker that's about to push a new reflection batch.
        """
        idle_ticks = 0
        for elapsed in range(grace_s):
            await asyncio.sleep(1)
            in_queue_empty = in_queue.empty()
            no_executing = not runner.task_network.executing_tasks
            pool_empty = not runner.task_network.tasks_pool
            studier_idle = runner.studier.queue.empty()
            if in_queue_empty and no_executing and pool_empty and studier_idle:
                idle_ticks += 1
                if idle_ticks >= 5:
                    logger.info(
                        f"Study drain: idle for 5s after {elapsed + 1}s, "
                        "proceeding to teardown."
                    )
                    return
            else:
                idle_ticks = 0
        logger.warning(f"Study drain: {grace_s}s grace exhausted; tearing down anyway.")

    pipeline_task = asyncio.create_task(pipeline.run(), name="rl-pipeline")

    if STUDY_ENABLED:
        assert study_task_queue is not None and qra_out_queue is not None
        study_resources = await build_study_worker_resources(
            wiki_version=STUDY_WIKI_VERSION,
            solver_backend=STUDY_SOLVER_BACKEND,
            solver_model_ckpt_path=STUDY_SOLVER_CKPT,
            verify_runtime=STUDY_VERIFY_RUNTIME,
            verify_runtime_pool_size=STUDY_VERIFY_RUNTIME_POOL_SIZE,
        )
        study_runner, study_rl_db = await setup_study_runner(
            experiment_name=study_experiment,
            resources=study_resources,
            in_queue=study_task_queue,
            qra_out_queue=qra_out_queue,
            num_workers=STUDY_WORKERS,
            json_path=Path("graphvis/public/study_data.json"),
            qra_budget_config=QRABudgetConfig(
                target_qras_per_task=8,
                max_attempts_multiplier=2.0,
            ),
        )
        study_task = asyncio.create_task(study_runner.run(), name="study-runner")

        try:
            async with analyzer:
                logger.info("Launching RL pipeline and StudyRunner in parallel.")
                await pipeline_task
                logger.info("RL pipeline finished. Draining Study pipeline...")
                await _drain_study_until_idle(
                    study_runner, study_task_queue, STUDY_DRAIN_TIMEOUT_S
                )
        except Exception:
            logger.exception("Combined launcher encountered an error")
        finally:
            for t in (study_task, pipeline_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(study_task, pipeline_task, return_exceptions=True)
            await teardown_study_runner(study_runner, study_rl_db)
    else:
        try:
            async with analyzer:
                logger.info("Launching RL pipeline (Study disabled).")
                await pipeline_task
        except Exception:
            logger.exception("RL launcher encountered an error")
        finally:
            if not pipeline_task.done():
                pipeline_task.cancel()
            await asyncio.gather(pipeline_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
