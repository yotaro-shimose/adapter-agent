"""STaR (Self-Taught Reasoner) 実行エントリ。

各 iteration で以下を実行:
  1. RL タスク群に対して N=16 サンプル rollout を 1 パス生成
  2. 実行+verify 成功サンプルだけを `SuccessfulQRABuffer` に累積
  3. バッファから window 件サンプルして cross_entropy SFT
  4. `eval_interval` 毎に評価を起動
  5. `checkpoint_interval` 毎にチェックポイント保存

GRPO は使わない (純 STaR)。初期ウェイトは既存 RL チェックポイントから resume する
前提 (`ModelLoadingSettings.resume_*`)。

実行: `uv run scripts/run_star.py`
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import tinker
from dotenv import load_dotenv

from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.simple_internalizer import STaRPipeline, STaRPipelineConfig
from adapter_agent.simple_internalizer.data_sources import load_gh_archive_suite
from adapter_agent.simple_internalizer.types import (
    CheckpointSettings,
    EvalSettings,
    RolloutSettings,
    STaRSettings,
)
from adapter_agent.util.logger_util import ClockCycleFilteredLogger

# --- STaR 設定 ---
RL_TASK_SLICE = slice(0, 40)
ADDITIONAL_RL_SLICE: slice | None = slice(60, 300)
EVAL_SLICE = slice(40, 60)
VERIFIER_MODEL: Literal["gemini", "gemini_lite"] = "gemini_lite"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.getLogger("adapter_agent.internalize").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"


async def main() -> None:
    load_dotenv()
    json_path = Path("repositories/numrs/target/doc/numrs2.json")
    if not json_path.exists():
        logger.error(f"RustDoc JSON not found at {json_path}")
        return

    analyzer = await AsyncRustDocAnalyzer.create_from_json(json_path)
    runtime_settings = RuntimeSettings.cloudrun_numrs2()
    simple_train_id = f"star_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Seed suites (RL 側と同じ構成: rl_0 は 0:40, 追加 60:300, eval 40:60)
    rl_suite = load_gh_archive_suite(
        name="gh_archive_study_suite",
        task_slice=RL_TASK_SLICE,
        for_rl=True,
        for_eval=False,
    )
    seed_suites = [rl_suite]
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

    # 既存 RL チェックポイントから resume
    model_loading = ModelLoadingSettings(
        model_name="Qwen/Qwen3-8B",
        resume_trainer_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/weights/rl_0020",
        resume_sampler_path="tinker://976a7c11-7e95-596e-9230-38bff6526aa1:train:0/sampler_weights/rl_0020",
        lora_rank=32,
    )

    verifier_model = get_gemini() if VERIFIER_MODEL == "gemini" else get_gemini_lite()

    config = STaRPipelineConfig(
        simple_train_id=simple_train_id,
        library_name="numrs2",
        model_loading_settings=model_loading,
        rollout=RolloutSettings(
            runtime_settings=runtime_settings,
            num_samples=4,
            max_output_tokens=4000,
            runtime_pool_size=50,
            verifier_model=verifier_model,
            use_ray=True,
            ray_num_processes=8,
            ray_workers_per_process=16,
            ray_runtime_pool_size_per_process=10,
            ray_actor_stagger_s=1.0,
            worker_stagger_s=2.0,
        ),
        eval=EvalSettings(
            eval_rollout=4,
            eval_interval=1,
            eval_concurrency=48,
        ),
        checkpoint=CheckpointSettings(checkpoint_interval=5),
        star=STaRSettings(
            buffer_max_size=5000,
            sft_window=256,
            sft_batch_size=64,
            sft_epochs_per_round=1,
            sft_adam_params=tinker.AdamParams(learning_rate=2e-4),
            cpt=False,
        ),
        max_iterations=100,
    )

    pipeline = await STaRPipeline.create(config=config, seed_suites=seed_suites)
    filtered_logger = ClockCycleFilteredLogger(pipeline.ml_logger)
    pipeline.ml_logger = filtered_logger
    pipeline.evaluate_worker.ml_logger = filtered_logger

    try:
        async with analyzer:
            logger.info("Launching STaR pipeline.")
            await pipeline.run()
    except Exception:
        logger.exception("STaR launcher encountered an error")


if __name__ == "__main__":
    asyncio.run(main())
