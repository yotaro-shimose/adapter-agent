"""STaR (Self-Taught Reasoner) 用の独立パイプライン。

ループ:
  for iter in range(max_iterations):
    1. rl タスクに対して N サンプル rollout を 1 パス生成
    2. 成功 outcome を QRA として抽出し、`SuccessfulQRABuffer` に追加
    3. バッファから `sft_window` 件サンプルして cross_entropy SFT
    4. `eval_interval` 毎に評価 (EvaluateWorker) を起動
    5. `checkpoint_interval` 毎にチェックポイント保存

SimplePipeline とは独立 (継承しない)。共有挙動は `TrainingRunner` 合成で再利用。
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.model_helper import get_gemini_lite
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient

from adapter_agent.simple_internalizer.evaluate_worker import EvaluateWorker
from adapter_agent.simple_internalizer.rollout_engine import (
    RolloutEngine,
    build_solver_system_prompt,
)
from adapter_agent.simple_internalizer.star_rollout_pool import STaRRolloutPool
from adapter_agent.simple_internalizer.successful_qra_buffer import SuccessfulQRABuffer
from adapter_agent.simple_internalizer.training_runner import TrainingRunner
from adapter_agent.simple_internalizer.types import SeedSuite, STaRPipelineConfig

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class STaRPipeline:
    """SFT データ生成 → SFT 学習 → eval を繰り返す STaR パイプライン。

    - rollout 生成: `STaRRolloutPool` (Ray) が task 群を fan-out して処理
    - SFT 学習/checkpoint/重み同期: `TrainingRunner` に委譲
    - 評価: `EvaluateWorker` を `eval_trigger` 経由で駆動
    - 成功サンプル蓄積: `SuccessfulQRABuffer` (FIFO, 上限付き)

    GRPO/RLGroup は使わない (純 STaR)。
    """

    config: STaRPipelineConfig
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    solver_model: TinkerModel
    ml_logger: MLLogger
    prisma_client: Prisma
    log_dir: Path
    shared_sampling_client: SharedSamplingClient
    evaluate_worker: EvaluateWorker
    training_runner: TrainingRunner
    star_rollout_pool: STaRRolloutPool
    star_buffer: SuccessfulQRABuffer
    seed_suites: list[SeedSuite]
    system_prompt: str
    eval_trigger: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    async def create(
        cls,
        config: STaRPipelineConfig,
        seed_suites: list[SeedSuite],
    ) -> Self:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        if not config.rollout.use_ray:
            raise ValueError(
                "STaRPipeline currently requires rollout.use_ray=True; "
                "set RolloutSettings.use_ray=True and Ray parallelism knobs."
            )

        verifier_model: AgentsSDKModel = (
            config.rollout.verifier_model or get_gemini_lite()
        )

        service_client = tinker.ServiceClient()

        if config.model_loading_settings.resume_trainer_path:
            if config.model_loading_settings.resume_optimizer_state:
                logger.info(
                    f"Loading trainer state + optimizer from {config.model_loading_settings.resume_trainer_path}..."
                )
                training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                    config.model_loading_settings.resume_trainer_path
                )
            else:
                logger.info(
                    f"Loading trainer state (weights only, fresh optimizer) from {config.model_loading_settings.resume_trainer_path}..."
                )
                training_client = await service_client.create_training_client_from_state_async(
                    config.model_loading_settings.resume_trainer_path
                )
        else:
            logger.info("Creating new Lora training client.")
            training_client = await service_client.create_lora_training_client_async(
                base_model=config.model_loading_settings.model_name,
                rank=config.model_loading_settings.lora_rank,
            )

        tinker_model, _, _ = setup_tinkermodel(
            model_name=config.model_loading_settings.model_name,
            service_client=service_client,
            path=config.model_loading_settings.resume_sampler_path,
        )

        log_dir = (
            Path("logs")
            / "simple_internalizer"
            / config.model_loading_settings.model_name
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ml_logger = ml_log.setup_logging(
            log_dir=str(log_dir),
            wandb_project="internalization",
            config={
                "model_name": config.model_loading_settings.model_name,
                "library": config.library_name,
                "pipeline": "star",
                "star_sft_window": config.star.sft_window,
                "star_buffer_max_size": config.star.buffer_max_size,
                "star_sft_batch_size": config.star.sft_batch_size,
                "star_sft_epochs_per_round": config.star.sft_epochs_per_round,
                "lora_rank": config.model_loading_settings.lora_rank,
            },
        )

        logger.info(
            f"STaRPipeline: {len(seed_suites)} seed suites "
            f"(rl={sum(1 for s in seed_suites if s.for_rl)}, "
            f"eval={sum(1 for s in seed_suites if s.for_eval)})"
        )
        for s in seed_suites:
            logger.info(
                f"Registered seed suite '{s.name}' with {len(s.tasks)} tasks "
                f"(for_rl={s.for_rl}, for_eval={s.for_eval})."
            )

        # DB 接続 + SimpleTrainRun を upsert (成功 QRA を star_qnas に永続化するため)
        db_manager = PostgresDB()
        await db_manager.connect()
        prisma_client = await db_manager.get_client()
        await prisma_client.simpletrainrun.upsert(
            where={"id": config.simple_train_id},
            data={"create": {"id": config.simple_train_id}, "update": {}},
        )

        shared_sampling_client = SharedSamplingClient(tinker_model.sampling_client)
        eval_trigger = asyncio.Event()

        # Evaluate worker は自前の RolloutEngine を持つ (driver プロセス内で回す)
        # ため、ここで driver ローカルの executor を構築せず、代わりに
        # RolloutEngine の executor は in-process な最小構成でよい —
        # ただし eval は実行+検証までやるので runtime_pool + verifier が必要。
        # シンプルさのため eval 用 runtime_pool を driver 側に 1 つ持たせる。
        from adapter_agent.hierarchical.agent.verifier import Verifier
        from adapter_agent.rl.env.runtime_pool import RuntimePool
        from adapter_agent.simple_internalizer.executor import InternalizeExecutor

        eval_runtime_pool = RuntimePool(
            config.rollout.runtime_settings,
            max_size=config.rollout.runtime_pool_size,
        )
        eval_executor = InternalizeExecutor(
            runtime_pool=eval_runtime_pool,
            verifier=Verifier(model=verifier_model, library_name=config.library_name),
        )
        system_prompt = build_solver_system_prompt(config.library_name)
        eval_rollout_engine = RolloutEngine(
            renderer=tinker_model.renderer,
            executor=eval_executor,
            system_prompt=system_prompt,
        )

        evaluate_worker = EvaluateWorker(
            ml_logger=ml_logger,
            shared_sampling_client=shared_sampling_client,
            eval_suites=[s for s in seed_suites if s.for_eval],
            trigger=eval_trigger,
            rollout_engine=eval_rollout_engine,
            eval_concurrency=config.eval.eval_concurrency,
            eval_rollout=config.eval.eval_rollout,
            max_output_tokens=config.rollout.max_output_tokens,
            prisma_client=prisma_client,
            simple_train_id=config.simple_train_id,
        )

        sampling_params = tinker.SamplingParams(
            max_tokens=config.rollout.max_output_tokens
        )
        star_rollout_pool = STaRRolloutPool(
            num_processes=config.rollout.ray_num_processes,
            workers_per_process=config.rollout.ray_workers_per_process,
            runtime_pool_size_per_process=config.rollout.ray_runtime_pool_size_per_process,
            actor_stagger_s=config.rollout.ray_actor_stagger_s,
            worker_stagger_s=config.rollout.worker_stagger_s,
            runtime_settings=config.rollout.runtime_settings,
            verifier_model=verifier_model,
            library_name=config.library_name,
            model_name=config.model_loading_settings.model_name,
            sampling_client=tinker_model.sampling_client,
            num_samples=config.rollout.num_samples,
            sampling_params=sampling_params,
        )

        async def _broadcast_hook(new_client: tinker.SamplingClient) -> None:
            await star_rollout_pool.broadcast_sampling_client(new_client)

        training_runner = TrainingRunner(
            training_client=training_client,
            shared_sampling_client=shared_sampling_client,
            ml_logger=ml_logger,
            log_dir=log_dir,
            model_name=config.model_loading_settings.model_name,
            eval_trigger=eval_trigger,
            broadcast_hook=_broadcast_hook,
        )

        star_buffer = SuccessfulQRABuffer(max_size=config.star.buffer_max_size)

        return cls(
            config=config,
            service_client=service_client,
            training_client=training_client,
            solver_model=tinker_model,
            ml_logger=ml_logger,
            prisma_client=prisma_client,
            log_dir=log_dir,
            shared_sampling_client=shared_sampling_client,
            evaluate_worker=evaluate_worker,
            training_runner=training_runner,
            star_rollout_pool=star_rollout_pool,
            star_buffer=star_buffer,
            seed_suites=seed_suites,
            system_prompt=system_prompt,
            eval_trigger=eval_trigger,
        )

    async def run(self) -> None:
        eval_worker_task = asyncio.create_task(self.evaluate_worker.run_loop())
        try:
            async with self.star_rollout_pool:
                rl_tasks = [
                    t for s in self.seed_suites if s.for_rl for t in s.tasks
                ]
                if not rl_tasks:
                    logger.warning(
                        "No tasks with for_rl=True found; STaR loop will have nothing to do."
                    )

                # task.id → suite.name マップ (DB 書き込み時の task_suite 用)
                task_suite_map: dict[str, str] = {
                    t.id: s.name
                    for s in self.seed_suites
                    if s.for_rl
                    for t in s.tasks
                }

                for iteration in range(self.config.max_iterations):
                    step = iteration + 1
                    logger.info(f"--- STaR iteration {step} ---")

                    # 1. Rollout 1 パス (結果を到着順にストリーム処理、~20 回に分けてログ)
                    logger.info(
                        f"Generating rollouts: {len(rl_tasks)} tasks × "
                        f"{self.config.rollout.num_samples} samples..."
                    )
                    log_every = max(1, len(rl_tasks) // 20)
                    new_qras_this_round = 0
                    total_outcomes_this_round = 0
                    n_done = 0
                    async for task, batch in self.star_rollout_pool.stream_one_pass(
                        rl_tasks
                    ):
                        n_done += 1
                        qras = [
                            QRA(
                                question=task.instruction,
                                reasoning=o.reasoning,
                                answer=o.answer,
                            )
                            for o in batch.outcomes
                            if o.success and o.parsed
                        ]
                        self.star_buffer.extend(qras)
                        new_qras_this_round += len(qras)
                        total_outcomes_this_round += len(batch.outcomes)
                        if qras:
                            await self._persist_star_qras(
                                qras=qras,
                                star_step=step,
                                task_id=task.id,
                                task_suite=task_suite_map.get(task.id, "unknown"),
                            )
                        if n_done % log_every == 0 or n_done == len(rl_tasks):
                            running_ratio = (
                                new_qras_this_round / total_outcomes_this_round
                                if total_outcomes_this_round else 0.0
                            )
                            logger.info(
                                f"[iter {step}] {n_done}/{len(rl_tasks)} tasks done, "
                                f"new_successful={new_qras_this_round}, "
                                f"running_success_ratio={running_ratio:.3f}, "
                                f"buffer={len(self.star_buffer)} "
                                f"(dropped_total={self.star_buffer.dropped_total})"
                            )

                    # 2. round 終了後の summary metrics
                    self.ml_logger.log_metrics({
                        "star/new_successful_qras": float(new_qras_this_round),
                        "star/buffer_size": float(len(self.star_buffer)),
                        "star/buffer_dropped_total": float(
                            self.star_buffer.dropped_total
                        ),
                        "star/rollout_success_ratio": (
                            new_qras_this_round / total_outcomes_this_round
                            if total_outcomes_this_round else 0.0
                        ),
                        "star/rollout_total_outcomes": float(total_outcomes_this_round),
                    })
                    logger.info(
                        f"Iter {step}: harvested {new_qras_this_round} successful QRAs "
                        f"(buffer={len(self.star_buffer)}, "
                        f"dropped_total={self.star_buffer.dropped_total})."
                    )

                    if len(self.star_buffer) == 0:
                        logger.warning(
                            f"Iter {step}: buffer empty, skipping SFT step this round."
                        )
                        # skipしてもcheckpointはループ先頭に合わせて継続
                    else:
                        # 3. STaR protocol: SFT 前に model をベースまで巻き戻す
                        #    (累積 buffer を fresh な model + fresh optimizer で学習)
                        await self._reset_training_client_to_base()
                        trigger_eval = step % self.config.eval.eval_interval == 0
                        # Tag the eval cycle's persisted rollouts with the
                        # STaR step so they share the rl_step bucket with
                        # the pre/post-update train rollouts in graphvis.
                        self.evaluate_worker.set_current_rl_step(step)
                        await self._run_star_sft_step(trigger_eval=trigger_eval)

                    # 4. checkpoint
                    if step % self.config.checkpoint.checkpoint_interval == 0:
                        logger.info(f"Saving STaR checkpoint at step {step}...")
                        await self.training_runner.save_checkpoint(
                            name=f"star_{step:04d}",
                            loop_state={"star_step": step},
                            ttl_seconds=self.config.checkpoint.ttl_seconds,
                        )
        finally:
            eval_worker_task.cancel()
            await asyncio.gather(eval_worker_task, return_exceptions=True)

    async def _reset_training_client_to_base(self) -> None:
        """STaR protocol: training_client をベース重み + fresh optimizer まで
        巻き戻す。rollout/eval が使う `shared_sampling_client` には触らないので、
        直前 iter の rollout 用重みは保たれる (SFT だけがリセット対象)。

        `training_runner.training_client` も張り替えて以降の update が新 client に
        向かうようにする。
        """
        ml = self.config.model_loading_settings
        if ml.resume_trainer_path:
            if ml.resume_optimizer_state:
                new_client = await self.service_client.create_training_client_from_state_with_optimizer_async(
                    ml.resume_trainer_path
                )
                reset_kind = "weights + optimizer"
            else:
                new_client = await self.service_client.create_training_client_from_state_async(
                    ml.resume_trainer_path
                )
                reset_kind = "weights only (fresh optimizer)"
        else:
            new_client = await self.service_client.create_lora_training_client_async(
                base_model=ml.model_name,
                rank=ml.lora_rank,
            )
            reset_kind = "new Lora (fresh)"
        self.training_client = new_client
        self.training_runner.training_client = new_client
        logger.info(
            f"STaR protocol: reset training_client to base ({reset_kind})."
        )

    async def _persist_star_qras(
        self,
        qras: list[QRA],
        star_step: int,
        task_id: str,
        task_suite: str,
    ) -> None:
        """Harvest した成功 QRA を star_qnas テーブルに保存。失敗は warn で流す
        (学習ループを止めるほどではない)。"""
        try:
            await self.prisma_client.starqna.create_many(
                data=[
                    {
                        "simple_train_id": self.config.simple_train_id,
                        "star_step": star_step,
                        "task_suite": task_suite,
                        "task_id": task_id,
                        "question": q.question,
                        "reasoning": q.reasoning,
                        "answer": q.answer,
                    }
                    for q in qras
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to persist {len(qras)} star_qnas to DB: {e}")

    async def _run_star_sft_step(self, trigger_eval: bool) -> None:
        # STaR protocol: 累積した成功 QRA を **全件** 使って学習 (sampling しない)。
        # buffer は `STaRSettings.buffer_max_size` で上限を持つので、多すぎる場合は
        # 古いものから自動で dropped されている。
        # Buffer は insertion 順 (古い → 新しい) で返ってくるので shuffle しておく。
        # 同一 task からの連続サンプルが同じ batch に固まるのを避け、iter 境界で
        # データが偏らないようにする。
        all_qras = self.star_buffer.peek_all()
        random.shuffle(all_qras)
        datums = self.training_runner.qras_to_datums(
            all_qras,
            system_prompt=self.system_prompt,
        )
        batch_iter = TrainingRunner.chunk_into_batches(
            datums,
            self.config.star.sft_batch_size,
            self.config.star.sft_epochs_per_round,
        )
        logger.info(
            f"Running STaR SFT step on {len(all_qras)} QRAs (full buffer, shuffled) "
            f"(batch_size={self.config.star.sft_batch_size}, "
            f"epochs={self.config.star.sft_epochs_per_round}, "
            f"trigger_eval={trigger_eval})."
        )
        await self.training_runner.run_sft_steps(
            batch_iter=batch_iter,
            adam_params=self.config.star.sft_adam_params,
            prefix="star_sft",
            trigger_eval=trigger_eval,
        )
