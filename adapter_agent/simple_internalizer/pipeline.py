import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Self

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from oai_utils.tinker.model_helper import get_tokenizer_renderer
from prisma import Prisma
from tinker import Datum
from tinker.types.loss_fn_type import LossFnType
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, TrainOnWhat
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import _remove_mask
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.library.async_rust_doc_analyzer import AsyncRustDocAnalyzer
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.task_net import StudyTask
from adapter_agent.rl.trajectory import prepare_minibatch_simplified

from adapter_agent.simple_internalizer.data_sources import generate_qras_cached
from adapter_agent.simple_internalizer.distilled_qra_manager import DistilledQRAManager
from adapter_agent.simple_internalizer.evaluate_worker import EvaluateWorker
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.ray.ray_rl_worker_pool import RayRLWorkerPool
from adapter_agent.simple_internalizer.rl_worker_pool import RLWorkerPool
from adapter_agent.simple_internalizer.rollout_engine import RolloutEngine, build_solver_system_prompt
from adapter_agent.simple_internalizer.types import PipelineConfig, SeedSuite

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("coder_mcp.runtime.runtime").setLevel(logging.WARNING)


@dataclass
class RLBatchState:
    """Accumulates RL groups with decoupled cadences:
    - `valid_buffer` feeds training (pops exactly `target_valid` when ready).
    - `window_groups` feeds rollout metrics (flushes every `metrics_window`
      groups over ALL produced groups, valid or not).
    Surplus valid groups are retained across calls — none are discarded.
    """

    target_valid: int
    metrics_window: int
    valid_buffer: list[RLGroup] = field(default_factory=list)
    window_groups: list[RLGroup] = field(default_factory=list)

    def add(self, group: RLGroup) -> dict[str, float] | None:
        self.window_groups.append(group)
        if max(group.rewards) > min(group.rewards):
            self.valid_buffer.append(group)
        if len(self.window_groups) >= self.metrics_window:
            return self._flush_window()
        return None

    def _flush_window(self) -> dict[str, float]:
        all_rewards = [r for g in self.window_groups for r in g.rewards]
        num_valid = sum(
            1 for g in self.window_groups if max(g.rewards) > min(g.rewards)
        )
        response_lengths = [
            sum(len(tr.ac.tokens) for tr in traj.transitions)
            for g in self.window_groups
            for traj in g.trajectories
        ]
        metrics = {
            "rollout/mean_reward": sum(all_rewards) / len(all_rewards),
            "rollout/valid_group_ratio": num_valid / len(self.window_groups),
            "rollout/window_size": float(len(self.window_groups)),
        }
        if response_lengths:
            mean_len = sum(response_lengths) / len(response_lengths)
            var_len = (
                statistics.pvariance(response_lengths)
                if len(response_lengths) > 1
                else 0.0
            )
            metrics["rollout/response_length_mean"] = float(mean_len)
            metrics["rollout/response_length_variance"] = float(var_len)
            metrics["rollout/response_length_max"] = float(max(response_lengths))
        self.window_groups = []
        return metrics

    def ready(self) -> bool:
        return len(self.valid_buffer) >= self.target_valid

    def pop_batch(self) -> list[RLGroup]:
        batch = self.valid_buffer[: self.target_valid]
        self.valid_buffer = self.valid_buffer[self.target_valid :]
        return batch


@dataclass
class SimplePipeline:
    """知識内在化 (SFT) と強化学習 (GRPO) を単一プロセスで回すパイプライン。

    インスタンスは `__init__` を直接呼ばず、外部依存のセットアップをまとめる
    `create` クラスメソッド経由で生成する想定。`__init__` は dataclass が
    自動生成するフィールド代入のみを行い、派生値や副作用のあるセットアップは
    すべて `create` 側に集約されている。
    """

    config: PipelineConfig
    """パイプライン全体の設定 (モデル, SFT, RL, バッチサイズ, キャッシュ先等)。"""

    service_client: tinker.ServiceClient
    """Tinker サービスへの接続クライアント。training/sampling client 生成に使う。"""

    training_client: tinker.TrainingClient
    """学習用クライアント。forward/backward/optim_step とチェックポイント保存を担う。"""

    solver_model: TinkerModel
    """RL/評価で推論する対象モデル。renderer, tokenizer, sampling_client をまとめた束。"""

    generator: GeneratorAgent | None
    """初期 SFT 用の QRA を生成するエージェント。`config.sft is None` の場合は None。"""

    executor: InternalizeExecutor
    """生成結果をランタイム上で実行し Verifier で正誤判定する実行器。RL 報酬の源。"""

    ml_logger: MLLogger
    """W&B とローカルログへメトリクスを出す ML ロガー。"""

    prisma_client: Prisma
    """学習過程 (SFT の QA, RL のトラジェクトリ) を DB に記録する Prisma クライアント。"""

    log_dir: Path
    """チェックポイントと ML ログの出力先ディレクトリ。"""

    knowledge_list: list[Knowledge]
    """内在化対象の知識一覧。初期 SFT の QRA 生成と RL のタスク割当てに使う。"""

    shared_sampling_client: SharedSamplingClient
    """複数 RL ワーカー間で共有する「最新重みの SamplingClient」のラッパー。
    GRPO ステップ後に `update_client` で差し替えられ、同じ参照を持つワーカー全員が
    新しい重みを使うようになる。"""

    evaluate_worker: EvaluateWorker
    """`eval_trigger` を購読して評価スイートを走らせるバックグラウンドワーカー。
    `run()` 内で別タスクとして起動される。"""

    _rl_batch_state: RLBatchState
    """有効 RL グループを蓄積し、`config.rl_batch_size` に達したら pop して GRPO
    ステップに回すバッファ。rollout メトリクスの窓平均も兼ねる。"""

    distilled: DistilledQRAManager
    """Study 由来の蒸留 QRA のバッファリング・払い出し、RL all-fail 時の
    replay / study 依頼ルーティングを一括管理するマネージャ。"""

    rollout_engine: RolloutEngine
    """1 問の「サンプリング→parse→実行/検証→DB 記録」を一括で行うエンジン。
    RL ワーカーと EvaluateWorker の両方が共有する。"""

    rl_pool: RLWorkerPool | RayRLWorkerPool
    """RL rollout のワーカープール。`async with` でライフサイクル管理され、
    `next_group()` で RLGroup を順次取り出せる。`config.rl_use_ray=True` のとき
    `RayRLWorkerPool` (マルチプロセス Ray actor 版) に差し替わる。"""

    eval_trigger: asyncio.Event = field(default_factory=asyncio.Event)
    """重み更新後に EvaluateWorker を起動するためのイベント。`create` で生成して
    evaluate_worker と共有する (両者が同一の Event インスタンスを握る必要がある)。"""

    @classmethod
    async def create(
        cls,
        config: PipelineConfig,
        rust_doc_analyzer: AsyncRustDocAnalyzer,
        knowledge_list: list[Knowledge],
        seed_suites: list[SeedSuite],
        study_task_queue: asyncio.Queue[StudyTask] | None = None,
        qra_in_queue: asyncio.Queue[tuple[str, QRA]] | None = None,
    ) -> Self:
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        verifier_model: AgentsSDKModel = config.verifier_model or get_gemini_lite()
        verifier = Verifier(model=verifier_model)
        generator: GeneratorAgent | None = None
        if config.sft is not None:
            generator_model: AgentsSDKModel = (
                config.sft.generator_model or get_gemini()
            )
            generator = GeneratorAgent(
                model=generator_model, rust_doc_analyzer=rust_doc_analyzer
            )

        service_client = tinker.ServiceClient()


        if config.model_loading_settings.resume_trainer_path:
            logger.info(
                f"Loading trainer state from {config.model_loading_settings.resume_trainer_path}..."
            )
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(config.model_loading_settings.resume_trainer_path)
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
                "sft_enabled": config.sft is not None,
                "sft_epochs": config.sft.epochs if config.sft else None,
                "sft_batch_size": config.sft.batch_size if config.sft else None,
                "lora_rank": config.model_loading_settings.lora_rank,
            },
        )

        db_manager = PostgresDB()
        await db_manager.connect()
        client = await db_manager.get_client()
        await client.simpletrainrun.upsert(
            where={"id": config.simple_train_id},
            data={"create": {"id": config.simple_train_id}, "update": {}},
        )

        logger.info(
            f"Using {len(knowledge_list)} knowledge items and {len(seed_suites)} seed suites."
        )
        for s in seed_suites:
            logger.info(
                f"Registered seed suite '{s.name}' with {len(s.tasks)} tasks "
                f"(for_rl={s.for_rl}, for_eval={s.for_eval})."
            )

        runtime_pool = RuntimePool(
            config.runtime_settings, max_size=config.runtime_pool_size
        )
        executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)

        shared_sampling_client = SharedSamplingClient(tinker_model.sampling_client)
        eval_trigger = asyncio.Event()

        rollout_engine = RolloutEngine(
            renderer=tinker_model.renderer,
            executor=executor,
            prisma_client=client,
            system_prompt=build_solver_system_prompt(config.library_name),
            simple_train_id=config.simple_train_id,
        )

        evaluate_worker = EvaluateWorker(
            config=config,
            ml_logger=ml_logger,
            shared_sampling_client=shared_sampling_client,
            eval_suites=[s for s in seed_suites if s.for_eval],
            trigger=eval_trigger,
            rollout_engine=rollout_engine,
        )
        rl_batch_state = RLBatchState(
            target_valid=config.rl_batch_size,
            metrics_window=config.rl_metrics_window,
        )
        distilled = DistilledQRAManager(
            qra_in_queue=qra_in_queue,
            study_task_queue=study_task_queue,
        )

        sampling_params = tinker.SamplingParams(max_tokens=config.max_output_tokens)
        rl_pool: RLWorkerPool | RayRLWorkerPool
        if config.rl_use_ray:
            rl_pool = RayRLWorkerPool(
                num_processes=config.rl_ray_num_processes,
                workers_per_process=config.rl_ray_workers_per_process,
                runtime_pool_size_per_process=config.rl_ray_runtime_pool_size_per_process,
                actor_stagger_s=config.rl_ray_actor_stagger_s,
                runtime_settings=config.runtime_settings,
                verifier_model=verifier_model,
                library_name=config.library_name,
                simple_train_id=config.simple_train_id,
                model_name=config.model_loading_settings.model_name,
                sampling_client=tinker_model.sampling_client,
                seed_suites=seed_suites,
                worker_stagger_s=config.rl_worker_stagger_s,
                num_samples=config.rl_rollout,
                sampling_params=sampling_params,
            )
        else:
            rl_pool = RLWorkerPool(
                rollout_engine=rollout_engine,
                shared_sampling_client=shared_sampling_client,
                seed_suites=seed_suites,
                num_workers=config.rl_worker_count,
                stagger_s=config.rl_worker_stagger_s,
                num_samples=config.rl_rollout,
                sampling_params=sampling_params,
                distilled=distilled,
            )

        return cls(
            config=config,
            service_client=service_client,
            training_client=training_client,
            solver_model=tinker_model,
            generator=generator,
            executor=executor,
            ml_logger=ml_logger,
            prisma_client=client,
            log_dir=log_dir,
            knowledge_list=knowledge_list,
            shared_sampling_client=shared_sampling_client,
            evaluate_worker=evaluate_worker,
            _rl_batch_state=rl_batch_state,
            distilled=distilled,
            rollout_engine=rollout_engine,
            rl_pool=rl_pool,
            eval_trigger=eval_trigger,
        )

    def _print_sft_generation_summary(
        self,
        knowledge_list: list[Knowledge],
        per_knowledge_qras: list[list[QRA]],
    ) -> None:
        assert self.config.sft is not None
        target = self.config.sft.k_sft
        print("\n" + "=" * 80)
        print(f"{'Knowledge Title':<50} | {'Target':<6} | {'Success':<7} | {'Status'}")
        print("-" * 80)
        for k, qras in zip(knowledge_list, per_knowledge_qras):
            success = len(qras)
            if success == target:
                status = "✅ OK"
            elif success > 0:
                status = "⚠️  PARTIAL"
            else:
                status = "❌ FAILED"
            title = (k.title[:47] + "...") if len(k.title) > 50 else k.title
            print(f"{title:<50} | {target:<6} | {success:<7} | {status}")
        print("=" * 80 + "\n")

    async def run(self) -> None:
        eval_worker_task = asyncio.create_task(self.evaluate_worker.run_loop())
        try:
            if self.config.sft is None:
                logger.info("Skipping initial SFT (config.sft is None).")
            elif self.config.model_loading_settings.resume_trainer_path:
                logger.info(
                    f"Skipping initial SFT because resume_trainer_path is provided: {self.config.model_loading_settings.resume_trainer_path}"
                )
            else:
                sft_qras = await self._prepare_sft_qras()
                sft_batch_iter = self._create_sft_batch_iterator(
                    sft_qras, self.config.sft.epochs
                )
                logger.info(f"Running initial SFT for {self.config.sft.epochs} epochs...")
                await self._run_sft_steps(
                    sft_batch_iter, adam_params=self.config.sft.adam_params
                )

                if self.config.sft.save_checkpoint:
                    logger.info("Saving initial SFT checkpoint...")
                    await self._save_checkpoint(
                        name="init_sft",
                        loop_state={"epochs": self.config.sft.epochs},
                    )

                    logger.info("Updated reference client to SFT state.")

            async with self.rl_pool:
                for iteration in range(self.config.max_iterations):
                    logger.info(f"--- Iteration {iteration + 1} (RL) ---")

                    batch_groups = await self._collect_valid_rl_batch()

                    logger.info(
                        f"Running RL (GRPO) update on valid batch of size {len(batch_groups)}..."
                    )
                    rl_step = iteration + 1
                    trigger_eval = rl_step % self.config.eval_interval == 0
                    await self._run_grpo_update(batch_groups, trigger_eval=trigger_eval)

                    await self._maybe_run_distilled_sft_step()

                    if rl_step % self.config.rl_checkpoint_interval == 0:
                        logger.info(f"Saving RL checkpoint at step {rl_step}...")
                        await self._save_checkpoint(
                            name=f"rl_{rl_step:04d}",
                            loop_state={"rl_step": rl_step},
                        )
        finally:
            eval_worker_task.cancel()
            await asyncio.gather(eval_worker_task, return_exceptions=True)

    async def _collect_valid_rl_batch(self) -> list[RLGroup]:
        state = self._rl_batch_state
        logger.info(
            f"Collecting until {state.target_valid} valid RL groups are ready "
            f"(buffered={len(state.valid_buffer)}, window={len(state.window_groups)}/{state.metrics_window})..."
        )
        while not state.ready():
            group = await self.rl_pool.next_group()
            rollout_metrics = state.add(group)
            if rollout_metrics is not None:
                if not self.config.rl_use_ray:
                    pool = self.executor.runtime_pool
                    rollout_metrics["rollout/runtime_pool_size"] = float(pool.current_size)
                    rollout_metrics["rollout/runtime_pool_idle"] = float(pool.idle_size)
                self.ml_logger.log_metrics(rollout_metrics)
        return state.pop_batch()

    async def _run_grpo_update(
        self, valid_groups: list[RLGroup], trigger_eval: bool = True
    ) -> None:
        if not valid_groups:
            return

        traj_groups = [
            TrajectoryGroup(
                trajectories_G=g.trajectories,
                final_rewards_G=g.rewards,
                metrics_G=[{} for _ in g.rewards],
            )
            for g in valid_groups
        ]
        data_D, kl_metrics = await prepare_minibatch_simplified(
            trajectory_groups=traj_groups,
            regularization="group_std",
        )
        if not data_D:
            logger.warning("No valid training data assembled for GRPO step.")
            return

        clean_data_D = [_remove_mask(d) for d in data_D]
        if kl_metrics:
            self.ml_logger.log_metrics({f"rl/{k}": v for k, v in kl_metrics.items()})
        self.ml_logger.log_metrics({"rl/train_batch_size": float(len(valid_groups))})

        batch_iter = (clean_data_D for _ in range(self.config.rl_update_epochs))
        await self._run_training_steps(
            batch_iter=batch_iter,
            prefix="rl",
            loss_fn=self.config.rl_loss_fn,
            adam_params=self.config.rl_adam_params,
        )

        logger.info("Synchronizing sampling weights...")
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.shared_sampling_client.update_client(new_client)
        if isinstance(self.rl_pool, RayRLWorkerPool):
            await self.rl_pool.broadcast_sampling_client(new_client)
        if trigger_eval:
            logger.info("Triggering evaluation.")
            self.eval_trigger.set()

    async def _save_checkpoint(self, name: str, loop_state: dict[str, Any]) -> None:
        await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=name,
            log_path=str(self.log_dir),
            loop_state=loop_state,
            kind="both",
            ttl_seconds=self.config.ttl_seconds,
        )

    async def _run_training_steps(
        self,
        batch_iter: Iterable[list[Datum]],
        prefix: str,
        loss_fn: LossFnType,
        adam_params: tinker.AdamParams,
    ) -> None:
        """Forward/backward + optim_step over batches using look-ahead pipelining:
        the next batch's update is enqueued before the current batch's results
        are awaited, keeping the server busy during client-side logging.
        """
        iterator = iter(batch_iter)
        try:
            first_batch = next(iterator)
        except StopIteration:
            return

        fwd_future = await self.training_client.forward_backward_async(
            data=first_batch, loss_fn=loss_fn
        )
        opt_future = await self.training_client.optim_step_async(adam_params)

        for next_batch in iterator:
            next_fwd_future = await self.training_client.forward_backward_async(
                data=next_batch, loss_fn=loss_fn
            )
            next_opt_future = await self.training_client.optim_step_async(adam_params)

            fwd_res = await fwd_future.result_async()
            await opt_future.result_async()
            self.ml_logger.log_metrics(
                {f"{prefix}/{k}": v for k, v in fwd_res.metrics.items()}
            )

            fwd_future = next_fwd_future
            opt_future = next_opt_future

        fwd_res = await fwd_future.result_async()
        await opt_future.result_async()
        self.ml_logger.log_metrics(
            {f"{prefix}/{k}": v for k, v in fwd_res.metrics.items()}
        )

    async def _run_sft_steps(
        self,
        batch_iter: Iterable[list[Datum]],
        adam_params: tinker.AdamParams,
        prefix: str = "sft",
    ) -> None:
        await self._run_training_steps(
            batch_iter=batch_iter,
            prefix=prefix,
            loss_fn="cross_entropy",
            adam_params=adam_params,
        )

        logger.info(
            f"Synchronizing sampling weights after {prefix} and triggering evaluation..."
        )
        new_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )
        self.shared_sampling_client.update_client(new_client)
        if isinstance(self.rl_pool, RayRLWorkerPool):
            await self.rl_pool.broadcast_sampling_client(new_client)
        self.eval_trigger.set()

    def _distilled_sft_batch_size(self) -> int:
        return (
            self.config.sft.batch_size
            if self.config.sft is not None
            else self.config.rl_batch_size
        )

    def _distilled_sft_adam_params(self) -> tinker.AdamParams:
        return (
            self.config.sft.adam_params
            if self.config.sft is not None
            else self.config.rl_adam_params
        )

    async def _maybe_run_distilled_sft_step(self) -> None:
        """蒸留 QRA を回収し、1 バッチ分溜まっていれば SFT ステップを回す。

        バッファ管理は `DistilledQRAManager` が担当。本メソッドは学習呼び出しの
        オーケストレーションのみを行う。
        """
        drained = self.distilled.ingest()
        batch_size = self._distilled_sft_batch_size()
        qras = self.distilled.try_take_batch(batch_size)
        if qras is None:
            if drained:
                logger.info(
                    f"Distilled SFT buffer: {self.distilled.buffered}/{batch_size} "
                    f"(drained {drained} this iteration)."
                )
            return

        cpt = self.config.sft.cpt if self.config.sft is not None else False
        datums = self._qras_to_datums(qras, cpt=cpt)
        batch_iter = self._chunk_into_batches(datums, batch_size, num_epochs=1)
        logger.info(
            f"Running distilled SFT step on {batch_size} QRAs "
            f"(buffer remaining: {self.distilled.buffered})."
        )
        await self._run_sft_steps(
            batch_iter=batch_iter,
            adam_params=self._distilled_sft_adam_params(),
            prefix="distilled_sft",
        )

    async def _prepare_sft_qras(self) -> list[QRA]:
        assert self.config.sft is not None, (
            "_prepare_sft_qras requires config.sft to be set"
        )
        assert self.generator is not None, (
            "_prepare_sft_qras requires a generator (set when config.sft is not None)"
        )

        per_knowledge_qras: list[list[QRA]] = await asyncio.gather(
            *[
                generate_qras_cached(
                    generator=self.generator,
                    knowledge=k,
                    count=self.config.sft.k_sft,
                    prefix="sft",
                    cache_dir=self.config.cache_dir,
                    generation_concurrency=self.config.generation_concurrency,
                )
                for k in self.knowledge_list
            ]
        )

        self._print_sft_generation_summary(self.knowledge_list, per_knowledge_qras)

        for k, qras in zip(self.knowledge_list, per_knowledge_qras):
            for qra in qras:
                try:
                    existing = await self.prisma_client.simplesftqna.find_first(
                        where={
                            "simple_train_id": self.config.simple_train_id,
                            "knowledge_id": k.id,
                            "question": qra.question,
                        }
                    )
                    if not existing:
                        await self.prisma_client.simplesftqna.create(
                            data={
                                "simple_train_id": self.config.simple_train_id,
                                "knowledge_id": k.id,
                                "knowledge_title": k.title,
                                "question": qra.question,
                                "reasoning": qra.reasoning,
                                "answer": qra.answer,
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to record SFT QRA to DB: {e}")

        return [q for qs in per_knowledge_qras for q in qs]

    def _qras_to_datums(self, qras: list[QRA], cpt: bool) -> list[Datum]:
        """Convert QRA samples into Datums suitable for cross-entropy SFT.

        Two conversation shapes are supported: `cpt=True` uses a custom example
        format (system + question + answer) with training limited to the answer
        turn; `cpt=False` uses the standard user/assistant shape with a thinking
        part on the assistant side, and trains on the last assistant message.
        """
        _, renderer = get_tokenizer_renderer(
            self.training_client, self.config.model_loading_settings.model_name
        )
        if cpt:
            return [
                conversation_to_datum(
                    conversation=[
                        Message(
                            role="system",
                            content=f"これは{self.config.library_name}の教育資料から抜粋した練習問題です。",
                            trainable=False,
                        ),
                        Message(role="example_question", content=q.question, trainable=False),
                        Message(role="example_answer", content=q.answer, trainable=True),
                    ],
                    renderer=renderer,
                    train_on_what=TrainOnWhat.CUSTOMIZED,
                    max_length=None,
                )
                for q in qras
            ]
        return [
            conversation_to_datum(
                conversation=[
                    Message(role="user", content=q.question),
                    Message(
                        role="assistant",
                        content=[
                            ThinkingPart(type="thinking", thinking=q.reasoning),
                            TextPart(type="text", text=f"\n\n{q.answer}"),
                        ],
                    ),
                ],
                renderer=renderer,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                max_length=None,
            )
            for q in qras
        ]

    @staticmethod
    def _chunk_into_batches(
        datums: list[Datum], batch_size: int, num_epochs: int
    ) -> Iterable[list[Datum]]:
        """Wrap-around chunking: replicate for epochs, then pad the tail."""
        if not datums:
            return iter([])
        all_datums = datums * num_epochs
        num_batches = (len(all_datums) + batch_size - 1) // batch_size
        batches: list[list[Datum]] = []
        for i in range(num_batches):
            batch = all_datums[i * batch_size : (i + 1) * batch_size]
            if len(batch) < batch_size:
                batch.extend(all_datums[: batch_size - len(batch)])
            batches.append(batch)
        logger.info(f"Created {len(batches)} batches for {num_epochs} epochs.")
        return iter(batches)

    def _create_sft_batch_iterator(
        self, sft_qras: list[QRA], num_epochs: int
    ) -> Iterable[list[Datum]]:
        assert self.config.sft is not None, (
            "_create_sft_batch_iterator requires config.sft to be set"
        )
        datums = self._qras_to_datums(sft_qras, cpt=self.config.sft.cpt)
        return self._chunk_into_batches(datums, self.config.sft.batch_size, num_epochs)
