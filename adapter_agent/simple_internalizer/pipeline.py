import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Self

import tinker
from oai_utils import AgentsSDKModel
from oai_utils.tinker import TinkerModel, setup_tinkermodel
from prisma import Prisma
from tinker import Datum
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.ml_log import Logger as MLLogger

from adapter_agent.data import QRA
from adapter_agent.hierarchical.agent.generator import GeneratorAgent
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import _remove_mask
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
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
from adapter_agent.simple_internalizer.training_runner import TrainingRunner
from adapter_agent.simple_internalizer.types import PipelineConfig, SeedSuite
from adapter_agent.util.config_util import flatten_config
from adapter_agent.util.logger_util import ClockCycleFilteredLogger

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
    Surplus valid groups are retained across calls — none are discarded
    except by `ready()`, which prunes groups whose `sampling_client_version`
    lags the current version by more than `max_version_lag` (off-policy guard).
    """

    target_valid: int
    metrics_window: int
    max_version_lag: int = 1
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
        num_any_correct = sum(
            1 for g in self.window_groups if max(g.rewards) > 0
        )
        response_lengths = [
            sum(len(tr.ac.tokens) for tr in traj.transitions)
            for g in self.window_groups
            for traj in g.trajectories
        ]
        versions = [g.sampling_client_version for g in self.window_groups]
        metrics = {
            "rollout/mean_reward": sum(all_rewards) / len(all_rewards),
            "rollout/valid_group_ratio": num_valid / len(self.window_groups),
            "rollout/any_correct_ratio": num_any_correct / len(self.window_groups),
            "rollout/window_size": float(len(self.window_groups)),
            "rollout/sampling_client_version_mean": sum(versions) / len(versions),
            "rollout/sampling_client_version_min": float(min(versions)),
            "rollout/sampling_client_version_max": float(max(versions)),
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

    def ready(self, current_version: int) -> bool:
        before = len(self.valid_buffer)
        self.valid_buffer = [
            g
            for g in self.valid_buffer
            if current_version - g.sampling_client_version <= self.max_version_lag
        ]
        pruned = before - len(self.valid_buffer)
        if pruned > 0:
            logger.info(
                f"Pruned {pruned} stale group(s) from valid_buffer "
                f"(current_version={current_version}, max_lag={self.max_version_lag})."
            )
        return len(self.valid_buffer) >= self.target_valid

    def pop_batch(self) -> list[RLGroup]:
        batch = self.valid_buffer[: self.target_valid]
        self.valid_buffer = self.valid_buffer[self.target_valid :]
        return batch


@dataclass
class SimplePipeline:
    """知識内在化 (SFT) と強化学習 (GRPO) を単一プロセスで回すパイプライン。

    インスタンスは `__init__` を直接呼ばず、外部依存のセットアップをまとめる
    `create` クラスメソッド経由で生成する想定。学習プリミティブ (SFT ステップ,
    checkpoint, QRA→Datum 変換, weight 同期) は `TrainingRunner` に委譲し、
    pipeline 本体は RL/GRPO 固有のオーケストレーションに集中する。
    """

    config: PipelineConfig
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    solver_model: TinkerModel
    generator: GeneratorAgent | None
    executor: InternalizeExecutor
    ml_logger: MLLogger
    prisma_client: Prisma
    log_dir: Path
    knowledge_list: list[Knowledge]
    shared_sampling_client: SharedSamplingClient
    evaluate_worker: EvaluateWorker
    _rl_batch_state: RLBatchState
    distilled: DistilledQRAManager
    rollout_engine: RolloutEngine
    rl_pool: RLWorkerPool | RayRLWorkerPool
    training_runner: TrainingRunner
    eval_trigger: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    async def create(
        cls,
        config: PipelineConfig,
        knowledge_list: list[Knowledge],
        seed_suites: list[SeedSuite],
        study_task_queue: asyncio.Queue[StudyTask] | None = None,
        qra_in_queue: asyncio.Queue[tuple[str, QRA]] | None = None,
    ) -> Self:
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        verifier_model: AgentsSDKModel = config.rollout.verifier_model or get_gemini_lite()
        verifier = Verifier(model=verifier_model)
        generator: GeneratorAgent | None = None
        if config.sft is not None:
            generator_model: AgentsSDKModel = (
                config.sft.generator_model or get_gemini()
            )
            generator = GeneratorAgent(
                model=generator_model
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
            config=flatten_config(config),
        )
        ml_logger = ClockCycleFilteredLogger(ml_logger)

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
            config.rollout.runtime_settings, max_size=config.rollout.runtime_pool_size
        )
        executor = InternalizeExecutor(runtime_pool=runtime_pool, verifier=verifier)

        shared_sampling_client = SharedSamplingClient(tinker_model.sampling_client)
        eval_trigger = asyncio.Event()

        rollout_engine = RolloutEngine(
            renderer=tinker_model.renderer,
            executor=executor,
            system_prompt=build_solver_system_prompt(config.library_name),
        )

        evaluate_worker = EvaluateWorker(
            ml_logger=ml_logger,
            shared_sampling_client=shared_sampling_client,
            eval_suites=[s for s in seed_suites if s.for_eval],
            trigger=eval_trigger,
            rollout_engine=rollout_engine,
            eval_concurrency=config.eval.eval_concurrency,
            eval_rollout=config.eval.eval_rollout,
            max_output_tokens=config.rollout.max_output_tokens,
        )
        rl_batch_state = RLBatchState(
            target_valid=config.rl_batch_size,
            metrics_window=config.rl_metrics_window,
            max_version_lag=config.rl_max_version_lag,
        )
        distilled = DistilledQRAManager(
            qra_in_queue=qra_in_queue,
            study_task_queue=study_task_queue,
        )

        sampling_params = tinker.SamplingParams(
            max_tokens=config.rollout.max_output_tokens
        )
        rl_pool: RLWorkerPool | RayRLWorkerPool
        if config.rollout.use_ray:
            rl_pool = RayRLWorkerPool(
                num_processes=config.rollout.ray_num_processes,
                workers_per_process=config.rollout.ray_workers_per_process,
                runtime_pool_size_per_process=config.rollout.ray_runtime_pool_size_per_process,
                actor_stagger_s=config.rollout.ray_actor_stagger_s,
                runtime_settings=config.rollout.runtime_settings,
                verifier_model=verifier_model,
                library_name=config.library_name,
                simple_train_id=config.simple_train_id,
                model_name=config.model_loading_settings.model_name,
                sampling_client=tinker_model.sampling_client,
                seed_suites=seed_suites,
                worker_stagger_s=config.rollout.worker_stagger_s,
                num_samples=config.rollout.num_samples,
                sampling_params=sampling_params,
            )
        else:
            rl_pool = RLWorkerPool(
                rollout_engine=rollout_engine,
                shared_sampling_client=shared_sampling_client,
                seed_suites=seed_suites,
                num_workers=config.rollout.worker_count,
                stagger_s=config.rollout.worker_stagger_s,
                num_samples=config.rollout.num_samples,
                sampling_params=sampling_params,
                distilled=distilled,
            )

        # Weight sync 時に RL pool (Ray 版) へブロードキャストするフック
        async def _broadcast_hook(new_client: tinker.SamplingClient) -> None:
            if isinstance(rl_pool, RayRLWorkerPool):
                await rl_pool.broadcast_sampling_client(new_client)

        training_runner = TrainingRunner(
            training_client=training_client,
            shared_sampling_client=shared_sampling_client,
            ml_logger=ml_logger,
            log_dir=log_dir,
            model_name=config.model_loading_settings.model_name,
            eval_trigger=eval_trigger,
            broadcast_hook=_broadcast_hook,
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
            training_runner=training_runner,
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
                await self.training_runner.run_sft_steps(
                    sft_batch_iter, adam_params=self.config.sft.adam_params
                )

                if self.config.sft.save_checkpoint:
                    logger.info("Saving initial SFT checkpoint...")
                    await self.training_runner.save_checkpoint(
                        name="init_sft",
                        loop_state={"epochs": self.config.sft.epochs},
                        ttl_seconds=self.config.checkpoint.ttl_seconds,
                    )
                    logger.info("Updated reference client to SFT state.")

            async with self.rl_pool:
                for iteration in range(self.config.max_iterations):
                    rl_step = iteration + 1
                    logger.info(f"--- Iteration {rl_step} (RL) ---")

                    batch_groups = await self._collect_valid_rl_batch()

                    trigger_eval = rl_step % self.config.eval.eval_interval == 0



                    logger.info(
                        f"Running RL (GRPO) update on valid batch of size {len(batch_groups)}..."
                    )
                    await self._run_grpo_update(batch_groups, trigger_eval=trigger_eval)

                    await self._maybe_run_distilled_sft_step()

                    if rl_step % self.config.checkpoint.checkpoint_interval == 0:
                        logger.info(f"Saving RL checkpoint at step {rl_step}...")
                        await self.training_runner.save_checkpoint(
                            name=f"rl_{rl_step:04d}",
                            loop_state={"rl_step": rl_step},
                            ttl_seconds=self.config.checkpoint.ttl_seconds,
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
        while not state.ready(self.shared_sampling_client.version):
            group = await self.rl_pool.next_group()
            rollout_metrics = state.add(group)
            if rollout_metrics is not None:
                if not self.config.rollout.use_ray:
                    pool = self.executor.runtime_pool
                    rollout_metrics["rollout/runtime_pool_size"] = float(pool.current_size)
                    rollout_metrics["rollout/runtime_pool_idle"] = float(pool.idle_size)
                self.ml_logger.log_metrics(rollout_metrics)
        return state.pop_batch()

    async def _run_grpo_update(
        self, valid_groups: list[RLGroup], trigger_eval: bool = True
    ) -> None:
        if self.config.rl_skip_update:
            logger.info(
                f"rl_skip_update=True: skipping GRPO update on batch of size {len(valid_groups)} "
                f"(rollouts and eval still run)."
            )

            return
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
        step_metrics: dict[str, float] = {"rl/train_batch_size": float(len(valid_groups))}
        if kl_metrics:
            step_metrics.update({f"rl/{k}": v for k, v in kl_metrics.items()})

        batch_versions = [g.sampling_client_version for g in valid_groups]
        current_version = self.training_runner.shared_sampling_client.version
        version_mean = sum(batch_versions) / len(batch_versions)
        step_metrics["rl/sampler_version_mean"] = version_mean
        step_metrics["rl/sampler_version_min"] = float(min(batch_versions))
        step_metrics["rl/sampler_version_max"] = float(max(batch_versions))
        step_metrics["rl/sampler_version_lag_mean"] = float(current_version) - version_mean
        step_metrics["rl/sampler_version_lag_max"] = float(current_version - min(batch_versions))

        batch_iter = (clean_data_D for _ in range(self.config.rl_update_epochs))
        await self.training_runner.run_training_steps(
            batch_iter=batch_iter,
            prefix="rl",
            loss_fn=self.config.rl_loss_fn,
            adam_params=self.config.rl_adam_params,
            extra_metrics=step_metrics,
        )

        logger.info("Synchronizing sampling weights...")
        await self.training_runner.sync_sampling_weights()
        if trigger_eval:
            logger.info("Triggering evaluation.")
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
        """蒸留 QRA を回収し、1 バッチ分溜まっていれば SFT ステップを回す。"""
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
        datums = self.training_runner.qras_to_datums(
            qras, cpt=cpt, library_name=self.config.library_name
        )
        batch_iter = TrainingRunner.chunk_into_batches(datums, batch_size, num_epochs=1)
        logger.info(
            f"Running distilled SFT step on {batch_size} QRAs "
            f"(buffer remaining: {self.distilled.buffered})."
        )
        await self.training_runner.run_sft_steps(
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

    def _create_sft_batch_iterator(
        self, sft_qras: list[QRA], num_epochs: int
    ) -> Iterable[list[Datum]]:
        assert self.config.sft is not None, (
            "_create_sft_batch_iterator requires config.sft to be set"
        )
        datums = self.training_runner.qras_to_datums(
            sft_qras, cpt=self.config.sft.cpt, library_name=self.config.library_name
        )
        return TrainingRunner.chunk_into_batches(
            datums, self.config.sft.batch_size, num_epochs
        )
