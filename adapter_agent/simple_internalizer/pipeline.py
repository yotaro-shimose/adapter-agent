import asyncio
import logging
import math
import random
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
from adapter_agent.hierarchical.agent.verifier import Verifier
from adapter_agent.hierarchical.grpo import _remove_mask
from adapter_agent.hierarchical.state import RLGroup
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.model_helper import get_gemini, get_gemini_lite
from adapter_agent.rl.env.runtime_pool import RuntimePool
from adapter_agent.rl.postgres_db import PostgresDB
from adapter_agent.rl.shared_sampling_client import SharedSamplingClient
from adapter_agent.rl.trajectory import prepare_minibatch_simplified

from adapter_agent.simple_internalizer.evaluate_worker import EvaluateWorker
from adapter_agent.simple_internalizer.executor import InternalizeExecutor
from adapter_agent.simple_internalizer.ray.ray_rl_worker_pool import RayRLWorkerPool
from adapter_agent.simple_internalizer.rl_worker_pool import RLWorkerPool
from adapter_agent.simple_internalizer.rollout_engine import RolloutEngine, build_solver_system_prompt
from adapter_agent.simple_internalizer.training_runner import TrainingRunner
from adapter_agent.simple_internalizer.types import PipelineConfig, SeedSuite, SftSuite
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
    executor: InternalizeExecutor
    ml_logger: MLLogger
    prisma_client: Prisma
    log_dir: Path
    knowledge_list: list[Knowledge]
    sft_suites: list[SftSuite]
    has_rl_suites: bool
    shared_sampling_client: SharedSamplingClient
    evaluate_worker: EvaluateWorker
    _rl_batch_state: RLBatchState | None
    rollout_engine: RolloutEngine
    rl_pool: RLWorkerPool | RayRLWorkerPool | None
    training_runner: TrainingRunner
    rl_task_count: int = 0
    eval_trigger: asyncio.Event = field(default_factory=asyncio.Event)

    @classmethod
    async def create(
        cls,
        config: PipelineConfig,
        knowledge_list: list[Knowledge],
        seed_suites: list[SeedSuite],
        sft_suites: list[SftSuite] | None = None,
    ) -> Self:
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        verifier_model: AgentsSDKModel = config.rollout.verifier_model or get_gemini_lite()
        verifier = Verifier(model=verifier_model, library_name=config.library_name)

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
        sampling_params = tinker.SamplingParams(
            max_tokens=config.rollout.max_output_tokens
        )
        rl_batch_state: RLBatchState | None = None
        rl_pool: RLWorkerPool | RayRLWorkerPool | None = None
        if config.rl is not None:
            rl_batch_state = RLBatchState(
                target_valid=config.rl.batch_size,
                metrics_window=config.rl.metrics_window,
                max_version_lag=config.rl.max_version_lag,
            )
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
                    rl_seed=config.rl.rl_seed,
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
            executor=executor,
            ml_logger=ml_logger,
            prisma_client=client,
            log_dir=log_dir,
            knowledge_list=knowledge_list,
            sft_suites=sft_suites or [],
            has_rl_suites=any(s.for_rl for s in seed_suites),
            shared_sampling_client=shared_sampling_client,
            evaluate_worker=evaluate_worker,
            _rl_batch_state=rl_batch_state,
            rollout_engine=rollout_engine,
            rl_pool=rl_pool,
            training_runner=training_runner,
            rl_task_count=sum(len(s.tasks) for s in seed_suites if s.for_rl),
            eval_trigger=eval_trigger,
        )

    async def run(self) -> None:
        rl_active = self.config.rl is not None and self.has_rl_suites
        eval_worker_task: asyncio.Task | None = None
        if rl_active:
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
                    sft_batch_iter,
                    adam_params=self.config.sft.adam_params,
                    trigger_eval=rl_active,
                )

                if self.config.sft.save_checkpoint:
                    logger.info("Saving initial SFT checkpoint...")
                    await self.training_runner.save_checkpoint(
                        name="init_sft",
                        loop_state={"epochs": self.config.sft.epochs},
                        ttl_seconds=self.config.checkpoint.ttl_seconds,
                    )
                    logger.info("Updated reference client to SFT state.")

            if self.config.rl is None or not self.has_rl_suites:
                reason = (
                    "config.rl is None"
                    if self.config.rl is None
                    else "no RL seed suites"
                )
                logger.info(
                    f"Skipping RL loop ({reason}); running a final evaluation and exiting."
                )
                if self.config.sft is None:
                    # SFT didn't sync weights for us — sync once so eval sees the
                    # current trainer state (may matter on resume_trainer_path).
                    await self.training_runner.sync_sampling_weights()
                await self.evaluate_worker.run_once()
                return

            assert self.rl_pool is not None  # ensured by config.rl is not None
            rl_cfg = self.config.rl
            if rl_cfg.num_passes is not None:
                if self.rl_task_count == 0:
                    raise RuntimeError(
                        f"RLConfig.num_passes={rl_cfg.num_passes} requires a non-empty "
                        f"RL task pool (rl_task_count={self.rl_task_count})."
                    )
                iters_per_pass = math.ceil(self.rl_task_count / rl_cfg.batch_size)
                effective_max_iters = rl_cfg.num_passes * iters_per_pass
                logger.info(
                    f"RL loop: num_passes={rl_cfg.num_passes} × "
                    f"iters_per_pass={iters_per_pass} = {effective_max_iters} iterations "
                    f"(rl_task_count={self.rl_task_count}, batch_size={rl_cfg.batch_size})."
                )
            else:
                assert rl_cfg.max_iterations is not None  # RLConfig.__post_init__
                effective_max_iters = rl_cfg.max_iterations
                logger.info(f"RL loop: max_iterations={effective_max_iters} (step-based).")
            async with self.rl_pool:
                for iteration in range(effective_max_iters):
                    rl_step = iteration + 1
                    logger.info(f"--- Iteration {rl_step} (RL) ---")

                    batch_groups = await self._collect_valid_rl_batch(rl_step=rl_step)

                    trigger_eval = rl_step % self.config.eval.eval_interval == 0



                    logger.info(
                        f"Running RL (GRPO) update on valid batch of size {len(batch_groups)}..."
                    )
                    await self._run_grpo_update(batch_groups, trigger_eval=trigger_eval)

                    if rl_step % self.config.checkpoint.checkpoint_interval == 0:
                        logger.info(f"Saving RL checkpoint at step {rl_step}...")
                        await self.training_runner.save_checkpoint(
                            name=f"rl_{rl_step:04d}",
                            loop_state={"rl_step": rl_step},
                            ttl_seconds=self.config.checkpoint.ttl_seconds,
                        )

                if (
                    effective_max_iters > 0
                    and effective_max_iters % self.config.checkpoint.checkpoint_interval != 0
                ):
                    logger.info(
                        f"Saving final RL checkpoint at step {effective_max_iters}..."
                    )
                    await self.training_runner.save_checkpoint(
                        name=f"rl_{effective_max_iters:04d}",
                        loop_state={"rl_step": effective_max_iters},
                        ttl_seconds=self.config.checkpoint.ttl_seconds,
                    )
        finally:
            if eval_worker_task is not None:
                eval_worker_task.cancel()
                await asyncio.gather(eval_worker_task, return_exceptions=True)

    async def _collect_valid_rl_batch(self, rl_step: int) -> list[RLGroup]:
        assert self._rl_batch_state is not None and self.rl_pool is not None
        state = self._rl_batch_state
        logger.info(
            f"Collecting until {state.target_valid} valid RL groups are ready "
            f"(buffered={len(state.valid_buffer)}, window={len(state.window_groups)}/{state.metrics_window})..."
        )
        # Buffer every group produced this iter so we can persist them in a single
        # `create_many` after the loop — including filtered (all-same-reward)
        # groups, since those are useful for audit.
        produced: list[RLGroup] = []
        while not state.ready(self.shared_sampling_client.version):
            group = await self.rl_pool.next_group()
            produced.append(group)
            rollout_metrics = state.add(group)
            if rollout_metrics is not None:
                if not self.config.rollout.use_ray:
                    pool = self.executor.runtime_pool
                    rollout_metrics["rollout/runtime_pool_size"] = float(pool.current_size)
                    rollout_metrics["rollout/runtime_pool_idle"] = float(pool.idle_size)
                self.ml_logger.log_metrics(rollout_metrics)
        await self._persist_rollouts(rl_step, produced)
        return state.pop_batch()

    async def _persist_rollouts(self, rl_step: int, groups: list[RLGroup]) -> None:
        """Bulk-write rollouts to the `simple_rl_rollouts` table for later audit.

        One `create_many` call per iteration. Wrapped in try/except so a DB
        failure logs a warning but never kills training. Skips groups that
        lack audit metadata (e.g. Ray rollouts — TODO).
        """
        records = []
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
                        "simple_train_id": self.config.simple_train_id,
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
            await self.prisma_client.simplerlrollout.create_many(data=records)
        except Exception as e:
            logger.warning(
                f"Failed to persist {len(records)} RL rollouts at step {rl_step}: {e}"
            )

    async def _run_grpo_update(
        self, valid_groups: list[RLGroup], trigger_eval: bool = True
    ) -> None:
        assert self.config.rl is not None  # only called from RL loop
        if self.config.rl.skip_update:
            logger.info(
                f"rl.skip_update=True: skipping GRPO update on batch of size {len(valid_groups)} "
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

        batch_iter = (clean_data_D for _ in range(self.config.rl.update_epochs))
        await self.training_runner.run_training_steps(
            batch_iter=batch_iter,
            prefix="rl",
            loss_fn=self.config.rl.loss_fn,
            adam_params=self.config.rl.adam_params,
            extra_metrics=step_metrics,
        )

        logger.info("Synchronizing sampling weights...")
        await self.training_runner.sync_sampling_weights()
        if trigger_eval:
            logger.info("Triggering evaluation.")
            self.eval_trigger.set()

    async def _prepare_sft_qras(self) -> list[QRA]:
        """Flatten the caller-supplied SFT suites into a single shuffled
        QRA pool. This pipeline is data-source-agnostic — the caller
        picks loaders (granular, sft_cache, study-root, ...) and passes
        the resulting `SftSuite` list at construction time."""
        assert self.config.sft is not None, (
            "_prepare_sft_qras requires config.sft to be set"
        )
        all_qras: list[QRA] = [q for s in self.sft_suites for q in s.qras]
        rng = random.Random(self.config.sft.sft_seed)
        rng.shuffle(all_qras)

        breakdown = ", ".join(
            f"{s.name}={len(s.qras)}" for s in self.sft_suites
        ) or "(none)"
        logger.info(
            f"SFT pool: {breakdown} -> {len(all_qras)} total "
            f"(shuffled with seed={self.config.sft.sft_seed})."
        )
        return all_qras

    def _create_sft_batch_iterator(
        self, sft_qras: list[QRA], num_epochs: int
    ) -> Iterable[list[Datum]]:
        assert self.config.sft is not None, (
            "_create_sft_batch_iterator requires config.sft to be set"
        )
        datums = self.training_runner.qras_to_datums(sft_qras)
        return TrainingRunner.chunk_into_batches(
            datums, self.config.sft.batch_size, num_epochs
        )
