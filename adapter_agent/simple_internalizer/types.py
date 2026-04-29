from dataclasses import dataclass, field
from pathlib import Path

import tinker
from oai_utils import AgentsSDKModel
from pydantic import BaseModel
from tinker.types.loss_fn_type import LossFnType

from adapter_agent.hierarchical.types import Task
from adapter_agent.rl.config import ModelLoadingSettings
from adapter_agent.rl.env.runtime_settings import RuntimeSettings


class SeedSuite(BaseModel):
    name: str  # eval metric suite name & RL source tag for DB logging
    tasks: list[Task]
    for_rl: bool = True
    for_eval: bool = True


@dataclass
class RLSource:
    """Origin tag for an RL task, used to group tasks by suite."""

    id: str
    title: str


@dataclass
class RuntimeExecutionResult:
    execution_output: str
    tree_output: str
    exit_success: bool


@dataclass
class VerificationOutcome:
    success: bool
    execution_output: str
    verification_output: str


@dataclass
class EvalResult:
    success_count: int
    total_count: int
    response_lengths: list[int] = field(default_factory=list)


# --- Shared SubConfigs (composed by both PipelineConfig and STaRPipelineConfig) ---


@dataclass
class RolloutSettings:
    """Rollout 生成に関する設定。RL と STaR の両方で使う。

    `num_samples` は 1 問あたりのサンプル数。RL の `rl_rollout`、STaR の
    per-task N サンプルに相当する。Ray 有効時は N プロセス × 各プロセス内
    並列ワーカーで分散実行。in-process 時は `worker_count` 個の asyncio
    ワーカーで回す。
    """

    runtime_settings: RuntimeSettings
    num_samples: int = 8
    max_output_tokens: int = 6000
    verifier_model: AgentsSDKModel | None = None
    runtime_pool_size: int = 50
    worker_count: int = 50
    worker_stagger_s: float = 2.0
    use_ray: bool = False
    ray_num_processes: int = 32
    ray_workers_per_process: int = 2
    ray_runtime_pool_size_per_process: int = 6
    ray_actor_stagger_s: float = 1.0


@dataclass
class EvalSettings:
    """評価 (EvaluateWorker) 設定。"""

    eval_rollout: int = 4
    eval_interval: int = 1
    eval_concurrency: int = 48


@dataclass
class CheckpointSettings:
    """チェックポイント保存の設定。"""

    checkpoint_interval: int = 10
    ttl_seconds: int = 604800


@dataclass
class STaRSettings:
    """STaR 固有の設定 (バッファ上限 + SFT ハイパラ)。"""

    buffer_max_size: int = 5000
    sft_window: int = 256
    sft_batch_size: int = 32
    sft_epochs_per_round: int = 1
    sft_adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=5e-6)
    )
    cpt: bool = False


@dataclass
class SFTConfig:
    """SFT-stage settings. Presence of this config enables the SFT stage;
    set `PipelineConfig.sft = None` to skip SFT entirely."""

    k_sft: int = 32
    epochs: int = 1
    batch_size: int = 32
    save_checkpoint: bool = False
    adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-4)
    )
    cpt: bool = False
    generator_model: AgentsSDKModel | None = None


@dataclass
class PipelineConfig:
    """RL (SimplePipeline) 用設定。SubConfig を合成で保持し、RL/GRPO 固有の
    フィールドだけ自前で持つ。"""

    # Identity / checkpoint resume
    simple_train_id: str
    library_name: str
    model_loading_settings: ModelLoadingSettings

    # Composed sub-configs
    rollout: RolloutSettings
    eval: EvalSettings
    checkpoint: CheckpointSettings

    # Pipeline flow
    max_iterations: int = 50
    cache_dir: Path = Path(".cache/simple_internalizer")
    generation_concurrency: int = 400

    # SFT (initial stage)
    sft: SFTConfig | None = None

    # RL (GRPO) specific
    rl_adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-5)
    )
    rl_loss_fn: LossFnType = "importance_sampling"
    rl_batch_size: int = 48
    rl_update_epochs: int = 1
    rl_metrics_window: int = 50
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0


@dataclass
class STaRPipelineConfig:
    """STaR (STaRPipeline) 用設定。PipelineConfig とは独立。共通部分は同じ
    SubConfig を再利用して DRY。GRPO 固有フィールドは持たない。"""

    # Identity / checkpoint resume
    simple_train_id: str
    library_name: str
    model_loading_settings: ModelLoadingSettings

    # Composed sub-configs
    rollout: RolloutSettings
    eval: EvalSettings
    checkpoint: CheckpointSettings
    star: STaRSettings

    # Pipeline flow
    max_iterations: int = 50
    cache_dir: Path = Path(".cache/simple_internalizer")
