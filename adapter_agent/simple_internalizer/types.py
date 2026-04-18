from dataclasses import dataclass, field
from pathlib import Path
import tinker
from tinker.types.loss_fn_type import LossFnType
from adapter_agent.hierarchical.types import Knowledge
from adapter_agent.data import QRA
from adapter_agent.rl.env.runtime_settings import RuntimeSettings
from adapter_agent.rl.config import ModelLoadingSettings

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

@dataclass
class PreparedTasks:
    sft_qras: list[QRA]
    eval_tasks: list[tuple[Knowledge, QRA]]

@dataclass
class PipelineConfig:
    runtime_pool_size: int
    rl_worker_count: int
    eval_concurrency: int
    generation_concurrency: int
    simple_train_id: str
    knowledge_list: list[Knowledge]
    library_name: str
    runtime_settings: RuntimeSettings
    model_loading_settings: ModelLoadingSettings
    k_sft: int = 32
    k_eval: int = 1
    eval_rollout: int = 4
    init_sft_steps: int = 5
    iter_sft_steps: int = 1
    sft_batch_size: int = 32
    adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-4)
    )
    max_iterations: int = 50
    cache_dir: Path = Path(".cache/simple_internalizer")
    k_rl: int = 4
    rl_rollout: int = 8
    rl_adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-5)
    )
    rl_loss_fn: LossFnType = "importance_sampling"
    rl_batch_size: int = 48
    rl_worker_stagger_s: float = 2.0
    kl_penalty_coef: float = 0.05
    kl_discount_factor: float = 1.0
    extra_eval_suites: dict[str, list[str]] = field(default_factory=dict)
    stop_grpo: bool = False
    ttl_seconds: int = 604800
    max_fix_attempts: int = 3
    sft_epochs: int = 1
    cpt: bool = False
