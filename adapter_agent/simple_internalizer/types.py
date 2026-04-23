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
    """Origin tag for an RL task, written to simpletrajectory for grouping."""

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
    runtime_pool_size: int
    rl_worker_count: int
    eval_concurrency: int
    generation_concurrency: int
    simple_train_id: str
    library_name: str
    runtime_settings: RuntimeSettings
    model_loading_settings: ModelLoadingSettings
    sft: SFTConfig | None = None
    eval_rollout: int = 4
    eval_interval: int = 1
    max_iterations: int = 50
    rl_checkpoint_interval: int = 10
    cache_dir: Path = Path(".cache/simple_internalizer")
    rl_rollout: int = 8
    rl_adam_params: tinker.AdamParams = field(
        default_factory=lambda: tinker.AdamParams(learning_rate=1e-5)
    )
    rl_loss_fn: LossFnType = "importance_sampling"
    rl_batch_size: int = 48
    rl_update_epochs: int = 1
    rl_metrics_window: int = 50
    rl_worker_stagger_s: float = 2.0
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    ttl_seconds: int = 604800
    verifier_model: AgentsSDKModel | None = None
    max_output_tokens: int = 6000
