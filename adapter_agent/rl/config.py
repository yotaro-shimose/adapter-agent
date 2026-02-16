from datetime import datetime
from pathlib import Path

import tinker
from pydantic import BaseModel, Field

from adapter_agent.hierarchical.gh import Library
from adapter_agent.rl.advantage import AdvantageRegularizer
from tinker.types.loss_fn_type import LossFnType


class OptimizerParams(BaseModel):
    adam_params: tinker.AdamParams
    loss_fn: LossFnType
    advantage_regularizer: AdvantageRegularizer = "output_token"
    num_steps: int = 1
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0


class RolloutParams(BaseModel):
    num_groups_per_batch: int = 2
    num_rollout_workers: int = 1
    rollouts_per_question: int = 4
    per_group_concurrency: int = 4
    temperature: float = 0.7


class EnvParams(BaseModel):
    max_turns: int = 5
    r_min: float = 0.5
    library: Library
    image_name: str


class ExperimentSettings(BaseModel):
    experiment_name: str = Field(
        default_factory=lambda: (
            f"Adapter_Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    wandb_project: str | None = "Adapter Agent"
    ttl_seconds: int = 604800  # 7 days

    def log_root(self) -> Path:
        return Path("logs") / "Adapter_Agent" / self.experiment_name


class ModelLoadingSettings(BaseModel):
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    resume_trainer_path: str | None = None
    resume_sampler_path: str | None = None
    lora_rank: int = 32


class RLConfig(BaseModel):
    experiment_setting: ExperimentSettings = ExperimentSettings()
    optimizer_params: OptimizerParams = OptimizerParams(
        adam_params=tinker.AdamParams(
            learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8
        ),
        loss_fn="ppo",
    )
    rollout_params: RolloutParams = RolloutParams(
        num_groups_per_batch=2,
        num_rollout_workers=1,
        rollouts_per_question=4,
        per_group_concurrency=1,
    )
    env_params: EnvParams = EnvParams(
        max_turns=5,
        r_min=0.5,
        library=Library(name="numrs2", local_path=Path("repositories/numrs")),
        image_name="coder-mcp-numrs2:latest",
    )
    model_loading_settings: ModelLoadingSettings
