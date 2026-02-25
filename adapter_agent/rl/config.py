from datetime import datetime
from pathlib import Path
from typing import Self

import tinker
from pydantic import BaseModel
from tinker.types.loss_fn_type import LossFnType

from adapter_agent.hierarchical.gh import Library
from adapter_agent.rl.advantage import AdvantageRegularizer


class OptimizerParams(BaseModel):
    adam_params: tinker.AdamParams
    loss_fn: LossFnType
    advantage_regularizer: AdvantageRegularizer = "output_token"
    num_steps: int
    kl_penalty_coef: float
    kl_discount_factor: float
    num_groups_per_batch: int = 2


class SFTOptimizerParams(BaseModel):
    adam_params: tinker.AdamParams
    batch_size: int
    num_epochs: int


class RolloutParams(BaseModel):
    num_rollout_workers: int = 1
    rollouts_per_question: int = 4
    per_group_concurrency: int = 4
    temperature: float = 0.7


class EnvParams(BaseModel):
    max_turns: int = 5
    r_min: float = 0.5
    library: Library
    image_name: str
    dataset_path: Path


class ExperimentSettings(BaseModel):
    experiment_name: str
    wandb_project: str | None
    ttl_seconds: int = 604800  # 7 days

    def log_root(self) -> Path:
        return Path("logs") / "Adapter_Agent" / self.experiment_name

    @classmethod
    def with_prefix(cls, prefix: str) -> Self:
        return cls(
            experiment_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_project=prefix,
        )


class ModelLoadingSettings(BaseModel):
    model_name: str
    resume_trainer_path: str | None = None
    resume_sampler_path: str | None = None
    lora_rank: int = 32


class RLConfig(BaseModel):
    experiment_setting: ExperimentSettings
    optimizer_params: OptimizerParams = OptimizerParams(
        adam_params=tinker.AdamParams(
            learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8
        ),
        loss_fn="ppo",
        num_steps=1,
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
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
        dataset_path="generated_qas.json",
    )
    model_loading_settings: ModelLoadingSettings


class SFTConfig(BaseModel):
    experiment_setting: ExperimentSettings
    optimizer_params: SFTOptimizerParams
    rollout_params: RolloutParams
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings
