from datetime import datetime
from pathlib import Path
from typing import Self

import tinker
from pydantic import BaseModel
from tinker.types.loss_fn_type import LossFnType

from adapter_agent.data import QASFTDataset, TinkerMessagesDataset
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
    num_rollout_workers: int
    rollouts_per_question: int
    per_group_concurrency: int
    temperature: float = 0.7


class EnvParams(BaseModel):
    max_turns: int = 5
    r_min: float = 0.5
    library: Library
    image_name: str
    dataset_path: Path
    single_turn: bool

    @classmethod
    def numrs2(
        cls, max_turns: int, r_min: float, dataset_path: Path, single_turn: bool
    ) -> Self:
        return cls(
            max_turns=max_turns,
            r_min=r_min,
            library=Library(name="numrs2", local_path=Path("repositories/numrs")),
            image_name="coder-mcp-numrs2:latest",
            dataset_path=dataset_path,
            single_turn=single_turn,
        )


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
    optimizer_params: OptimizerParams
    rollout_params: RolloutParams
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings


class SFTConfig(BaseModel):
    experiment_setting: ExperimentSettings
    optimizer_params: SFTOptimizerParams
    rollout_params: RolloutParams
    env_params: EnvParams
    model_loading_settings: ModelLoadingSettings


class QADataConfig(BaseModel):
    data_path: Path
    train_ratio: float
    test_ratio: float

    def train_test_split(self, seed: int) -> tuple[QASFTDataset, QASFTDataset]:
        ds = QASFTDataset.load(self.data_path)
        train, test = ds.train_test_split(self.train_ratio, seed=seed)
        return train, test


class TrajectorySFTDataConfig(BaseModel):
    data_path: Path
    train_ratio: float
    test_ratio: float

    def train_test_split(
        self, seed: int = 42
    ) -> tuple[TinkerMessagesDataset, TinkerMessagesDataset]:
        ds = TinkerMessagesDataset.load(self.data_path)
        uniqued = ds.id_unique()
        train, test = uniqued.train_test_split(self.train_ratio, seed=seed)
        return train, test
