from datetime import datetime
from pathlib import Path

import tinker
from pydantic import BaseModel, Field

from adapter_agent.hierarchical.gh import Library
from adapter_agent.rl.advantage import AdvantageRegularizer


class RLConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    base_url: str | None = None
    lora_rank: int = 32
    log_path: str = "logs/hoge_rl"
    wandb_project: str | None = "Hoge RL"
    experiment_name: str = Field(
        default_factory=lambda: f"Hoge_RL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    adam_params: tinker.AdamParams = tinker.AdamParams(
        learning_rate=1e-8, beta1=0.9, beta2=0.95, eps=1e-8
    )
    loss_fn: str = "dro"
    advantage_regularizer: AdvantageRegularizer = "output_token"
    num_groups_per_batch: int = 2  # TODO: Increase this
    num_rollout_workers: int = 1  # TODO: Increase this
    num_steps: int = 1
    rollouts_per_question: int = 4
    per_group_concurrency: int = 1
    max_turns: int = 10
    image_name: str = "coder-mcp-numrs2:latest"
    library: Library = Library(name="numrs2", local_path=Path("repositories/numrs"))
    r_min: float = 0.5
    csv_output_path: str = "logs/rl_results.csv"
    correlation_csv_path: str = "logs/rl_correlation.csv"
