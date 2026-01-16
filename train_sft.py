from hindsight_generator import HindsightOutput
import datetime
from pathlib import Path
from typing import Self

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from adapter_agent.qra import QRADataset, QRA


class AdapterSFTConfig(BaseModel):
    dataset_path: Path
    base_model_name: str
    output_dir: Path
    model_save_dir: Path
    max_seq_length: int
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float

    @classmethod
    def default(cls) -> Self:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        return cls(
            dataset_path=Path("data/qra/experiment_generic_aligned"),
            base_model_name="Qwen/Qwen3-4B",
            output_dir=Path(f"./outputs/{today}"),
            model_save_dir=Path(f"./checkpoints/qwen3-4b-numrs-qra-{today}"),
            max_seq_length=4096 * 5,
            learning_rate=4e-4,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            weight_decay=0.01,
        )


def load_qra_dataset(project_dir: Path) -> QRADataset:
    hss = [HindsightOutput.load(path) for path in project_dir.iterdir()]
    return QRADataset(
        problems=[
            QRA(question=hs.question, reasoning=hs.reasoning, answer=hs.answer)
            for hs in hss
        ]
    )


config = AdapterSFTConfig.default()

# Load reflection results
# Wait for file to exist if running concurrently (or just assume it will exist)
if not config.dataset_path.exists():
    raise FileNotFoundError(
        f"{config.dataset_path} not found. Please run create_problems.py first."
    )

qra_dataset = load_qra_dataset(config.dataset_path)
print(f"Loaded {len(qra_dataset.problems)} training reflection results")

eval_path = Path("data/reflection_eval.json")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name,
    trust_remote_code=True,
)

# Load model with quantization config for efficiency
use_quantization = False  # Set to True to use 4-bit quantization

if use_quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()  # type: ignore

# Convert reflections to dataset
dataset = qra_dataset.as_conversational()
print(f"Prepared dataset with {len(dataset)} examples")

# Determine if bf16 is supported
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = SFTConfig(
    output_dir=str(config.output_dir),
    completion_only_loss=True,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    warmup_steps=config.warmup_steps,
    num_train_epochs=config.num_train_epochs,
    learning_rate=config.learning_rate,
    logging_steps=1,
    optim="adamw_torch_fused",  # Fused optimizer for better performance
    weight_decay=config.weight_decay,
    max_length=config.max_seq_length,
    lr_scheduler_type="cosine",  # Cosine scheduler often works better than linear
    seed=3407,
    save_strategy="epoch",
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Multi-GPU configuration for 2 devices
    ddp_find_unused_parameters=False,
    dataloader_num_workers=8,  # Increased for better data loading throughput
    dataset_num_proc=4,  # Increased preprocessing parallelism
    remove_unused_columns=False,
    packing=False,
    # Additional optimizations
    dataloader_pin_memory=True,  # Pin memory for faster GPU transfers
    dataloader_prefetch_factor=2,  # Prefetch batches
)

# Create trainer
trainer = SFTTrainer(
    model=model,  # type: ignore
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Start training
print("\nStarting training...")
trainer.train()

# Save the fine-tuned model
config.model_save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(config.model_save_dir))
tokenizer.save_pretrained(str(config.model_save_dir))
print(f"\nModel saved to {config.model_save_dir}")
