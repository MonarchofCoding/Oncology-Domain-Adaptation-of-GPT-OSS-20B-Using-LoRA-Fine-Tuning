
# ============================================================
# GPT-OSS-20B ONCOLOGY LoRA SFT — BF16 TRAINING
# Base model: GPT-OSS-20B BF16
# Training: 862 retained NCI examples
# Evaluation: Separate second stage
# ============================================================

import os
import gc
import json
import inspect
import random
import numpy as np
import torch

from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "unsloth/gpt-oss-20b-BF16"

# CHANGE THESE PATHS
TRAIN_FILE = "/path/to/final_862_train.jsonl"

OUTPUT_DIR = "/path/to/gpt_oss_20b_oncology_bf16_lora"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FINAL_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "final_adapter")

MAX_SEQ_LENGTH = 1024

NUM_EPOCHS = 3

LORA_RANK = 8
LORA_ALPHA = 16

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 0.3

SEED = 3407

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# 2. REPRODUCIBILITY AND HARDWARE CHECKS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. A GPU is required.")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "The current GPU does not support native BF16. "
        "Use a BF16-capable GPU such as an A100 or H100."
    )

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("BF16 supported:", torch.cuda.is_bf16_supported())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)

    print(f"\nGPU {i}: {props.name}")
    print(f"Total VRAM: {total / 1024**3:.2f} GiB")
    print(f"Free VRAM: {free / 1024**3:.2f} GiB")

if not os.path.isfile(TRAIN_FILE):
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

# ============================================================
# 3. LOAD AND VALIDATE TRAINING DATA
# ============================================================

training_rows = []

with open(TRAIN_FILE, "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):

        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid training JSON at line {line_number}"
            ) from error

        question = item.get("input")
        answer = item.get("output")

        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(
                f"Invalid input/output fields at line {line_number}"
            )

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            continue

        training_rows.append({
            "prompt": [
                {"role": "user", "content": question}
            ],
            "completion": [
                {"role": "assistant", "content": answer}
            ],
        })

if not training_rows:
    raise ValueError("No valid training examples were found.")

train_dataset = Dataset.from_list(training_rows)

print("\nTraining examples:", len(train_dataset))
print("Training format:", train_dataset.features)
print("Example prompt:", train_dataset[0]["prompt"])
print("Example completion:", train_dataset[0]["completion"])

# Expected professor-selected training split
if len(train_dataset) != 862:
    raise ValueError(
        f"Expected 862 training examples, found {len(train_dataset)}. "
        "Check the training file."
    )

# ============================================================
# 4. LOAD BF16 BASE MODEL
# ============================================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=False,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False

print("\nBF16 base model loaded.")
print("Model:", MODEL_PATH)
print("Requested dtype: BF16")
print("4-bit quantization: Disabled")

# ============================================================
# 5. ATTACH FRESH LoRA ADAPTERS
# ============================================================

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

trainable_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

total_parameters = sum(
    p.numel() for p in model.parameters()
)

if trainable_parameters == 0:
    raise RuntimeError("No trainable LoRA parameters were found.")

print("\nFresh LoRA adapter attached.")
print(f"Trainable parameters: {trainable_parameters:,}")
print(f"Total parameters: {total_parameters:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.5f}%"
)

# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

training_args = SFTConfig(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,

    optim="adamw_8bit",

    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    gradient_checkpointing=True,

    # Train on assistant completion tokens only.
    completion_only_loss=True,

    max_length=MAX_SEQ_LENGTH,
    packing=False,

    # Explicit BF16 mixed-precision training.
    bf16=True,
    fp16=False,

    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

# ============================================================
# 7. FINITE LOSS AND GRADIENT CHECKS
# ============================================================

class FiniteLossSFTTrainer(SFTTrainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        parent_compute_loss = super().compute_loss

        parent_parameters = inspect.signature(
            parent_compute_loss
        ).parameters

        parent_kwargs = {}

        if "return_outputs" in parent_parameters:
            parent_kwargs["return_outputs"] = return_outputs

        if (
            "num_items_in_batch" in parent_parameters
            and num_items_in_batch is not None
        ):
            parent_kwargs["num_items_in_batch"] = num_items_in_batch

        for key, value in kwargs.items():
            if key in parent_parameters:
                parent_kwargs[key] = value

        result = parent_compute_loss(
            model,
            inputs,
            **parent_kwargs,
        )

        loss = result[0] if isinstance(result, tuple) else result

        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError(
                "Non-finite training loss detected. "
                "Training stopped before backpropagation."
            )

        return result


class FiniteGradientCallback(TrainerCallback):

    def on_pre_optimizer_step(self, args, state, control, **kwargs):

        current_model = kwargs.get("model")

        if current_model is None:
            return control

        for name, parameter in current_model.named_parameters():

            if (
                parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError(
                    f"Non-finite gradient detected in {name} "
                    f"at optimizer step {state.global_step}."
                )

        return control


# ============================================================
# 8. CREATE TRAINER
# ============================================================

trainer = FiniteLossSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# ============================================================
# 9. VERIFY TOKENIZATION AND COMPLETION-ONLY LOSS
# ============================================================

processed_dataset = trainer.train_dataset

if len(processed_dataset) == 0:
    raise RuntimeError("Processed training dataset is empty.")

required_columns = {"input_ids", "completion_mask"}

missing_columns = required_columns - set(
    processed_dataset.column_names
)

if missing_columns:
    raise RuntimeError(
        f"Missing tokenized fields: {sorted(missing_columns)}. "
        f"Available fields: {processed_dataset.column_names}"
    )

supervised_example_count = 0

for index in range(len(processed_dataset)):

    example = processed_dataset[index]

    input_ids = example["input_ids"]
    completion_mask = example["completion_mask"]

    if len(input_ids) != len(completion_mask):
        raise RuntimeError(
            f"Token/mask length mismatch at example {index}."
        )

    if len(input_ids) > MAX_SEQ_LENGTH:
        raise RuntimeError(
            f"Example {index} exceeds MAX_SEQ_LENGTH."
        )

    if any(int(value) == 1 for value in completion_mask):
        supervised_example_count += 1

if supervised_example_count != len(processed_dataset):
    raise RuntimeError(
        f"Only {supervised_example_count}/"
        f"{len(processed_dataset)} examples contain supervised tokens."
    )

sample_examples = [
    processed_dataset[i]
    for i in range(min(2, len(processed_dataset)))
]

batch = trainer.data_collator(sample_examples)

if "labels" not in batch:
    raise RuntimeError("Training collator did not produce labels.")

for row_index, example in enumerate(sample_examples):

    completion_mask = torch.tensor(
        example["completion_mask"],
        dtype=torch.bool,
    )

    labels = batch["labels"][row_index][:len(completion_mask)]

    actual_supervision = labels.ne(-100)

    if not torch.equal(actual_supervision.cpu(), completion_mask):
        raise RuntimeError(
            f"Completion-only masking failed for sample {row_index}."
        )

print("\nTrainer created and tokenization checks passed.")
print("Processed dataset columns:", processed_dataset.column_names)
print(
    "Examples with supervised completion tokens:",
    supervised_example_count,
    "/",
    len(processed_dataset),
)
print("Collator verified: prompt masked, completion supervised.")

del sample_examples, batch

trainer.add_callback(FiniteGradientCallback())

# ============================================================
# 10. TRAIN
# ============================================================

print("\nStarting fresh GPT-OSS-20B BF16 oncology SFT.")
print("Model:", MODEL_PATH)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("LoRA rank:", LORA_RANK)
print("LoRA alpha:", LORA_ALPHA)
print("Batch size per device:", 1)
print("Gradient accumulation:", 8)
print("Maximum sequence length:", MAX_SEQ_LENGTH)
print("Gradient clipping:", MAX_GRAD_NORM)
print("Optimizer:", "adamw_8bit")
print("Precision:", "BF16")
print("4-bit quantization:", False)
print("Checkpoint interval:", "100 optimizer steps")

train_result = trainer.train(resume_from_checkpoint=False)

# ============================================================
# 11. SAVE FINAL ADAPTER
# ============================================================

os.makedirs(FINAL_ADAPTER_DIR, exist_ok=True)

model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

# Save training metrics separately.
with open(
    os.path.join(OUTPUT_DIR, "training_metrics.json"),
    "w",
    encoding="utf-8",
) as file:
    json.dump(train_result.metrics, file, indent=2)

print("\nTraining completed successfully.")
print("Final LoRA adapter:", FINAL_ADAPTER_DIR)
print("Training metrics:", train_result.metrics)

# The adapter and tokenizer are now saved.
