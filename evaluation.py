
# ============================================================
# GPT-OSS-20B ONCOLOGY — BF16 LoRA EVALUATION
# Load saved adapter -> Generate predictions -> BERTScore
# ============================================================

import os
import gc
import json
import torch

from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from bert_score import score


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "unsloth/gpt-oss-20b-BF16"

# CHANGE THESE PATHS
FINAL_ADAPTER_DIR = "/path/to/gpt_oss_20b_oncology_bf16_lora/final_adapter"

TEST_FILE = "/path/to/final_102_test.jsonl"

OUTPUT_DIR = "/path/to/gpt_oss_20b_oncology_bf16_evaluation"

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "bf16_lora_predictions.jsonl",
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "bf16_lora_bertscore_metrics.json",
)

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 96
REASONING_EFFORT = "low"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. HARDWARE AND PATH CHECKS
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "The current GPU does not support native BF16."
    )

if not os.path.isdir(FINAL_ADAPTER_DIR):
    raise FileNotFoundError(
        f"Saved adapter not found: {FINAL_ADAPTER_DIR}"
    )

if not os.path.isfile(TEST_FILE):
    raise FileNotFoundError(
        f"Test file not found: {TEST_FILE}"
    )

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

# ============================================================
# 3. LOAD HELD-OUT TEST DATA
# ============================================================

questions = []
references = []

with open(TEST_FILE, "r", encoding="utf-8") as file:

    for line_number, line in enumerate(file, start=1):

        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid test JSON at line {line_number}"
            ) from error

        question = item.get("input")
        reference = item.get("output")

        if not isinstance(question, str) or not isinstance(reference, str):
            raise ValueError(
                f"Invalid input/output fields at test line {line_number}"
            )

        question = question.strip()
        reference = reference.strip()

        if question and reference:
            questions.append(question)
            references.append(reference)

if not questions:
    raise ValueError("No valid test examples found.")

if len(questions) != len(references):
    raise RuntimeError("Question/reference count mismatch.")

# Expected professor-selected test split
if len(questions) != 102:
    raise ValueError(
        f"Expected 102 test examples, found {len(questions)}. "
        "Check the test file."
    )

print("\nHeld-out test examples:", len(questions))
print("First question:", questions[0])
print("First reference:", references[0])

# ============================================================
# 4. LOAD THE SAVED BF16 LoRA ADAPTER
# ============================================================

# Loading from the adapter directory restores the trained
# LoRA weights on top of their original base model.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=FINAL_ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=False,
)

FastLanguageModel.for_inference(model)

model.eval()

print("\nFine-tuned BF16 model loaded.")
print("Adapter:", FINAL_ADAPTER_DIR)

# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

def generate_answer(question):

    messages = [
        {
            "role": "user",
            "content": (
                "Answer the following general cancer-knowledge question "
                "clearly and concisely. Give a direct answer without "
                "unnecessary reasoning.\n\n"
                f"Question: {question}"
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=REASONING_EFFORT,
    )

    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, prompt_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    del inputs, output_ids, generated_ids

    return answer


# ============================================================
# 6. GENERATE AND SAVE ALL TEST PREDICTIONS
# ============================================================

predictions = []

start_time = __import__("time").time()

with open(PREDICTIONS_FILE, "w", encoding="utf-8") as file:

    for index, (question, reference) in enumerate(
        zip(questions, references)
    ):

        try:
            prediction = generate_answer(question)

        except Exception as error:
            raise RuntimeError(
                f"Generation failed at question {index + 1}: {error}"
            ) from error

        predictions.append(prediction)

        row = {
            "input": question,
            "reference": reference,
            "prediction": prediction,
        }

        # Save immediately so progress is not lost.
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()

        if (
            (index + 1) % 10 == 0
            or index + 1 == len(questions)
        ):
            elapsed = __import__("time").time() - start_time

            print(
                f"Question {index + 1}/{len(questions)} | "
                f"Elapsed: {elapsed / 60:.2f} min"
            )

print("\nPrediction generation completed.")
print("Questions processed:", len(predictions))
print("Total time:", round(
    (__import__("time").time() - start_time) / 60, 2
), "minutes")
print("Predictions saved:", PREDICTIONS_FILE)

# ============================================================
# 7. RELEASE GPT-OSS GPU MEMORY
# ============================================================

del model
del tokenizer

gc.collect()
torch.cuda.empty_cache()

print("\nGPT-OSS model released from GPU memory.")

# ============================================================
# 8. CALCULATE BERTSCORE
# ============================================================

print("\nCalculating BERTScore using roberta-large on CPU.")

P, R, F1 = score(
    cands=predictions,
    refs=references,
    model_type="roberta-large",
    lang="en",
    batch_size=8,
    device="cpu",
    rescale_with_baseline=False,
    verbose=True,
)

mean_precision = P.mean().item()
mean_recall = R.mean().item()
mean_f1 = F1.mean().item()

# ============================================================
# 9. SAVE METRICS
# ============================================================

results = {
    "model": "GPT-OSS-20B",
    "base_checkpoint": MODEL_PATH,
    "fine_tuned": True,
    "fine_tuning_method": "LoRA",
    "base_precision": "BF16",
    "quantization": False,
    "metric": "BERTScore",
    "encoder": "roberta-large",
    "rescale_with_baseline": False,
    "num_test_examples": len(questions),
    "max_new_tokens": MAX_NEW_TOKENS,
    "reasoning_effort": REASONING_EFFORT,
    "precision": mean_precision,
    "recall": mean_recall,
    "f1": mean_f1,
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

# ============================================================
# 10. DISPLAY FINAL RESULTS
# ============================================================

print("\n========== BF16 LoRA RESULTS ==========")
print("Number of test questions:", len(questions))
print(f"Mean BERTScore Precision: {mean_precision:.6f}")
print(f"Mean BERTScore Recall:    {mean_recall:.6f}")
print(f"Mean BERTScore F1:        {mean_f1:.6f}")
print("=======================================")

print("\nSaved metrics:", METRICS_FILE)
print("Saved predictions:", PREDICTIONS_FILE)
print("\nEvaluation completed successfully.")
