# Oncology-Domain-Adaptation-of-GPT-OSS-20B-Using-LoRA-Fine-Tuning


This repository contains the implementation of an oncology domain adaptation pipeline for **GPT-OSS-20B** using **Low-Rank Adaptation (LoRA)**.

The project investigates whether parameter-efficient fine-tuning on oncology question–answer data can improve the model's ability to answer general oncology knowledge questions. It also incorporates an embedding-based semantic similarity audit to identify and reduce potential overlap between training and evaluation datasets.

## Project Overview

Large language models may lack specialized domain knowledge, particularly in areas such as oncology. This project explores adapting GPT-OSS-20B to oncology-related question answering through supervised fine-tuning while retaining the original model's general capabilities.

The pipeline consists of four major stages:

1. **Dataset Generation:** Generate oncology question–answer pairs from National Cancer Institute (NCI) information using the Ministral-14B-2512 language model.
2. **Semantic Similarity Audit:** Identify potentially overlapping training and evaluation examples using sentence embeddings and cosine similarity.
3. **LoRA Fine-Tuning:** Adapt GPT-OSS-20B using parameter-efficient supervised fine-tuning.
4. **Evaluation:** Compare the base and fine-tuned models using Precision, Recall, and F1 derived from BERTScore.

## Dataset Construction

- **Training data source:** National Cancer Institute (NCI).
- **Evaluation data source:** World Health Organization (WHO).
- **QA generation model:** Ministral-14B-2512.
- **Original training dataset:** 2,537 question–answer pairs.
- **Evaluation dataset:** 102 question–answer pairs.

The generated data follows a question–answer structure, with questions used as prompts and corresponding answers as supervised targets.

## Semantic Similarity-Based Data Leakage Audit

To identify potential semantic overlap between training and evaluation examples, the project uses:

- Embedding model: `BAAI/bge-base-en-v1.5`
- Embedding dimension: 768
- Similarity measure: Cosine similarity
- Similarity threshold: 0.70

The audit compares concatenated question–answer representations across the two datasets.

Under the selected filtering policy, training examples with a similarity score greater than or equal to 0.70 against any evaluation example are removed.

| Dataset | Number of examples |
|---|---:|
| Original training dataset | 2,537 |
| Training examples removed | 1,675 |
| Final filtered training dataset | 862 |
| Evaluation dataset retained | 102 |

This procedure aims to reduce potential semantic overlap. It does not establish that all forms of data leakage have been eliminated.

## Model Fine-Tuning

The project uses GPT-OSS-20B as the base model and applies LoRA for parameter-efficient supervised fine-tuning.

### LoRA Configuration

| Parameter | Value |
|---|---|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| LoRA dropout | 0 |
| Maximum sequence length | 1,024 |
| Learning rate | 2e-4 |
| Training epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 8 |
| Optimizer | AdamW 8-bit |
| Scheduler | Linear |
| Warmup ratio | 0.03 |
| Random seed | 3407 |

### Training Precision

Two experimental configurations were investigated:

- **NF4:** 4-bit quantized fine-tuning.
- **BF16:** BFloat16 fine-tuning.

The BF16 experiment used the semantically filtered training dataset containing 862 examples.

## Evaluation

The base and fine-tuned models are evaluated against the oncology evaluation dataset.

Evaluation uses BERTScore with:

- Evaluation model: `roberta-large`
- Language: English
- Baseline rescaling: Disabled
- Generation strategy: Greedy decoding
- Maximum generated tokens: 96

Precision, Recall, and F1 are reported for model comparison.

### Experimental Results

**NF4 Configuration — Original 102-question Evaluation Set**

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base | 0.839304 | 0.879390 | 0.858703 |
| LoRA Fine-Tuned | 0.904412 | 0.894874 | 0.899382 |

**BF16 Configuration — Filtered Training Dataset**

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base | 0.854070 | 0.889471 | 0.871043 |
| LoRA Fine-Tuned | 0.916300 | 0.898117 | 0.906837 |

The best observed F1 score was **0.906837**, achieved by the BF16 + LoRA configuration.

### Bootstrap Analysis

For the BF16 experiment, bootstrap analysis was performed using 10,000 resamples to estimate uncertainty in the F1 improvement.

| Statistic | Value |
|---|---:|
| Mean F1 improvement | 0.035794 |
| 95% Confidence Interval | [0.029655, 0.041930] |
| Number of resamples | 10,000 |

The confidence interval lies entirely above zero, indicating a positive estimated improvement over the base model on the evaluation set.

## Reproducibility

The implementation includes scripts for dataset generation, semantic similarity analysis, model fine-tuning, and evaluation.

The experiments use fixed random seeds where specified. Training and inference require substantial GPU memory and compatible CUDA/PyTorch environments, particularly for GPT-OSS-20B.

Model checkpoints and datasets are not necessarily included in this repository. Refer to the individual scripts for input paths, configuration parameters, and expected data formats.

## Limitations

- BERTScore measures semantic similarity between generated and reference answers; it does not independently establish medical factual correctness.
- The evaluation dataset is relatively small, containing 102 questions.
- Semantic similarity filtering reduces potential overlap but cannot guarantee complete absence of data leakage.
- The experiments do not constitute clinical validation or establish suitability for medical decision-making.

## Disclaimer

This project is intended for academic deliverable and experimentation in domain adaptation of large language models. It is **not a medical diagnostic system** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Acknowledgements

This work was conducted as a final-year project in the Department of Computer Science and Engineering at **IIT (ISM) Dhanbad**, under the supervision of **Prof. Saurabh Srivastava**.

Special thanks to the supervisor for his guidance and for providing the resources necessary to conduct this work.
