# Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity

[![arXiv](https://img.shields.io/badge/arXiv-2507.23121-black)](<https://arxiv.org/abs/2507.23121>)
[![Project](https://img.shields.io/badge/GitHub-LLM--Chinese--Textual--Disambiguation-blue)](<https://github.com/ictup/LLM-Chinese-Textual-Disambiguation>)

> **TL;DR** — This repository accompanies our study on how **Chinese textual ambiguity** undermines the **trustworthiness** of large language models (LLMs). We formalize tasks for **ambiguity detection**, **ambiguity understanding** (locating the source, enumerating all plausible readings, and rewriting disambiguated variants), and **end-to-end handling**. We provide strong baselines and reference pipelines (prompting strategies and optional RAG) so others can replicate and extend our findings.

---

## Table of Contents
- [Motivation & Contributions](#motivation--contributions)
- [Tasks & Evaluation](#tasks--evaluation)
- [Methodology & System Design](#methodology--system-design)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training & Evaluation](#training--evaluation)
- [Optional: RAG-Enhanced Pipeline](#optional-rag-enhanced-pipeline)
- [Repository Structure](#repository-structure)
- [Applications](#applications)
- [Limitations & Ethics](#limitations--ethics)
- [Cite](#cite)
- [Authors & Contact](#authors--contact)
- [License](#license)

---

## Motivation & Contributions

LLMs are increasingly deployed in **assistant**, **agent**, and **retrieval-augmented** workflows. However, **real-world Chinese** contains abundant forms of **ambiguity** (lexical, syntactic, and semantic–pragmatic). If a model commits to a single interpretation too early, it can execute the **wrong action**, provide **misleading answers**, or fail to ask for **clarification**.

**This project contributes:**

1. **A principled problem formulation for Chinese ambiguity under trustworthiness.** We center on whether models can **identify ambiguous inputs**, **enumerate competing readings**, and **rewrite explicit disambiguations** before acting.
2. **Task protocols and metrics** that evaluate detection and understanding under practical settings.
3. **Reference implementations** covering (i) a lightweight classifier baseline for **ambiguity detection**, (ii) multiple **prompting strategies** for LLMs, and (iii) an **optional RAG+Few-shot** pipeline to stabilize outputs.
4. **Reproducible scripts** for training, evaluation, and analysis, designed for academic transparency and fair comparison.

---

## Tasks & Evaluation

We evaluate three complementary capabilities:

1. **Ambiguity Detection** — Decide whether an input sentence is **ambiguous**.  
   - *Metrics*: Accuracy, Precision, Recall, Macro-F1.
2. **Ambiguity Understanding** — (a) **Locate** the ambiguity source, (b) **enumerate** all plausible interpretations, and (c) **rewrite** their corresponding **disambiguated sentences**.  
   - *Metrics*: Set-F1, Recall, Exact Match (EM) at the interpretation set level.
3. **End-to-End Handling** — A pipeline that first detects ambiguity, then triggers understanding and controlled rewriting before acting or answering downstream.  
   - *Metrics*: Task-specific combinations of the above, plus latency/cost proxy if applicable.

> **Why these metrics?**  
> In realistic systems, it’s not enough to “be correct once.” We want models to explicitly **surface uncertainties** (detection), **cover all valid readings** (set-level evaluation), and **produce actionable disambiguations** the user or a downstream module can approve.

---

## Methodology & System Design

### 1) Detection Baseline (Reference)
A compact Chinese encoder (e.g., `hfl/chinese-roberta-wwm-ext`) is fine-tuned as a **binary classifier** to flag potential ambiguity. Feature ablations (e.g., sequence length, POS patterns, or shallow parsing depth) can be toggled via CLI flags to examine robustness.

### 2) Prompting Strategies for LLMs
We provide multiple prompting paradigms to stress-test understanding:
- **Direct**: ask once, no examples.
- **Few-shot**: prepend a small number of labeled examples.
- **Knowledge-augmented**: inject short notes about Chinese ambiguity phenomena.
- **Chain-of-Thought (CoT)**: encourage stepwise reasoning.
- **CoT + Few-shot**: combine them.
- **RAG + Few-shot (recommended)**: retrieve semantically similar ambiguous cases as exemplars.

### 3) Optional Retrieval-Augmented Generation (RAG)
We maintain a FAISS index built from ambiguous–disambiguated pairs. At inference time, we retrieve top-*K* similar cases to **ground** the model’s reasoning and reduce over- or under-interpretation.

> **Note:** The repository’s scripts are intentionally **modular** so you can swap models, prompts, and retrieval backends without changing the data format.

---

## Quick Start

### Environment
```bash
conda create -n ambiguity python=3.10 -y
conda activate ambiguity
pip install -r requirements.txt
# Recommended deps: transformers, datasets, accelerate, peft,
# faiss-cpu or faiss-gpu, jieba, scikit-learn, optuna,
# and one of: vllm or llama-cpp-python for local inference.
```

> If you use GPUs, set `CUDA_VISIBLE_DEVICES` and consider `accelerate` for model parallelism / multi-GPU.

### Configure Models
- **Local**: Hugging Face models via `transformers` or `vllm` (e.g., Qwen, Gemma, etc.).  
- **Hosted**: If you rely on an API, wrap the call in `scripts/providers/*.py` and set your keys as env vars.

---

## Data Preparation

We assume the following **unified JSONL schema**. If your raw data differs, use `scripts/prepare_data.py` to convert it.

```json
{
  "id": "sample_000123",
  "text_ambiguous": "我们需要组织人员。",
  "meanings": [
    "需要去组织（招募/安排）人员",
    "需要负责组织工作的人员"
  ],
  "disambiguations": [
    "我们需要去招募并安排人手。",
    "我们需要负责组织工作的那批人。"
  ],
  "category": "Syntactic.Structural",
  "notes": "Toy example for illustration"
}
```

**Convert & Split**
```bash
python scripts/prepare_data.py   --input data/raw   --output data/processed   --split 0.7 0.15 0.15   --with_labels
```

This produces `train/dev/test` JSONL files under `data/processed/` matching the schema above.

---

## Training & Evaluation

### 1) Train the Ambiguity Detector (Reference Baseline)
```bash
python scripts/train_detector.py   --model hfl/chinese-roberta-wwm-ext   --train data/processed/train.jsonl   --dev   data/processed/dev.jsonl   --test  data/processed/test.jsonl   --batch_size 16 --lr 2e-5 --epochs 5   --feature pos,seglen,treedepth   --early_stop f1   --cv 5 --optuna
```

### 2) LLM Evaluation — Detection
```bash
python scripts/eval_llm_detection.py   --model qwen2.5-32b-instruct   --prompt_strategy rag_fewshot   --data data/processed/test.jsonl
```

### 3) LLM Evaluation — Understanding
```bash
python scripts/eval_llm_understanding.py   --model deepseek-r1   --prompt_strategy rag_fewshot   --metrics set_f1,recall,em   --data data/processed/test.jsonl
```

> You can change `--model` to any local or hosted provider supported in `scripts/providers/`.  
> Logs, prompts, and outputs are stored under `results/` for inspection and reproducibility.

---

## Optional: RAG-Enhanced Pipeline

### Build a Vector Index
```bash
python scripts/build_index.py   --input data/processed/train.jsonl   --index faiss.index
```

### Inference with Retrieval-Augmented Few-Shot
```bash
python scripts/run_rag_fs.py   --model qwen2.5-14b-instruct   --index faiss.index --k 4   --data data/processed/test.jsonl
```

> Tuning `k` and distance metrics materially affects performance; start with `k ∈ {3,4,5}` and measure set-level coverage.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                 # Original sources
│   └── processed/           # JSONL in unified schema (train/dev/test)
├── scripts/
│   ├── prepare_data.py
│   ├── train_detector.py
│   ├── eval_llm_detection.py
│   ├── eval_llm_understanding.py
│   ├── build_index.py
│   └── run_rag_fs.py
├── configs/                 # Configs for training/eval
├── results/                 # Metrics, logs, experiment cards
└── README.md
```

---

## Applications

- **Conversational assistants & agents** — Ask **clarifying questions** before acting; surface **multiple readings** with explicit disambiguations for user approval.
- **E-commerce & customer support** — Triage ambiguous intents and route to templates that elicit clarifying constraints.
- **Search & QA** — Detect ambiguity on the query side; retrieve **closest ambiguous exemplars** and disambiguation templates as scaffolding.
- **Evaluation & alignment** — Use this benchmark as a **stress test** for uncertainty modeling, calibration, and selective prediction.

---

## Limitations & Ethics

- **Subjectivity** — Ambiguity is context-sensitive; our annotations aim to align with **human judgments** rather than define unique ground truth.
- **Scope** — Focus is on **Chinese** and selected model families; cross-lingual and out-of-domain generalization requires care.
- **Automatic evaluation** — Set-level comparison via model-assisted matching may deviate from human preferences; include **human validation** for critical studies.
- **Safety** — Do not deploy models that act on **ambiguous** instructions without user confirmation. Respect data licenses and privacy constraints.

---

## Cite

If you find this work useful, please cite:

```bibtex
@misc{wu2025uncoveringfragilitytrustworthyllms,
  title        = {Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity},
  author       = {Xinwei Wu and Haojie Li and Hongyu Liu and Xinyu Ji and Ruohan Li and Yule Chen and Yigeng Zhang},
  year         = {2025},
  eprint       = {2507.23121},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2507.23121}
}
```

**Paper**: <https://arxiv.org/abs/2507.23121>  
**Project**: <https://github.com/ictup/LLM-Chinese-Textual-Disambiguation>

---

## Authors & Contact

**Xinwei Wu***, **Haojie Li***, **Hongyu Liu***, Xinyu Ji, Ruohan Li, Yule Chen, **Yigeng Zhang†**  
(*co-first authors; †corresponding author*).  
For questions, please open an issue or contact the corresponding author listed in the paper.

---

## License

We recommend permissive licensing such as **Apache-2.0** or **MIT**. Please ensure that datasets and third-party models adhere to their original terms. If you adopt this repository structure, add a `LICENSE` file accordingly.
