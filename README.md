# FinMini: Financial Sentiment Analysis with LLMs

**EECS 545: Machine Learning — University of Michigan**

> JiaYi Zhang · JiaYuan Jiang · Yue Yu · YiXiang Fei

---

## Overview

This project benchmarks open-source LLMs on financial sentiment classification and explores domain adaptation via LoRA fine-tuning. We also investigate text-driven long-short trading strategies powered by LLM-generated sentiment signals.

**Task:** 3-class sentiment classification (positive / negative / neutral) across four financial NLP benchmarks.

**Models evaluated:**

| Model | Params | Type |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7B | Instruction-tuned |
| Meta-Llama-3-8B-Instruct | 8B | Instruction-tuned |
| DeepSeek-R1-Distill-Llama-8B | 8B | Reasoning-distilled |
| DeepSeek-R1-Distill-Llama-8B + LoRA | 8B | Domain fine-tuned |

---

## Benchmarks

| Dataset | Domain | Size | Description |
|---|---|---|---|
| **FPB** (Financial PhraseBank) | EN news | 393 | Annotated financial news sentences |
| **FiQASA** | EN Q&A | 217 | Financial Q&A sentiment |
| **SM-BigData** | Social media | 1,472 | Twitter financial sentiment (BigData22) |
| **SM-CIKM** | Social media | 1,143 | Twitter financial sentiment (CIKM18) |

---

## Key Results

| Model | FPB | FiQASA | SM-BigData | SM-CIKM | Avg |
|---|---|---|---|---|---|
| Mistral-7B | 0.855 | 0.728 | 0.491 | 0.473 | 0.637 |
| Llama-3-8B | 0.695 | **0.820** | 0.431 | 0.393 | 0.585 |
| DeepSeek-8B (baseline) | 0.835 | 0.576 | 0.550 | 0.493 | 0.613 |
| **DeepSeek-8B + LoRA** | **0.931** | **0.945** | 0.523 | 0.522 | **0.730** |

LoRA fine-tuning delivers +11.7 pp average gain, with dramatic improvements on FPB (+9.7 pp) and FiQASA (+36.9 pp). Social media benchmarks remain challenging due to informal language and noise.

See [`results/results_summary.md`](results/results_summary.md) for the full analysis.

---

## Repository Structure

```
EECS545/
├── notebooks/                   # Jupyter notebooks
│   ├── FT.ipynb                 # QLoRA fine-tuning (DeepSeek + LoRA)
│   ├── CoT_SFT.ipynb            # Chain-of-Thought supervised fine-tuning
│   └── llm_long_short_strategy.ipynb  # LLM-driven trading strategy
│
├── scripts/                     # Evaluation & visualization
│   ├── run_baseline.py          # Zero-shot baseline evaluation (all models)
│   ├── run_finetuned.py         # Fine-tuned model evaluation
│   ├── debug_sm_bigdata.py      # Debug inference on SM-BigData
│   ├── generate_plots.py        # Training loss & accuracy plots
│   ├── generate_bar.py          # Bar chart generation
│   └── generate_heatmap.py      # Heatmap generation
│
├── results/                     # Experiment outputs
│   ├── baseline_results.json    # Zero-shot accuracy across all models/benchmarks
│   ├── finetuned_results.json   # Fine-tuned model accuracy
│   ├── checkpoint_0050000.json  # Mid-training checkpoint results
│   ├── training_history.log     # Training loss log
│   └── results_summary.md       # Full results analysis
│
├── figures/                     # Generated plots
│
├── models/                      # Saved model adapters
│   └── deepseek-financial-lora-final/  # LoRA adapter weights
│
├── poster/                      # Conference poster (LaTeX)
│   ├── poster_new.tex
│   └── poster_new.bib
│
└── docs/                        # Reports & course materials
    ├── Project Proposal.docx
    ├── FinMini_Progress_Report.docx
    └── 环境搭建并运行llm.md      # Environment setup guide
```

---

## Setup & Usage

### Environment

```bash
conda create -n llm python=3.10 -y
conda activate llm
pip install torch transformers>=4.41 accelerate peft bitsandbytes sentencepiece safetensors
```

See [`docs/环境搭建并运行llm.md`](docs/环境搭建并运行llm.md) for detailed setup instructions (including HuggingFace cache configuration).

### Run Baseline Evaluation

```bash
python scripts/run_baseline.py
```

### Run Fine-tuned Evaluation

```bash
python scripts/run_finetuned.py 2>&1 | tee run_finetuned.log
```

### Generate Figures

```bash
python scripts/generate_plots.py
python scripts/generate_bar.py
python scripts/generate_heatmap.py
```

### Fine-tuning

See `notebooks/FT.ipynb` for the full QLoRA fine-tuning pipeline. The trained LoRA adapter is saved in `models/deepseek-financial-lora-final/`.

**Inference config:** 4-bit NF4 double-quant (bitsandbytes), `max_new_tokens=32`, greedy decoding.
