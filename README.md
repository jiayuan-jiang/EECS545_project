# FinMini: Financial Sentiment Analysis with LLMs

**EECS 545: Machine Learning — University of Michigan**

> JiaYi Zhang · JiaYuan Jiang · Yue Yu · YiXiang Fei

---

## Overview

This project benchmarks open-source LLMs on financial sentiment classification and explores domain adaptation via a two-stage fine-tuning pipeline (QLoRA-SFT → Chain-of-Thought SFT). We further evaluate LLM-generated sentiment signals in a text-driven long-short equity trading strategy, backtested over 11 quarters.

**Task:** 3-class sentiment classification (positive / negative / neutral) across four financial NLP benchmarks.

**Models evaluated:**

| Model | Params | Type |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7B | Instruction-tuned |
| Meta-Llama-3-8B-Instruct | 8B | Instruction-tuned |
| DeepSeek-R1-Distill-Llama-8B | 8B | Reasoning-distilled |
| FinMini QLoRA-SFT | 8B | Domain fine-tuned (Stage 1) |
| FinMini CoT-SFT | 8B | Chain-of-Thought fine-tuned (Stage 2) |

**Training data (CoT-SFT pipeline):**

| Dataset | HuggingFace Source | Train | Test | Notes |
|---|---|---|---|---|
| DS2 (multi-source) | `sjyuxyz/financial-sentiment-analysis` | ~63K | ~7.9K | FinGPT, Twitter, zeroshot tweets |
| FPB | `FinanceMTEB/financial_phrasebank` | ~0.8K | ~0.2K | 80/20 split; 100% annotator agreement |
| FLARE-SM-ACL | `TheFinAI/flare-sm-acl` | 20.8K | 3.7K | Stock movement prediction (Rise/Fall → pos/neg) |
| **Merged** | — | **84,176** | **11,699** | Neutral labels filtered; binary positive/negative only |

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

### Sentiment Classification (Accuracy)

| Model | FPB | FiQASA | SM-BigData | SM-CIKM | Avg |
|---|---|---|---|---|---|
| Mistral-7B | 0.855 | 0.728 | 0.491 | 0.473 | 0.637 |
| Llama-3-8B | 0.695 | 0.820 | 0.431 | 0.393 | 0.585 |
| DeepSeek-8B (baseline) | 0.835 | 0.576 | 0.550 | 0.493 | 0.613 |
| FinMini QLoRA-SFT | 0.931 | 0.945 | 0.571 | 0.522 | 0.742 |
| **FinMini CoT-SFT** | **0.985** | **0.977** | **0.573** | **0.555** | **0.772** |

**FinMini CoT-SFT** achieves state-of-the-art results across all benchmarks. Compared to the DeepSeek-8B baseline, CoT-SFT delivers +15.0 pp average gain, with breakthrough improvements on FPB (+15.0 pp) and FiQASA (+40.1 pp). Stage 2 (CoT-SFT) further improves over Stage 1 (QLoRA-SFT) by adding explicit `<think>` reasoning traces during training.

Social media benchmarks remain challenging due to informal language and noise, though CoT-SFT narrows the gap slightly.

### Trading Strategy (Backtested over 11 Quarters, 2023Q1–2025Q3)

| Metric | Value |
|---|---|
| Cumulative Return | 21.83% |
| Annualized Return | 7.44% |
| Annualized Volatility | 10.83% |
| Sharpe Ratio | 0.6875 |
| Max Drawdown | -9.47% |
| Long Hit Rate | 81.82% |
| Short Hit Rate | 36.36% |

LLM-scored earnings call transcripts drive quarterly long-short position selection. Long signals demonstrate strong predictive accuracy (81.82%), while short signals remain harder to capture (36.36%), suggesting future work should focus on improving the short-side signal quality.

See [`results/results_summary.md`](results/results_summary.md) for the full analysis.

---

## Fine-Tuning Pipeline

We adopt a two-stage fine-tuning approach for DeepSeek-R1-Distill-Llama-8B:

**Stage 1 — QLoRA-SFT:** Label-only supervised fine-tuning using LoRA adapters (r=16, α=32, target modules: q/k/v/o/gate/up/down projections). Training runs ~12,370 gradient steps, reducing loss from 2.91 → 1.72.

**Stage 2 — CoT-SFT:** Chain-of-Thought supervised fine-tuning. Training data is first constructed by calling the DeepSeek API (`deepseek-chat`) to generate 3–4 sentence reasoning traces for each labeled sample, producing `train_cot_sft.json`. The model is then trained on examples with explicit `<think>reasoning</think>` output format, teaching it to reason step-by-step before producing a sentiment label. This stage yields additional accuracy gains of ~3 pp on formal text benchmarks.

**Inference config:** 4-bit NF4 double-quant (bitsandbytes), `max_new_tokens=32`, greedy decoding.

---

## Repository Structure

```
EECS545_project/
├── notebooks/                        # Jupyter notebooks
│   ├── data_processing.ipynb         # Data loading, merging & CoT annotation
│   ├── FT.ipynb                      # QLoRA fine-tuning (Stage 1)
│   ├── CoT_SFT.ipynb                 # Chain-of-Thought SFT (Stage 2)
│   └── llm_long_short_strategy.ipynb # LLM-driven trading strategy
│
├── scripts/                          # Evaluation & visualization
│   ├── run_baseline.py               # Zero-shot baseline evaluation (all models)
│   ├── run_finetuned.py              # Fine-tuned model evaluation
│   ├── debug_sm_bigdata.py           # Debug inference on SM-BigData
│   ├── generate_plots.py             # Training loss, radar & accuracy plots
│   ├── generate_bar.py               # Bar chart (5 model variants)
│   └── generate_heatmap.py           # Heatmap generation
│
├── results/                          # Experiment outputs
│   ├── baseline_results.json         # Zero-shot accuracy (all models/benchmarks)
│   ├── finetuned_results.json        # QLoRA-SFT model accuracy
│   ├── checkpoint_0050000.json       # Mid-training CoT checkpoint examples
│   ├── training_history.log          # Training loss log
│   ├── results_summary.md            # Full results analysis
│   ├── results_summary.pdf           # PDF version of results summary
│   ├── performance_summary.json      # Trading strategy backtest metrics
│   ├── reasoning_cases.json          # Quarterly LLM scoring & reasoning traces
│   └── backtest_results.csv          # Detailed trade-by-trade backtest log
│
├── figures/                          # Generated plots
│
├── models/
│   └── deepseek-financial-lora-final/ # Saved LoRA adapter weights
│
├── poster/                           # Conference poster (LaTeX)
│   ├── poster_new.tex
│   └── poster_new.bib
│
└── docs/                             # Reports & course materials
    ├── final_report.pdf              # Final project report
    ├── 545_progress_report.pdf       # Progress report
    ├── Project Proposal.docx
    └── 环境搭建并运行llm.md          # Environment setup guide (Chinese)
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

### Data Preprocessing

Run `notebooks/data_processing.ipynb` to reproduce the training data pipeline:
1. Loads DS2, FPB, and FLARE-SM-ACL from HuggingFace
2. Filters neutral labels and unifies label format (binary: positive/negative)
3. Merges all sources → `train_financial_sentiment_merged.json` (84,176 train / 11,699 test)
4. Calls DeepSeek API to generate `<think>` reasoning traces → `train_cot_sft.json`

Requires a DeepSeek API key set in the notebook (`API_KEY` variable).

### Fine-tuning

See `notebooks/FT.ipynb` for the Stage 1 QLoRA fine-tuning pipeline and `notebooks/CoT_SFT.ipynb` for the Stage 2 Chain-of-Thought fine-tuning. The trained LoRA adapter is saved in `models/deepseek-financial-lora-final/`.

### Trading Strategy

See `notebooks/llm_long_short_strategy.ipynb` for the full backtesting pipeline. Backtest results are saved in `results/backtest_results.csv` and `results/performance_summary.json`.
