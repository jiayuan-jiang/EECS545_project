# Financial Sentiment Analysis — Experiment Results Summary

## Overview

This project evaluates three open-source LLMs on financial sentiment classification, then fine-tunes the best-performing candidate with LoRA adapters to assess domain adaptation gains.

**Task:** Three-class sentiment classification (positive / negative / neutral) across four financial benchmarks.

**Models evaluated:**
| Model | Parameters | Type |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | 7B | Instruction-tuned |
| Meta-Llama-3-8B-Instruct | 8B | Instruction-tuned |
| DeepSeek-R1-Distill-Llama-8B | 8B | Reasoning-distilled |
| DeepSeek-R1-Distill-Llama-8B + LoRA | 8B | Domain fine-tuned |

**Inference config:** 4-bit NF4 double-quant (bitsandbytes), `max_new_tokens=32`, greedy decoding.

---

## Benchmarks

| Benchmark | Domain | Size | Description |
|---|---|---|---|
| **FPB** (Financial PhraseBank) | EN News | 393 | Annotated financial news sentences |
| **FiQASA** | EN Q&A | 217 | Financial question & answer sentiment |
| **SM-BigData** | Social media | 1,472 | Twitter financial sentiment (BigData22) |
| **SM-CIKM** | Social media | 1,143 | Twitter financial sentiment (CIKM18) |

---

## Baseline Results

![Grouped bar chart](figures/fig1_grouped_bar.png)

| Model | FPB | FiQASA | SM-BigData | SM-CIKM | **Avg** |
|---|---|---|---|---|---|
| Mistral-7B | 0.8550 | 0.7281 | 0.4905 | 0.4733 | **0.6367** |
| DeepSeek-8B | 0.8346 | 0.5760 | **0.5496** | **0.4934** | 0.6134 |
| Llama-3-8B | 0.6947 | **0.8203** | 0.4314 | 0.3928 | 0.5848 |

![Radar chart](figures/fig2_radar.png)

**Key observations:**

- **Mistral-7B** achieves the highest overall average (63.7%) and is strongest on FPB, showing solid instruction-following for news-style financial text.
- **Llama-3-8B** excels on FiQASA (82.0%) but collapses on SM benchmarks (~39–43%), suggesting poor robustness to informal/social text.
- **DeepSeek-8B** leads on both SM benchmarks despite no domain-specific training, likely benefiting from its reasoning-distillation pretraining. However it underperforms on FiQASA (57.6%).
- All models struggle on SM benchmarks (<55%), reflecting the informal language, abbreviations, and noise in social media financial text.

---

## Fine-tuning Results

**Setup:** LoRA fine-tuning on DeepSeek-R1-Distill-Llama-8B, adapter saved to `./models/deepseek-financial-lora-final`. Training ran from step ~3,410 to step ~15,780 (≈ 12,370 gradient steps), with loss decreasing from **~2.91 → ~1.72**.

![Training loss](figures/fig5_training_loss.png)

### Accuracy Comparison

| Benchmark | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| FPB | 0.8346 | **0.9313** | **+0.097** |
| FiQASA | 0.5760 | **0.9447** | **+0.369** |
| SM-BigData | 0.5496 | 0.5231 | −0.027 |
| SM-CIKM | 0.4934 | **0.5223** | +0.029 |
| **Average** | 0.6134 | **0.7304** | **+0.117** |

![Delta chart](figures/fig3_delta.png)

### Overall Average

![Average bar](figures/fig4_avg.png)

---

## Analysis

### What worked well

Fine-tuning produced dramatic gains on English-language benchmarks:
- **FiQASA: +36.9 pp** — the largest improvement. The baseline model was clearly ill-calibrated for this Q&A sentiment format; fine-tuning almost closes the gap with Llama-3-8B's strong zero-shot performance.
- **FPB: +9.7 pp** — brings FPB accuracy to 93.1%, the highest of any model tested.
- Fine-tuned DeepSeek (avg 73.0%) **outperforms all baselines**, including Mistral-7B (63.7%), confirming that LoRA domain adaptation provides substantial, generalizable gains.

### What did not improve

- **SM-BigData: −2.7 pp** — a small regression. The model did not degrade badly, but training data likely underrepresents noisy social-media language.
- **SM-CIKM: +2.9 pp** — marginal improvement only. Social media sentiment remains the hardest category across all models.
- Fine-tuned results show `unknown` outputs (responses that could not be parsed as positive/negative/neutral): 25 on FPB, 8 on FiQASA, 48 on SM-BigData, 34 on SM-CIKM. The SM unknown count is notably higher, suggesting the model sometimes generates reasoning traces instead of a clean label when faced with ambiguous social text.

### Root cause of SM performance gap

All models show a systematic accuracy drop of ~30–40 pp when moving from EN news/Q&A to social media benchmarks. This is consistent with known challenges:
1. Informal language, slang, hashtags, and ticker symbols not well represented in instruction-tuning data
2. Shorter, more ambiguous sentences with implicit sentiment
3. Class imbalance in social media datasets

The fine-tuning training data appears dominated by formal financial text (news/earnings), which explains why SM gains are limited.

---

## Conclusion

LoRA fine-tuning of DeepSeek-R1-Distill-Llama-8B achieves strong results on formal financial sentiment tasks (FPB: 93.1%, FiQASA: 94.5%) and sets a new best across all models tested. The remaining performance gap is concentrated in social media benchmarks, suggesting that future work should incorporate Twitter/Reddit financial corpora in training data to achieve more balanced coverage.
