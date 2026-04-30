import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT = Path("./figures")
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open("baseline_results.json") as f:
    base = json.load(f)
with open("finetuned_results.json") as f:
    ft = json.load(f)

BENCH_LABELS = {
    "benchmark_en_fpb":    "FPB\n(EN News)",
    "benchmark_fiqasa":    "FiQASA\n(EN Q&A)",
    "benchmark_sm_bigdata":"SM-BigData\n(Social)",
    "benchmark_sm_cikm":   "SM-CIKM\n(Social)",
}
BENCHMARKS = list(BENCH_LABELS.keys())

MODEL_SHORT = {
    "mistralai/Mistral-7B-Instruct-v0.3":          "Mistral-7B",
    "meta-llama/Meta-Llama-3-8B-Instruct":          "Llama-3-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":     "DeepSeek-8B\n(baseline)",
}
COLORS = {
    "Mistral-7B":            "#4C72B0",
    "Llama-3-8B":            "#DD8452",
    "DeepSeek-8B\n(baseline)": "#55A868",
    "DeepSeek-8B\n(fine-tuned)": "#C44E52",
}

# build lookup: model_short -> benchmark -> accuracy
data = {}
for m in base["models"]:
    short = MODEL_SHORT[m["model"]]
    data[short] = {r["benchmark"]: r["accuracy"] for r in m["results"]}

ft_short = "DeepSeek-8B\n(fine-tuned)"
data[ft_short] = {r["benchmark"]: r["accuracy"] for r in ft["results"]}

models_order = list(MODEL_SHORT.values()) + [ft_short]

# ── Fig 1: Grouped bar – accuracy per benchmark ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
n_models = len(models_order)
n_bench  = len(BENCHMARKS)
x = np.arange(n_bench)
width = 0.18
offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

for i, model in enumerate(models_order):
    accs = [data[model].get(b, 0) for b in BENCHMARKS]
    bars = ax.bar(x + offsets[i], accs, width, label=model.replace("\n", " "),
                  color=COLORS[model], edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels([BENCH_LABELS[b] for b in BENCHMARKS], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Financial Sentiment Classification – Accuracy by Benchmark", fontsize=13, pad=12)
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig1_grouped_bar.png", dpi=150)
plt.close()
print("Saved fig1_grouped_bar.png")

# ── Fig 2: Radar / spider chart ────────────────────────────────────────────────
labels = [BENCH_LABELS[b].replace("\n", " ") for b in BENCHMARKS]
angles = np.linspace(0, 2*np.pi, len(BENCHMARKS), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for model in models_order:
    vals = [data[model].get(b, 0) for b in BENCHMARKS]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, label=model.replace("\n", " "), color=COLORS[model])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[model])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Performance Radar", fontsize=13, pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig2_radar.png", dpi=150)
plt.close()
print("Saved fig2_radar.png")

# ── Fig 3: Fine-tuning delta (DeepSeek baseline → fine-tuned) ─────────────────
ds_base = data["DeepSeek-8B\n(baseline)"]
ds_ft   = data[ft_short]
deltas  = [ds_ft.get(b, 0) - ds_base.get(b, 0) for b in BENCHMARKS]
bar_colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([BENCH_LABELS[b].replace("\n", " ") for b in BENCHMARKS],
              deltas, color=bar_colors, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, deltas):
    ypos = v + 0.003 if v >= 0 else v - 0.016
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{v:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Accuracy Change (Fine-tuned − Baseline)", fontsize=10)
ax.set_title("LoRA Fine-tuning Effect on DeepSeek-R1-Distill-Llama-8B", fontsize=12, pad=10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig3_delta.png", dpi=150)
plt.close()
print("Saved fig3_delta.png")

# ── Fig 4: Overall average bar ────────────────────────────────────────────────
avgs = {m: np.mean([data[m].get(b, 0) for b in BENCHMARKS]) for m in models_order}
fig, ax = plt.subplots(figsize=(8, 4.5))
model_labels = [m.replace("\n", " ") for m in models_order]
colors_list  = [COLORS[m] for m in models_order]
bars = ax.bar(model_labels, list(avgs.values()), color=colors_list, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, avgs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Average Accuracy (4 benchmarks)", fontsize=10)
ax.set_title("Overall Average Accuracy Comparison", fontsize=12, pad=10)
ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig4_avg.png", dpi=150)
plt.close()
print("Saved fig4_avg.png")

# ── Fig 5: Training loss curve ────────────────────────────────────────────────
steps, losses = [], []
pat = re.compile(r"step=\s*(\d+)\s+loss=([\d.]+)")
with open("training_history.log") as f:
    for line in f:
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

# smooth with rolling window
window = 20
smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
smooth_steps = steps[window-1:]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, losses, color="#aac4e0", linewidth=0.8, alpha=0.6, label="raw loss")
ax.plot(smooth_steps, smooth, color="#2166ac", linewidth=2, label=f"smoothed (w={window})")
ax.set_xlabel("Training Step", fontsize=10)
ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
ax.set_title("LoRA Fine-tuning Training Loss Curve", fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig5_training_loss.png", dpi=150)
plt.close()
print("Saved fig5_training_loss.png")

print("\nAll figures saved to ./figures/")
