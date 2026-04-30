import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("./figures")
OUT.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
BENCHMARKS  = ["FPB\n(EN News)", "FiQASA\n(EN Q&A)", "SM-BigData\n(Social)", "SM-CIKM\n(Social)"]
MODEL_NAMES = ["Mistral-7B", "Llama-3-8B", "DeepSeek-8B\n(base)", "FinMini\nQLoRA-SFT", "FinMini\nCoT-SFT"]

values = np.array([
    [85.5, 72.8, 49.1, 47.3],     # Mistral-7B
    [69.5, 82.0, 43.1, 39.3],     # Llama-3-8B
    [83.5, 57.6, 55.0, 49.3],     # DeepSeek-8B base
    [93.1, 94.5, 57.1, 52.2],     # FinMini QLoRA-SFT
    [98.47, 97.70, 57.33, 55.45], # FinMini CoT-SFT
])

# ── Palette ───────────────────────────────────────────────────────────────────
UMICH_BLUE  = "#00274C"
UMICH_MAIZE = "#FFCB05"

# Baselines: muted blue-grey family; FinMini SFT: maize; CoT-SFT: pink
BAR_COLORS  = ["#8AAFC9", "#5580A0", "#2D5F82", UMICH_MAIZE, "#F4A7B9"]
EDGE_COLORS = ["#6090B0", "#3A6482", "#1A4560", UMICH_BLUE,  "#C2607A"]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.2))
fig.patch.set_facecolor("white")

n_bench  = len(BENCHMARKS)
n_models = len(MODEL_NAMES)
x        = np.arange(n_bench)
width    = 0.15
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

for i, (model, color, edge) in enumerate(zip(MODEL_NAMES, BAR_COLORS, EDGE_COLORS)):
    accs = values[i]
    bars = ax.bar(
        x + offsets[i], accs, width,
        color=color, edgecolor=edge, linewidth=0.8,
        label=model.replace("\n", " "),
        zorder=3,
    )
    # Value labels
    for bar, v in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=7.2,
            color=UMICH_BLUE if i < 3 else ("#7a5c00" if i == 3 else "#8B2252"),
            fontweight="bold" if i >= 3 else "normal",
        )

# Star annotation over CoT-SFT bars (best results)
for j in range(n_bench):
    bx = x[j] + offsets[-1]
    by = values[-1][j]
    ax.annotate(
        "★",
        xy=(bx, by + 4.5),
        ha="center", va="bottom",
        fontsize=9, color="#8B2252",
    )

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(BENCHMARKS, fontsize=11, color=UMICH_BLUE)
ax.set_ylabel("Accuracy (%)", fontsize=11, color=UMICH_BLUE)
ax.set_ylim(0, 105)
ax.set_yticks(range(0, 101, 20))
ax.tick_params(colors=UMICH_BLUE, length=0)
ax.yaxis.set_tick_params(labelsize=9)

# Grid behind bars
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="#cccccc", zorder=0)
ax.set_axisbelow(True)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

# Separator lines between benchmark groups
for xi in x[:-1] + 0.5:
    ax.axvline(xi, color="#dddddd", linewidth=1.0, zorder=1)

# Title
ax.set_title(
    "Accuracy on Open FinLLM Benchmarks",
    fontsize=13, fontweight="bold", color=UMICH_BLUE, pad=12,
)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=BAR_COLORS[i], edgecolor=EDGE_COLORS[i],
                   linewidth=0.8, label=MODEL_NAMES[i].replace("\n", " "))
    for i in range(n_models)
]
ax.legend(
    handles=legend_patches,
    fontsize=9.5, loc="upper right",
    bbox_to_anchor=(1.0, 0.92),
    frameon=True, framealpha=0.9,
    edgecolor="#cccccc",
    ncol=3,
)

fig.tight_layout(pad=1.0)
fig.savefig(OUT / "fig7_bar.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved figures/fig7_bar.png")
