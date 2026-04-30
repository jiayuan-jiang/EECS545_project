import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("./figures")
OUT.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
models = ["Mistral-7B", "Llama-3-8B", "DeepSeek-8B\n(base)", "FinMini\n(Ours)"]
benchmarks = ["FPB", "FiQASA", "SM-BigData", "SM-CIKM"]
bench_sub  = ["(EN News)", "(EN Q&A)", "(Social)", "(Social)"]

values = np.array([
    [85.5, 72.8, 49.1, 47.3],
    [69.5, 82.0, 43.1, 39.3],
    [83.5, 57.6, 55.0, 49.3],
    [93.1, 94.5, 57.1, 52.2],   # FinMini — row index 3
])

# Per-column min-max normalise so colour reflects within-benchmark rank
col_min = values.min(axis=0)
col_max = values.max(axis=0)
norm = (values - col_min) / (col_max - col_min + 1e-9)

# ── UMich palette ─────────────────────────────────────────────────────────────
UMICH_BLUE  = "#00274C"
UMICH_MAIZE = "#FFCB05"

# Custom blue colormap: white → UMich blue
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "umich", ["#e8f1f8", "#4a90c4", UMICH_BLUE], N=256
)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 3.6))
fig.patch.set_facecolor("white")

im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

# Cell annotations
for i in range(len(models)):
    for j in range(len(benchmarks)):
        v   = values[i, j]
        nv  = norm[i, j]
        fg  = "white" if nv > 0.55 else UMICH_BLUE
        bld = "bold"  if i == 3 else "normal"
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                fontsize=12.5, color=fg, fontweight=bld, family="sans-serif")

# Gold border on FinMini row
finmini_row = len(models) - 1
for j in range(len(benchmarks)):
    rect = mpatches.FancyBboxPatch(
        (j - 0.5, finmini_row - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0",
        linewidth=2.5, edgecolor=UMICH_MAIZE, facecolor="none",
        zorder=3
    )
    ax.add_patch(rect)

# Axes
ax.set_xticks(range(len(benchmarks)))
ax.set_xticklabels(
    [f"{b}\n{s}" for b, s in zip(benchmarks, bench_sub)],
    fontsize=10.5, color=UMICH_BLUE
)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=11, color=UMICH_BLUE)
ax.tick_params(length=0)

# Bold FinMini y-label
labels = ax.get_yticklabels()
labels[-1].set_fontweight("bold")

# Gridlines between cells
for x in np.arange(-0.5, len(benchmarks), 1):
    ax.axvline(x, color="white", linewidth=1.5)
for y in np.arange(-0.5, len(models), 1):
    ax.axhline(y, color="white", linewidth=1.5)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Title & legend note
ax.set_title("Accuracy on Open FinLLM Benchmarks",
             fontsize=13, color=UMICH_BLUE, fontweight="bold", pad=10)

note = mpatches.Patch(facecolor=UMICH_MAIZE, edgecolor=UMICH_BLUE,
                       linewidth=0.8, label="  FinMini (Ours)  ")
ax.legend(handles=[note], loc="upper right",
          bbox_to_anchor=(1.0, -0.18), fontsize=9.5,
          frameon=True, framealpha=0.9, edgecolor="#cccccc",
          ncol=1)

fig.tight_layout(pad=0.8)
fig.savefig(OUT / "fig6_heatmap.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("Saved figures/fig6_heatmap.png")
