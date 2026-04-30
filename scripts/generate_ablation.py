"""Generate ablation stacked-bar figure (fig9_ablation.png)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

benchmarks = ["FPB\n(EN News)", "FiQASA\n(EN Q&A)", "SM-BigData\n(Social)", "SM-CIKM\n(Social)"]

baseline   = [83.5, 57.6, 55.0, 49.3]
qlora_gain = [ 9.6, 36.9,  2.1,  2.9]
cot_gain   = [ 5.4,  3.2,  0.2,  3.3]

x = np.arange(len(benchmarks))
bar_w = 0.52

# ── colours matching the existing figure palette ──────────────────────────
C_BASE  = "#4a6fa5"   # muted blue  – baseline
C_QLORA = "#f0a500"   # amber       – QLoRA gain
C_COT   = "#e05c5c"   # rose        – CoT gain

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_base  = ax.bar(x, baseline,   bar_w, color=C_BASE,  label="Baseline (zero-shot)", zorder=3)
bars_qlora = ax.bar(x, qlora_gain, bar_w, bottom=baseline,
                    color=C_QLORA, label="QLoRA-SFT gain", zorder=3)
cot_bottom = [b + q for b, q in zip(baseline, qlora_gain)]
bars_cot   = ax.bar(x, cot_gain,   bar_w, bottom=cot_bottom,
                    color=C_COT,   label="CoT-SFT gain",   zorder=3)

# ── delta labels inside each gain segment ─────────────────────────────────
for i, (b, q, c, cb) in enumerate(zip(baseline, qlora_gain, cot_gain, cot_bottom)):
    if q >= 1.5:
        ax.text(x[i], b + q / 2, f"+{q:.1f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")
    if c >= 0.5:
        ax.text(x[i], cb + c / 2, f"+{c:.1f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")

# ── total accuracy label above each bar ───────────────────────────────────
for i, total in enumerate([b + q + c for b, q, c in zip(baseline, qlora_gain, cot_gain)]):
    ax.text(x[i], total + 0.6, f"{total:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#222222")

# ── baseline dashed reference lines ───────────────────────────────────────
for i, b in enumerate(baseline):
    ax.hlines(b, x[i] - bar_w / 2, x[i] + bar_w / 2,
              colors="#555555", linewidths=1.2, linestyles="--", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 108)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.set_title("Ablation: Accuracy Gain per Fine-Tuning Stage",
             fontsize=13, fontweight="bold", pad=12)

legend = ax.legend(handles=[
    mpatches.Patch(color=C_BASE,  label="Baseline (zero-shot)"),
    mpatches.Patch(color=C_QLORA, label="QLoRA-SFT gain"),
    mpatches.Patch(color=C_COT,   label="CoT-SFT gain"),
], fontsize=10, loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/fig9_ablation.png", dpi=180, bbox_inches="tight")
print("Saved figures/fig9_ablation.png")
