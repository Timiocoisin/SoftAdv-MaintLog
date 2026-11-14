import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

BLUE = "#0072B2"
VERM = "#D55E00"
GREY = "#6E6E6E"

N, FA, CFA, CMISS = 877, 0.265, 1.57, 45.0
run_h = N * FA * CFA / 60.0
D_meas = (0.987 + 0.922 + 0.372 + 0.291 + 0.274 + 0.265) / 6
D_card = (0.987 + 0.922) / 6
p_meas = FA * CFA / (CMISS * D_meas)
p_card = FA * CFA / (CMISS * D_card)

p = np.linspace(0.0, 0.06, 400)
manual = p * N * CMISS / 60.0
verif = run_h + p * N * CMISS * (1 - D_meas) / 60.0
card = run_h + p * N * CMISS * (1 - D_card) / 60.0

fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.plot(p * 100, manual, color=GREY, lw=1.5, label="manual review of every pass")
ax.plot(p * 100, verif, color=BLUE, lw=1.7, label="verifier, as measured")
ax.plot(p * 100, card, color=VERM, lw=1.2, ls="--", label="verifier, card-only accounting")

for pc, col in ((p_meas, BLUE), (p_card, VERM)):
    yc = pc * N * CMISS / 60.0
    ax.plot([pc * 100], [yc], "o", ms=5, color=col, mec="white", mew=0.8, zorder=5)
    ax.plot([pc * 100, pc * 100], [0, yc], color=col, lw=0.7, ls=":", zorder=1)

ax.text(p_meas * 100 - 0.12, 0.7, f"{p_meas*100:.1f}%", fontsize=8, color=BLUE, ha="right")
ax.text(p_card * 100 + 0.12, 0.7, f"{p_card*100:.1f}%", fontsize=8, color=VERM, ha="left")

ax.set_xlabel("wrong-log prevalence $p$ (%)", fontsize=8.6)
ax.set_ylabel("expected cost per campaign (h)", fontsize=8.6)
ax.set_xlim(0, 6)
ax.set_ylim(0, 22)
ax.legend(frameon=False, fontsize=7.2, loc="upper left", handlelength=1.6)
plt.tight_layout(pad=0.25)
fig.savefig(str(ROOT_DIR / "figures/Fig06_breakeven.pdf"))
fig.savefig(str(ROOT_DIR / "figures/Fig06_breakeven.png"), dpi=220)
plt.close(fig)
print(f"p_meas {p_meas*100:.2f}%  p_card {p_card*100:.2f}%")
