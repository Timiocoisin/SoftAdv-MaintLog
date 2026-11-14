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
BAND = "#E8E8E8"

classes = [
    ("off-by-one", 0.265, 0.213, 0.325, False),
    ("off-by-two", 0.274, 0.220, 0.334, False),
    ("off-by-three", 0.291, 0.236, 0.352, False),
    ("uniform small (3-8)", 0.372, 0.312, 0.435, False),
    ("digit transposition", 0.922, 0.879, 0.951, True),
    ("uniform large (15-40)", 0.987, 0.963, 0.996, True),
]

fig, ax = plt.subplots(figsize=(6.6, 3.15))
y = np.arange(len(classes))
ax.axvspan(0.29, 0.84, color=BAND, lw=0, zorder=0)
ax.axvline(0.475, color=GREY, lw=0.9, ls="--", zorder=1)

for i, (name, det, lo, hi, qual) in enumerate(classes):
    c = BLUE if qual else VERM
    ax.plot([lo, hi], [i, i], color=c, lw=1.5, solid_capstyle="round", zorder=3)
    ax.plot([det], [i], "o", ms=5.5, color=c, mec="white", mew=0.8, zorder=4)
    ax.text(hi + 0.012, i, f"{det:.3f}", va="center", fontsize=7.4, color=c)

ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in classes], fontsize=8.6)
ax.set_xlim(0.15, 1.12)
ax.set_ylim(-1.05, len(classes) - 0.35)
ax.set_xlabel("detection rate for the class (Wilson 95% interval)", fontsize=8.6)

# dashed bar label, at the top of its own line, inside the axes
ax.text(0.482, len(classes) - 0.55, "0.475", fontsize=7.2, color=GREY, ha="left", va="center")
# band label, under the points, inside the shaded region
ax.text(0.565, -0.72, "bar implied by the elicited miss cost, 30-60 min",
        fontsize=7.2, color="#4D4D4D", ha="center", va="center")
# chance note, in the empty corner
ax.text(1.11, 1.15, "off-by-one catches at the\nfalse-review rate 0.265",
        fontsize=7.2, color=VERM, ha="right", va="center")

plt.tight_layout(pad=0.25)
fig.savefig(str(ROOT_DIR / "figures/Fig05_class_qualification.pdf"))
fig.savefig(str(ROOT_DIR / "figures/Fig05_class_qualification.png"), dpi=220)
plt.close(fig)
print("fig A ok")
