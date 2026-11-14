"""Recompute every pattern row on the detection-board protocol: thr=10, all 234 passes."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = ROOT_DIR
OUT = ROOT / "outputs/softadv_protocol_20261106"
CENTRES = {25, 55, 85, 115, 145}
H = 4
CORE = sorted({g for c in CENTRES for g in range(c - H, c + H + 1) if 1 <= g <= 169})
THR = 10.0

def load_seedavg(pattern, seeds=(0, 1, 2)):
    bags = {}
    for s in seeds:
        df = pd.read_csv(next(OUT.glob(pattern.format(seed=s))))
        for _, r in df.iterrows():
            bags.setdefault((int(r["stage"]), int(r["pass_id"])), []).append(float(r["prediction"]))
    return pd.DataFrame([{"stage": k[0], "pass_id": k[1], "ghat": float(np.mean(v))}
                         for k, v in bags.items()])

feat = pd.read_csv(ROOT / "outputs/idea_validation_20261106/pass_features.csv")
te = feat[feat.stage.isin(CORE)]
core = load_seedavg("auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv")
core = core[core.stage.isin(CORE)].merge(te[["stage", "pass_id", "bright_40"]], on=["stage", "pass_id"])
g = core["stage"].to_numpy(float)
ghat = core["ghat"].to_numpy(float)
rng = np.random.default_rng(0)

def det(claim):
    ok = ~np.isnan(claim)
    hit = (np.abs(ghat[ok] - claim[ok]) > THR).mean()
    return hit, int(ok.sum())

rows = []
rows.append(("off-by-one (+1)", det(g + 1)))
rows.append(("off-by-two", det(g + 2)))
rows.append(("off-by-three", det(g + 3)))

trans = []
for s in g:
    a, b = int(s) // 10, int(s) % 10
    trans.append(float(b * 10 + a) if (a != 0 and a != b) else np.nan)
rows.append(("digit transposition", det(np.array(trans))))

small = g + rng.integers(3, 9, size=len(g))
rows.append(("uniform small (3-8)", det(small.astype(float))))
large = g + rng.integers(15, 41, size=len(g))
rows.append(("uniform large (15-40)", det(large.astype(float))))

print(f"{'pattern':24s} {'Det':>6} {'n':>4}")
for name, (d, n) in rows:
    print(f"{name:24s} {d:6.3f} {n:4d}")
