"""Block jackknife: is the card the same inside every stage block?

Five blocks of nine stages make up the scored core. We score the classes inside each
block separately, and then leaving one block out, to see whether the pooled verdicts
come from agreement across blocks or from one block dominating.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = ROOT_DIR
OUT = ROOT / "outputs/softadv_protocol_20261106"
CENTRES = [25, 55, 85, 115, 145]
H = 4
THR, TAU = 10.0, 0.475

feat = pd.read_csv(ROOT / "outputs/idea_validation_20261106/pass_features.csv")
core_stages = {g for c in CENTRES for g in range(c - H, c + H + 1)}
te = feat[feat.stage.isin(core_stages)][["stage", "pass_id", "bright_40"]]

frames = []
for s in (0, 1, 2):
    df = pd.read_csv(OUT / f"auditord__multi_gap_guard10__aord_erm__seed{s}_predictions.csv")
    frames.append(df.set_index(["stage", "pass_id"])["prediction"])
ghat_avg = pd.concat(frames, axis=1).mean(axis=1).rename("ghat").reset_index()
data = ghat_avg.merge(te, on=["stage", "pass_id"])

blocks = {c: data[(data.stage >= c - H) & (data.stage <= c + H)].copy() for c in CENTRES}

def class_det(df, rng):
    g = df.stage.to_numpy(float)
    gh = df.ghat.to_numpy(float)
    trans = np.array([float((int(s) % 10) * 10 + int(s) // 10)
                      if (int(s) // 10 != 0 and int(s) // 10 != int(s) % 10) else np.nan for s in g])
    small = g + rng.integers(3, 9, size=len(g)).astype(float)
    large = g + rng.integers(15, 41, size=len(g)).astype(float)
    fa = float((np.abs(gh - g) > THR).mean())
    out = {}
    for name, claim in (("off-by-one", g + 1), ("off-by-two", g + 2), ("off-by-three", g + 3),
                        ("transposition", trans), ("small", small), ("large", large)):
        ok = ~np.isnan(claim)
        out[name] = float((np.abs(gh - claim) > THR)[ok].mean())
    return out, fa

names = ["off-by-one", "off-by-two", "off-by-three", "transposition", "small", "large"]
print("=== margin Det - FA inside each block (thr = 10, n per block shown) ===")
print(f"{'block':>6s} {'n':>4s} {'FA':>6s} " + " ".join(f"{c[:9]:>10s}" for c in names))
per_block = {}
for c, df in blocks.items():
    d, fa = class_det(df, np.random.default_rng(0))
    per_block[c] = (d, fa)
    cells = " ".join(f"{d[n]-fa:+10.2f}" for n in names)
    print(f"{c:6d} {len(df):4d} {fa:6.3f} {cells}")

print("\n=== leaving one block out ===")
print(f"{'held out':>9s} {'n':>4s} {'FA':>6s} " + " ".join(f"{c[:9]:>10s}" for c in names))
for c in CENTRES:
    rest = pd.concat([b for k, b in blocks.items() if k != c])
    d, fa = class_det(rest, np.random.default_rng(0))
    cells = " ".join(f"{d[n]-fa:+10.2f}" + ("Q" if d[n]-fa > TAU else " ") for n in names)
    print(f"{c:9d} {len(rest):4d} {fa:6.3f} {cells}")

# verdict agreement
print("\n=== verdicts per block at tau = 0.475 (Q = qualified) ===")
for c in CENTRES:
    d, fa = per_block[c]
    vs = "".join("Q" if d[n] - fa > TAU else "." for n in names)
    print(f"  block {c:3d}: {vs}")
print("            " + "".join(f"{n[:1]:>1s}" for n in names) + "   (o=off-by-one, t=transposition, l=large)")
