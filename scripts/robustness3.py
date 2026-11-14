"""Is the qualification robust? Three checks that need no new data.

(a) Seed stability: the detector was trained with three seeds and the scored estimate
    is their average. Does each seed produce the same class verdicts?
(b) Threshold stability: the verdicts are taken at thr = 10. Do they survive a sweep?
(c) Block jackknife: leaving one stage block out, does the card change?
"""
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
TAU = 0.475

feat = pd.read_csv(ROOT / "outputs/idea_validation_20261106/pass_features.csv")
te = feat[feat.stage.isin(CORE)][["stage", "pass_id", "bright_40"]]

def load(seed):
    p = OUT / f"auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv"
    df = pd.read_csv(p)
    df = df[df.stage.isin(CORE)][["stage", "pass_id", "prediction"]].rename(columns={"prediction": "ghat"})
    return df.merge(te, on=["stage", "pass_id"])

claims_of = ["off-by-one", "off-by-two", "off-by-three", "transposition",
             "small 3-8", "large 15-40"]

def detections(df, thr, rng):
    g = df.stage.to_numpy(float)
    gh = df.ghat.to_numpy(float)
    trans = np.array([float((int(s) % 10) * 10 + int(s) // 10)
                      if (int(s) // 10 != 0 and int(s) // 10 != int(s) % 10) else np.nan for s in g])
    small = g + rng.integers(3, 9, size=len(g)).astype(float)
    large = g + rng.integers(15, 41, size=len(g)).astype(float)
    out = {}
    for name, claim in zip(claims_of, [g + 1, g + 2, g + 3, trans, small, large]):
        ok = ~np.isnan(claim)
        out[name] = float((np.abs(gh - claim) > thr)[ok].mean())
    fa = float((np.abs(gh - g) > thr).mean())
    return out, fa

# ---------------------------------------------------------------- (a) seeds
frames = {s: load(s) for s in (0, 1, 2)}
rng = np.random.default_rng(0)
det_seed, fa_seed = {}, {}
for s, df in frames.items():
    d, fa = detections(df, 10.0, np.random.default_rng(0))
    det_seed[s], fa_seed[s] = d, fa

avg = pd.DataFrame({s: frames[s].set_index(["stage", "pass_id"]).ghat for s in frames}).mean(axis=1).reset_index()
avg = avg.rename(columns={0: "ghat"}).merge(te, on=["stage", "pass_id"])
d_avg, fa_avg = detections(avg, 10.0, np.random.default_rng(0))

print("=== (a) per-seed class detection and verdict at tau = 0.475 ===")
hdr = f"{'class':15s} {'s0':>7s} {'s1':>7s} {'s2':>7s} {'avg':>7s}   verdicts (s0/s1/s2/avg)"
print(hdr)
for c in claims_of:
    vs = []
    for s in (0, 1, 2):
        m = det_seed[s][c] - fa_seed[s]
        vs.append("Q" if m > TAU else "x")
    m_avg = d_avg[c] - fa_avg
    vs.append("Q" if m_avg > TAU else "x")
    print(f"{c:15s} {det_seed[0][c]:7.3f} {det_seed[1][c]:7.3f} {det_seed[2][c]:7.3f} {d_avg[c]:7.3f}"
          f"   {'/'.join(vs)}   margins {det_seed[0][c]-fa_seed[0]:+.2f}/{det_seed[1][c]-fa_seed[1]:+.2f}/{det_seed[2][c]-fa_seed[2]:+.2f}/{m_avg:+.2f}")
print(f"FA per seed: {fa_seed[0]:.3f} {fa_seed[1]:.3f} {fa_seed[2]:.3f}  avg {fa_avg:.3f}")

# ---------------------------------------------------------------- (b) threshold sweep
print("\n=== (b) verdicts across thresholds (average seed) ===")
print(f"{'thr':>5s} {'FA':>6s} " + " ".join(f"{c[:9]:>9s}" for c in claims_of))
for thr in (6, 8, 10, 12, 15):
    d, fa = detections(avg, float(thr), np.random.default_rng(0))
    cells = []
    for c in claims_of:
        m = d[c] - fa
        cells.append(f"{m:+.2f}" + ("Q" if m > TAU else " "))
    print(f"{thr:5d} {fa:6.3f} " + " ".join(f"{x:>9s}" for x in cells))
