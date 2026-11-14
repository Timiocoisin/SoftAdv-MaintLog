"""Confidence intervals for every pass/fail verdict in the protocol.

The protocol is a threshold instrument, so a verdict that sits inside sampling error
is not a verdict. We compute Wilson score intervals for each rate, a paired bootstrap
for each margin Det - FA (the two rates come from the same passes, so they are not
independent), and the rule of three for a zero count.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

ROOT = ROOT_DIR
OUT = ROOT / "outputs/softadv_protocol_20261106"
CENTRES = {25, 55, 85, 115, 145}
H = 4
CORE = sorted({g for c in CENTRES for g in range(c - H, c + H + 1) if 1 <= g <= 169})
THR = 10.0
Z = 1.959963985

def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

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
n = len(g)
rng = np.random.default_rng(0)

# false-review indicator and its interval
fa_ind = (np.abs(ghat - g) > THR).astype(int)
FA = fa_ind.mean()
lo, hi = wilson(fa_ind.sum(), n)
print(f"false-review rate on the core: {FA:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]  (n={n})\n")

trans = np.array([float((int(s) % 10) * 10 + int(s) // 10)
                  if (int(s) // 10 != 0 and int(s) // 10 != int(s) % 10) else np.nan for s in g])
small = g + rng.integers(3, 9, size=n).astype(float)
large = g + rng.integers(15, 41, size=n).astype(float)

claims = [("off-by-one", g + 1), ("off-by-two", g + 2), ("off-by-three", g + 3),
          ("digit transposition", trans), ("uniform small 3-8", small),
          ("uniform large 15-40", large)]

TAU = 0.475  # derived at p = 1.25%, c_miss = 45 min
print(f"{'class':22s} {'Det':>6} {'Wilson':>15} {'margin':>7} {'boot 95%':>16}  verdict@0.475")
rows = []
for name, claim in claims:
    ok = ~np.isnan(claim)
    hit = (np.abs(ghat - claim) > THR)[ok].astype(int)
    k, m = int(hit.sum()), int(ok.sum())
    det = k / m
    dl, dh = wilson(k, m)
    # paired bootstrap over passes for the margin
    boots = []
    idx = np.arange(n)
    base_hit = np.zeros(n, dtype=float)
    base_hit[ok] = hit
    okf = ok.astype(float)
    for _ in range(2000):
        s = rng.integers(0, n, n)
        mask = okf[s] > 0
        if mask.sum() == 0:
            continue
        d = base_hit[s][mask].mean()
        f = fa_ind[s].mean()
        boots.append(d - f)
    bl, bh = np.percentile(boots, [2.5, 97.5])
    verdict = "qualified" if (bl > TAU) else ("excluded" if bh < TAU else "INDETERMINATE")
    rows.append((name, det, (dl, dh), det - FA, (bl, bh), verdict))
    print(f"{name:22s} {det:6.3f} [{dl:.3f},{dh:.3f}] {det-FA:+7.3f} [{bl:+.3f},{bh:+.3f}]  {verdict}")

print("\n=== record-only check, full record (n=877) ===")
for label, k in (("back-shift, both rules", 862), ("forward-shift, both rules", 859)):
    lo_, hi_ = wilson(k, 877)
    print(f"  {label:26s} {k}/877 = {k/877:.3f}  Wilson [{lo_:.3f}, {hi_:.3f}]")

print("\n=== write-time rejections under the measured order (n=857) ===")
lo_, hi_ = wilson(845, 857)
print(f"  back-shift at the write      845/857 = {845/857:.3f}  Wilson [{lo_:.3f}, {hi_:.3f}]")
print(f"  forward-shift at the write   0/854  = 0.000  rule of three: < {3/854:.4f} (95%)")
