"""Conformal grid + prevalence-dependent optimal alpha."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import random
import statistics as st
from collections import defaultdict

C_FA, C_MISS = 2.0, 45.0
REPEATS, DRAWS = 10, 20

def load(paths):
    preds, core = defaultdict(dict), {}
    for s, path in enumerate(paths):
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (int(r["stage"]), int(r["pass_id"]))
                preds[key][s] = float(r["prediction"])
                core[key] = r.get("in_core45", "False").strip().lower() == "true"
    out = []
    for k, d in preds.items():
        g = k[0]
        gh = sum(d.values()) / len(d)
        out.append({"stage": g, "g": g, "ghat": gh, "abs_err": abs(gh - g), "core": core[k]})
    return [r for r in out if r["core"]]

def stage_split(rows, seed):
    stages = sorted({r["stage"] for r in rows})
    random.Random(seed).shuffle(stages)
    cal = set(stages[: len(stages) // 2])
    return ([r for r in rows if r["stage"] in cal],
            [r for r in rows if r["stage"] not in cal])

def q_of(cal, alpha):
    s = sorted(r["abs_err"] for r in cal)
    return s[min(len(s) - 1, int((len(s) + 1) * (1 - alpha)))]

def det(rows, q, lo, hi, rng):
    hit = tot = 0
    for r in rows:
        for _ in range(DRAWS):
            d = rng.randint(lo, hi)
            tot += 1
            hit += abs(r["ghat"] - (r["g"] + d)) > q
    return hit / tot

rows = load([f"aord_erm_s{s}.csv" for s in range(3)])
grid = []
for i in range(1, 46):
    a = i / 100
    qs, fas, dls, dss, covs = [], [], [], [], []
    for rep in range(REPEATS):
        cal, test = stage_split(rows, rep)
        q = q_of(cal, a)
        rng = random.Random(1000 + rep)
        fa = sum(1 for r in test if r["abs_err"] > q) / len(test)
        qs.append(q); fas.append(fa)
        dls.append(det(test, q, 15, 40, rng)); dss.append(det(test, q, 3, 8, rng))
        covs.append(1 - fa)
    grid.append((a, st.mean(qs), st.mean(covs), st.mean(fas), st.mean(dls), st.mean(dss)))

print("alpha   q      cov     FA     DetL    DetS")
for g in grid:
    print(f"{g[0]:5.2f} {g[1]:6.2f} {g[2]:6.3f} {g[3]:6.3f} {g[4]:6.3f} {g[5]:6.3f}")

print("\n=== alpha*(p) ===")
print("p      alpha*  q*     FA*    DetL*   cost_h  p*_breakeven")
for p in (0.005, 0.01, 0.02, 0.05):
    best = None
    for (a, q, cov, fa, dl, ds) in grid:
        risk = (1 - p) * C_FA * fa + p * C_MISS * (1 - dl)
        if best is None or risk < best[0]:
            best = (risk, a, q, fa, dl)
    risk, a, q, fa, dl = best
    hours = fa * 877 * C_FA / 60
    pstar = (fa * C_FA) / (fa * C_FA + dl * C_MISS)
    print(f"{p*100:5.1f}% {a:6.2f} {q:6.2f} {fa:6.3f} {dl:6.3f} {hours:7.2f} {100*pstar:8.2f}%")

print("\n=== site operating point ===")
for a in (0.265,):
    g = min(grid, key=lambda x: abs(x[0] - a))
    print(f"alpha={g[0]:.3f} q={g[1]:.2f} cov={g[2]:.3f} FA={g[3]:.3f} DetL={g[4]:.3f} DetS={g[5]:.3f}")
