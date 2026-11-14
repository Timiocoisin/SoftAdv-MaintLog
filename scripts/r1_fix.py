"""R1 blocking items: (A) block-wise conformal coverage, (B) structured error injection."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import random
import statistics as st
from collections import defaultdict

ALPHA = 0.265
REPEATS = 20
CORE_CSV = "aord_erm_s0.csv"

def load(nseeds=3):
    preds, core = defaultdict(dict), {}
    for s in range(nseeds):
        with open(f"aord_erm_s{s}.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                k = (int(r["stage"]), int(r["pass_id"]))
                preds[k][s] = float(r["prediction"])
                core[k] = r.get("in_core45", "False").strip().lower() == "true"
    rows = []
    for k, d in preds.items():
        if not core[k]:
            continue
        g = k[0]
        rows.append({"stage": g, "pass": k[1], "g": g,
                     "ghat": sum(d.values()) / len(d)})
    return rows

rows = load()
stages = sorted({r["stage"] for r in rows})
# five contiguous stage blocks (quintiles of the stage range)
edges = [stages[int(len(stages) * q)] for q in (0.2, 0.4, 0.6, 0.8)]
def block(s):
    for i, e in enumerate(edges):
        if s < e:
            return i
    return len(edges)
for r in rows:
    r["block"] = block(r["stage"])

print("block sizes:", {b: sum(1 for r in rows if r["block"] == b) for b in range(5)})

def q_of(vals, alpha):
    v = sorted(vals)
    return v[min(len(v) - 1, int((len(v) + 1) * (1 - alpha)))]

print("\n=== (A) leave-one-block-out conformal ===")
covs, qs, sizes = [], [], []
for b in range(5):
    cal = [r for r in rows if r["block"] != b]
    test = [r for r in rows if r["block"] == b]
    q = q_of([abs(r["ghat"] - r["g"]) for r in cal], ALPHA)
    cov = sum(1 for r in test if abs(r["ghat"] - r["g"]) <= q) / len(test)
    covs.append(cov); qs.append(q); sizes.append(len(test))
    print(f"block {b}: n_train={len(cal):3d} n_test={len(test):3d} q={q:6.2f} coverage={cov:.3f}")

print(f"\ncoverage: min={min(covs):.3f} median={st.median(covs):.3f} max={max(covs):.3f} "
      f"(target {1-ALPHA:.3f})")
print(f"q: min={min(qs):.2f} median={st.median(qs):.2f} max={max(qs):.2f}")

print("\n=== (B) structured error injection ===")
# calibration q from a random half, evaluated on the other half, averaged
def det_for(transform, q, test):
    hit = 0
    for r in test:
        claim = transform(r["g"])
        if claim is None or claim < 1 or claim > 169:
            continue
        hit += abs(r["ghat"] - claim) > q
    return hit / len(test)

def transposed(g):
    a, b = g // 10, g % 10
    if a == 0 or a == b:
        return None
    return b * 10 + a

patterns = {
    "off-by-one (+1)": lambda g: g + 1,
    "off-by-one (-1)": lambda g: g - 1,
    "off-by-two (+2)": lambda g: g + 2,
    "off-by-three (+3)": lambda g: g + 3,
    "digit transposition": transposed,
}

res = defaultdict(list)
for rep in range(REPEATS):
    rng = random.Random(rep)
    stages_shuffled = stages[:]
    rng.shuffle(stages_shuffled)
    cal_stages = set(stages_shuffled[: len(stages_shuffled) // 2])
    cal = [r for r in rows if r["stage"] in cal_stages]
    test = [r for r in rows if r["stage"] not in cal_stages]
    q = q_of([abs(r["ghat"] - r["g"]) for r in cal], ALPHA)
    for name, fn in patterns.items():
        res[name].append(det_for(fn, q, test))
    # uniform baselines, sampled
    for name, lo, hi in (("uniform 3-8 (current)", 3, 8), ("uniform 15-40 (current)", 15, 40)):
        vals = []
        for r in test:
            hits = tot = 0
            for _ in range(20):
                d = rng.randint(lo, hi)
                tot += 1
                hits += abs(r["ghat"] - (r["g"] + d)) > q
            vals.append(hits / tot)
        res[name].append(sum(vals) / len(vals))

print(f"{'pattern':26s} {'Det @ FA=0.265':>16s}")
for name in list(patterns) + ["uniform 3-8 (current)", "uniform 15-40 (current)"]:
    v = res[name]
    print(f"{name:26s} {st.mean(v):8.3f} ± {st.pstdev(v):.3f}")
