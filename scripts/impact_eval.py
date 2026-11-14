"""Protocol transfer to IMPACT (industrial assembly, ordered steps).

Leave-one-view-out inside each trial: prototypes from 4 views, evaluate on the
5th. Decision rule identical to the paper: REVIEW if |ghat - g_log| > q.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import json
import os
import random
import statistics as st
from collections import OrderedDict

import numpy as np

ROOT = str(ROOT_DIR / "datasets/impact")
FEAT = f"{ROOT}/sample/IMPACT-v1.1/sample/features/I3D"
LAB = f"{ROOT}/annot/IMPACT-v1.1/annotations/TAS-B/groundTruth_TAS-BL"
VIEWS = ["ego", "front", "left", "right", "top"]
MIN_RUN = 8

trials = sorted({f.rsplit("_", 1)[0] for f in os.listdir(FEAT) if f.endswith(".npy")})
print("trials:", trials)

def load(trial, view):
    xp = os.path.join(FEAT, f"{trial}_{view}.npy")
    lp = os.path.join(LAB, f"{trial}_{view}.txt")
    if not (os.path.exists(xp) and os.path.exists(lp)):
        return None
    X = np.load(xp, allow_pickle=True)
    y = [l.strip() for l in open(lp, encoding="utf-8")]
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]
    keep = [i for i in range(n) if y[i] and y[i] != "NULL"]
    if not keep:
        return None
    X = X[keep]
    y = [y[i] for i in keep]
    # contiguous runs
    runs, start = [], 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            if i - start >= MIN_RUN:
                runs.append((y[start], X[start:i].mean(0)))
            start = i
    return runs

def ordinal_map(runs):
    order = OrderedDict()
    for name, _ in runs:
        if name not in order:
            order[name] = len(order)
    return order

results = []
for trial in trials:
    data = {}
    for v in VIEWS:
        r = load(trial, v)
        if r:
            data[v] = r
    if len(data) < 3:
        continue
    order = ordinal_map([r for v in data for r in data[v]])
    G = len(order)
    n_runs = {v: len(data[v]) for v in data}
    print(f"{trial}: G={G}, runs per view={n_runs}")

    for held in data:
        others = [v for v in data if v != held]
        proto = {}
        for name, idx in order.items():
            vecs = [r[1] for v in others for r in data[v] if r[0] == name]
            if vecs:
                p = np.mean(vecs, 0)
                proto[name] = p / (np.linalg.norm(p) + 1e-9)

        def margin(name, feat):
            if name not in proto:
                return None
            f = feat / (np.linalg.norm(feat) + 1e-9)
            s = float(f @ proto[name])
            best = -2.0
            i0 = order[name]
            for nm, i in order.items():
                if 1 <= abs(i - i0) <= 2 and nm in proto:
                    best = max(best, float(f @ proto[nm]))
            return s - best if best > -2 else 0.0

        margins = []
        for name, feat in data[held]:
            m = margin(name, feat)
            if m is not None:
                margins.append((name, feat, m))
        if len(margins) < 8:
            continue
        q = float(np.percentile([m for _, _, m in margins], 26.5))
        fa = sum(1 for _, _, m in margins if m < q) / len(margins)

        def det(delta):
            rng = random.Random(11)
            hit = tot = 0
            for name, feat, _ in margins:
                for _ in range(10):
                    i = order[name] + rng.choice([-delta, delta])
                    cands = [nm for nm, k in order.items() if k == i]
                    if not cands:
                        continue
                    m = margin(cands[0], feat)
                    if m is None:
                        continue
                    tot += 1
                    hit += m < q
            return hit / max(1, tot)

        results.append({"trial": trial, "held": held, "G": G, "n": len(margins),
                        "FA": fa, "D1": det(1), "D2": det(2), "D3": det(3)})

if results:
    print("\nview  trial                 G   n    FA    D1    D2    D3")
    for r in results:
        print(f"{r['held']:5s} {r['trial']:20s} {r['G']:3d} {r['n']:3d} "
              f"{r['FA']:.3f} {r['D1']:.3f} {r['D2']:.3f} {r['D3']:.3f}")
    for k in ("FA", "D1", "D2", "D3"):
        vals = [r[k] for r in results]
        print(f"mean {k}: {st.mean(vals):.3f} ± {st.pstdev(vals):.3f}")
    json.dump(results, open("impact_result.json", "w"), indent=2)
