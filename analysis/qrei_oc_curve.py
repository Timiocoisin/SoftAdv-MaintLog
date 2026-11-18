#!/usr/bin/env python3
"""Operating-characteristic curve for the competence criterion.

The criterion accepts a class when the lower bound of  Det - FA  clears the site margin
lambda. Writing the bootstrap standard error of that difference as s, the lower bound is
approximately mu - 1.96 s, so the criterion accepts with probability

    P(accept | mu) = Phi( (mu - lambda) / s - 1.96 ).

s is measured by the paired bootstrap, separately for the two resampling units. The curve
gives the margin at which acceptance is a coin flip (P = 0.5) and the margin at which it is
0.8, which is what a site has to know before it issues a card.

Also reports the number of stage clusters needed to move the 0.5 point to a stated margin.

Usage
    python analysis/qrei_oc_curve.py [--scores data/scores_per_pass.csv]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import norm


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["pid"] = (int(r["stage"]), int(r["pass_index"]))
        r["flagged"] = int(r["flagged"])
    return rows


def se_units(rows, target_class, draws, seed):
    correct = {r["pid"]: r["flagged"] for r in rows if r["class"] == "correct"}
    target = {r["pid"]: r["flagged"] for r in rows if r["class"] == target_class}
    pids = sorted(correct)
    stage_of = np.array([p[0] for p in pids])
    c = np.array([correct[p] for p in pids], float)
    t = np.array([target[p] for p in pids], float)

    rng = np.random.default_rng(seed)
    n = len(c)

    bp = np.empty(draws)
    for b in range(draws):
        ii = rng.integers(0, n, n)
        bp[b] = t[ii].mean() - c[ii].mean()

    stages = np.unique(stage_of)
    idx = {s: np.where(stage_of == s)[0] for s in stages}
    k = len(stages)
    bc = np.empty(draws)
    for b in range(draws):
        pick = rng.integers(0, k, k)
        ii = np.concatenate([idx[stages[i]] for i in pick])
        bc[b] = t[ii].mean() - c[ii].mean()

    return {
        "point": float(t.mean() - c.mean()),
        "pass": {"sd": float(bp.std(ddof=1)), "lower": float(np.percentile(bp, 2.5))},
        "cluster": {"sd": float(bc.std(ddof=1)), "lower": float(np.percentile(bc, 2.5))},
        "n_stages": int(k),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores_per_pass.csv")
    ap.add_argument("--class", dest="target_class", default="mirror")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.50)
    ap.add_argument("--draws", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--out")
    a = ap.parse_args()

    d = se_units(load(a.scores), a.target_class, a.draws, a.seed)
    out = {"class": a.target_class, "lambda": a.lam, "point": round(d["point"], 4),
           "draws": a.draws, "seed": a.seed}
    for unit in ("pass", "cluster"):
        s = d[unit]["sd"]
        mu50 = a.lam + 1.96 * s
        mu80 = mu50 + norm.ppf(0.8) * s
        acc = float(norm.cdf((d["point"] - a.lam) / s - 1.96))
        out[unit] = {
            "sd": round(s, 4),
            "lower": round(d[unit]["lower"], 3),
            "mu50": round(float(mu50), 3),
            "mu80": round(float(mu80), 3),
            "accept_at_observed": round(acc, 3),
        }

    # clusters needed to move the cluster-unit mu50 to a stated margin
    s = d["cluster"]["sd"]
    k = d["n_stages"]
    need = {}
    for target in (0.60, 0.62, 0.65):
        s_t = (target - a.lam) / 1.96
        need["%.2f" % target] = int(np.ceil(k * (s / s_t) ** 2))
    out["clusters_needed"] = need
    out["n_stages"] = d["n_stages"]

    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
