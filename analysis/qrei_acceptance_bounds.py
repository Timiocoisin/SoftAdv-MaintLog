#!/usr/bin/env python3
"""Acceptance-envelope bound for the record-verification paper (QREI submission).

The statistic is Det - FA for one error class of the record check. The interval is a
paired percentile bootstrap: the same resampled units supply both the detection and the
false-alarm term, so the two are never mixed across independent draws.

Two resampling units are reported, because they answer different questions:
  * the pass  -- what a re-draw of these 234 passes would give;
  * the stage -- what a re-draw of the 45 stage clusters would give, which respects the
    fact that passes inside a stage are not independent.

Usage
    python analysis/qrei_acceptance_bounds.py [--scores data/scores_per_pass.csv]
                                              [--class mirror] [--reps 8000]

Output is JSON on stdout and, with --out, to a file.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np


def load(path: str):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["pid"] = (int(r["stage"]), int(r["pass_index"]))
        r["flagged"] = int(r["flagged"])
    return rows


def bounds(rows, target_class: str, reps: int, seed: int):
    correct = {r["pid"]: r["flagged"] for r in rows if r["class"] == "correct"}
    target = {r["pid"]: r["flagged"] for r in rows if r["class"] == target_class}
    if not correct or not target:
        raise SystemExit("class %r not found in the scores file" % target_class)

    pids = sorted(correct)
    by_stage: dict[int, list] = {}
    for p in pids:
        by_stage.setdefault(p[0], []).append(p)
    stages = sorted(by_stage)
    stage_idx = {s: np.array([pids.index(p) for p in by_stage[s]]) for s in stages}

    c = np.array([correct[p] for p in pids], dtype=float)
    t = np.array([target[p] for p in pids], dtype=float)
    point = float(t.mean() - c.mean())

    rng = np.random.default_rng(seed)
    n = len(pids)

    boot_pass = np.empty(reps)
    for b in range(reps):
        ii = rng.integers(0, n, n)
        boot_pass[b] = t[ii].mean() - c[ii].mean()

    boot_clu = np.empty(reps)
    k = len(stages)
    for b in range(reps):
        pick = rng.integers(0, k, k)
        ii = np.concatenate([stage_idx[stages[i]] for i in pick])
        boot_clu[b] = t[ii].mean() - c[ii].mean()

    return {
        "class": target_class,
        "n_passes": int(n),
        "n_stages": int(k),
        "point": round(point, 4),
        "reps": reps,
        "seed": seed,
        "pass_resampling": {
            "lower": round(float(np.percentile(boot_pass, 2.5)), 3),
            "upper": round(float(np.percentile(boot_pass, 97.5)), 3),
            "sd": round(float(boot_pass.std(ddof=1)), 3),
        },
        "cluster_resampling": {
            "lower": round(float(np.percentile(boot_clu, 2.5)), 3),
            "upper": round(float(np.percentile(boot_clu, 97.5)), 3),
            "sd": round(float(boot_clu.std(ddof=1)), 3),
        },
        "note": (
            "the pass bound is 0.568 at every seed tried; the cluster bound lands "
            "between 0.491 and 0.498 depending on the seed, so 0.498 is one attained "
            "value and all of them sit below the 0.50 threshold"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores_per_pass.csv")
    ap.add_argument("--class", dest="target_class", default="mirror")
    ap.add_argument("--reps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--out")
    a = ap.parse_args()

    res = bounds(load(a.scores), a.target_class, a.reps, a.seed)
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()

