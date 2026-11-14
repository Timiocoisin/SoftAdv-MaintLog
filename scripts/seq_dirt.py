"""Detection and false alarms of the record-only rule under dirty records.

Dirt = missing passes (gaps), which is what a real handwritten sheet produces.
Reported: false alarms on the record as-is, and incremental detection on rows
that were not already flagged.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import random
from collections import defaultdict

def load_core(nseeds=3):
    keep = {}
    for s in range(nseeds):
        with open(f"aord_erm_s{s}.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                k = (int(r["stage"]), int(r["pass_id"]))
                keep[k] = r.get("in_core45", "False").strip().lower() == "true"
    return sorted([k for k, v in keep.items() if v])

core = load_core()

def build(rows):
    d = defaultdict(set)
    for g, p in rows:
        d[g].add(p)
    return d

def gap(ps):
    return bool(ps) and set(range(max(ps) + 1)) != set(ps)

def flagged_after_slip(g, p, d, use_gap=True):
    dup = p in d.get(g + 1, set())
    if not use_gap:
        return dup
    src = set(d[g]) - {p}
    tgt = set(d.get(g + 1, set())) | {p}
    return dup or gap(src) or gap(tgt)

print(f"{'dirt':>6} {'FA(clean rows)':>15} {'det(incr.)':>12} {'det(dup only)':>14}")
for drop in (0.0, 0.02, 0.05, 0.10):
    fa_l, det_l, det_dup = [], [], []
    for rep in range(30):
        rng = random.Random(rep)
        kept = [r for r in core if drop == 0 or rng.random() >= drop]
        d = build(kept)
        # false alarms: rows sitting in a stage that already has a gap
        flagged = [r for r in kept if gap(d[r[0]])]
        fa = len(flagged) / len(kept)
        # eligible rows: those not already flagged
        elig = [r for r in kept if not gap(d[r[0]])]
        if not elig:
            continue
        det = sum(1 for g, p in elig if flagged_after_slip(g, p, d, True)) / len(elig)
        dup = sum(1 for g, p in elig if flagged_after_slip(g, p, d, False)) / len(elig)
        fa_l.append(fa); det_l.append(det); det_dup.append(dup)
    print(f"{int(drop*100):5d}% {sum(fa_l)/len(fa_l):15.3f} "
          f"{sum(det_l)/len(det_l):12.3f} {sum(det_dup)/len(det_dup):14.3f}")
