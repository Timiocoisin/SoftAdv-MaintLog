"""Counts per split: training, guard band, scored core."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
from collections import defaultdict

CENTRES = [25, 55, 85, 115, 145]
H = 4
GUARD = 10

rows = []
with open(str(ROOT_DIR / "outputs/idea_validation_20261106/pass_features.csv"),
          encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(int(r["stage"]))

stages = sorted(set(rows))
total_passes = len(rows)
total_stages = len(stages)

def role(g):
    for c in CENTRES:
        if abs(g - c) <= H:
            return "core"
    for c in CENTRES:
        if abs(g - c) <= H + GUARD:
            return "guard"
    return "train"

per_stage = defaultdict(int)
for g in rows:
    per_stage[g] += 1

counts = defaultdict(lambda: [0, 0])
for g in stages:
    r = role(g)
    counts[r][0] += 1
    counts[r][1] += per_stage[g]

print(f"{'split':8s} {'stages':>7} {'passes':>7}")
for k in ("train", "guard", "core"):
    s, p = counts[k]
    print(f"{k:8s} {s:7d} {p:7d}")
print(f"{'total':8s} {total_stages:7d} {total_passes:7d}")

frames_per_pass = 17511 / total_passes
print(f"\nframes per pass ~ {frames_per_pass:.2f}")
for k in ("train", "guard", "core"):
    print(f"{k:8s} frames ~ {counts[k][1] * frames_per_pass:7.0f}")
