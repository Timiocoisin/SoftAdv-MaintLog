"""Is the write-time result robust to the order in which the crew writes?

The record has no timestamps, so the writing order is an assumption. We test it:
run the latency analysis under the sorted order, the reverse order, and random
permutations, and report what fraction of detected shifts are rejected by the write
that enters them.

Two rules can fire:
  duplicate   - the shifted row lands on a pass number already written (or one that
                will arrive later), so it fires at max(0, pos(target) - i)
  contiguity  - the stage's pass set is short, which is only knowable once the last
                row of that stage has been written, at max(pos) of that stage
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import random
from collections import defaultdict

PATH = str(ROOT_DIR / "outputs/idea_validation_20261106/pass_features.csv")

rows = []
with open(PATH, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["stage"]), int(r["pass_id"])))
N = len(rows)
last_stage = max(g for g, _ in rows)

by_stage = defaultdict(list)
for g, p in rows:
    by_stage[g].append(p)

def has_gap(ps):
    return bool(ps) and set(range(max(ps) + 1)) != set(ps)

def analyse(order, delta):
    pos = {row: i for i, row in enumerate(order)}
    stage_last = defaultdict(int)
    for i, (g, _) in enumerate(order):
        stage_last[g] = i

    hits, at_write, within, later = 0, 0, 0, 0
    for i, (g, p) in enumerate(order):
        gt = g + delta
        if not (1 <= gt <= last_stage):
            continue
        lat = None
        # duplicate: collide with the row at (gt, p) if one exists in the record
        if (gt, p) in pos:
            lat = max(0, pos[(gt, p)] - i)
        # contiguity: the short stage is only knowable when its last row is written
        for s in (g, gt):
            ps = set(by_stage[s])
            if s == g:
                ps.discard(p)
            else:
                ps.add(p)
            if has_gap(ps):
                lat_s = stage_last[s] - i
                lat = lat_s if lat is None else min(lat, lat_s)
        if lat is None:
            continue
        hits += 1
        if lat == 0:
            at_write += 1
        elif lat <= 9:
            within += 1
        else:
            later += 1
    return hits, at_write, within, later

def show(name, order):
    for delta in (1, -1):
        h, w, s, l = analyse(order, delta)
        print(f"  {name:12s} delta {delta:+d}: hits {h:3d}  at-write {w:3d}  "
              f"within9 {s:3d}  later {l:3d}")

sorted_order = sorted(rows)
print("sorted writing order (the assumption):")
show("sorted", sorted_order)

print("\nreverse order:")
show("reverse", list(reversed(sorted_order)))

print("\nrandom permutations (50):")
random.seed(0)
for delta in (1, -1):
    aw, hi = [], []
    for _ in range(50):
        o = sorted_order[:]
        random.shuffle(o)
        h, w, s_, l = analyse(o, delta)
        aw.append(w)
        hi.append(h)
    print(f"  delta {delta:+d}: at-write mean {sum(aw)/len(aw):.1f} of {sum(hi)/len(hi):.1f} hits "
          f"({100*sum(aw)/max(1,sum(hi)):.1f}%)  range {min(aw)}-{max(aw)}")
