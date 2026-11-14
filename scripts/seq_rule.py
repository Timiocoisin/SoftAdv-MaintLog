"""Two-tier record-only rule: duplicate (needs nothing) vs gap (needs contiguity)."""
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

def flags(g, p, delta, d, use_gap=True):
    """Would the rule flag the record if (g,p) were mis-written as (g+delta,p)?"""
    dup = p in d.get(g + delta, set())
    if not use_gap:
        return dup
    src = set(d[g]) - {p}
    tgt = set(d.get(g + delta, set())) | {p}
    return dup or gap(src) or gap(tgt)

d0 = build(core)
print(f"clean record: {len(core)} rows, pre-existing gaps: "
      f"{sum(1 for s in d0 if gap(d0[s]))}")

for delta in (1, -1):
    for use_gap, label in ((False, "duplicate only"), (True, "duplicate + gap")):
        hit = sum(1 for g, p in core if flags(g, p, delta, d0, use_gap))
        print(f"delta {delta:+d} [{label}]: {hit}/{len(core)} = {hit/len(core):.3f}")

print("\nfalse alarms when pass numbering is incomplete:")
for drop in (0.02, 0.05, 0.10):
    fa = []
    for rep in range(30):
        rng = random.Random(rep)
        kept = [r for r in core if rng.random() >= drop]
        d = build(kept)
        # false alarm = a pass in a stage that now has a gap, under the gap rule
        rows_in_gappy = sum(len(d[s]) for s in d if gap(d[s]))
        fa.append(rows_in_gappy / len(kept))
    print(f"  drop {int(drop*100):2d}%: clean rows flagged by the gap rule "
          f"{sum(fa)/len(fa):.3f}; by the duplicate rule 0.000")
