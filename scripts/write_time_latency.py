"""When is a shifted row caught: at the write, before the stage is left, or only
at the audit?

The record is written one pass at a time, in stage order. A shifted row can be
rejected by the write that enters it, by a later write in the same stage, or only
after the record is complete. The timing matters because a slip caught while the
crew is still on the panel can be re-entered, and one caught at audit needs the
supervisor to reopen a closed decision.

We measure latency in written rows, for both shift directions.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
from collections import defaultdict

PATH = str(ROOT_DIR / "outputs/idea_validation_20261106/pass_features.csv")

rows = []
with open(PATH, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append((int(r["stage"]), int(r["pass_id"])))
rows.sort()
N = len(rows)
last_stage = max(g for g, _ in rows)

def has_gap(ps):
    return bool(ps) and set(range(max(ps) + 1)) != set(ps)

def check(prefix):
    """Return the first violation visible in this prefix, or None."""
    seen = defaultdict(set)
    twice = False
    for g, p in prefix:
        if p in seen[g]:
            twice = True
        seen[g].add(p)
    if twice:
        return "duplicate"
    maxseen = max(g for g, _ in prefix)
    for s, ps in seen.items():
        if s < maxseen and has_gap(ps):
            return "gap"
    return None

def latency(i, delta):
    g, p = rows[i]
    gt = g + delta
    if not (1 <= gt <= last_stage):
        return None
    shifted = list(rows)
    shifted[i] = (gt, p)
    for k in range(i, N):
        v = check(shifted[: k + 1])
        if v is not None:
            return k - i, v
    return None

print(f"record: {N} passes over {len(set(g for g, _ in rows))} stages\n")
for delta in (1, -1):
    res = [latency(i, delta) for i in range(N)]
    hit = [(lat, v) for r in res if r for lat, v in [r]]
    zero = sum(1 for lat, _ in hit if lat == 0)
    near = sum(1 for lat, _ in hit if 1 <= lat <= 9)
    far = sum(1 for lat, _ in hit if lat > 9)
    kinds = defaultdict(int)
    for _, v in hit:
        kinds[v] += 1
    print(f"delta {delta:+d}: detected {len(hit)}/{N} = {len(hit)/N:.3f}")
    print(f"   rejected by the write itself : {zero}")
    print(f"   within 9 written rows        : {near}")
    print(f"   later than 9 rows            : {far}")
    print(f"   by rule: {dict(kinds)}")
    if hit:
        lats = sorted(lat for lat, _ in hit)
        print(f"   median latency {lats[len(lats)//2]}, max {lats[-1]}\n")
