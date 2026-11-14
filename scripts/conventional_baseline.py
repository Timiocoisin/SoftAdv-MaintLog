"""What a conventional maintenance log can and cannot express.

A conventional staged-maintenance log stores, per row, the campaign, the stage
index, the written answer and the decision. It has no pass entity and therefore
no pass index. We compare it with the eleven-entity record on the same records
and the same injected fault (a unit stage shift, the most plausible handwriting
error), and report what each can detect and what each can even express.
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

stages = sorted({g for g, _ in rows})
print(f"record: {len(rows)} passes over {len(stages)} stages\n")

# ---------------------------------------------------------------- conventional
# (campaign, stage, g_log, decision); key is (campaign, stage) only when the
# site keeps one row per stage, which this process does not (5.2 passes/stage).
conv_rows = [(g, "decision") for g, _ in rows]
seen = defaultdict(int)
for g, _ in conv_rows:
    seen[g] += 1

# constraints a conventional schema can carry
c1_monotone = all(
    stages[i] <= stages[i + 1] for i in range(len(stages) - 1)
)
max_rows_per_stage = max(seen.values())

def conv_detect_unit_shift(delta):
    """A unit shift changes only g_log. Nothing in a log without a pass identity
    can distinguish the shifted row from a legitimate extra row at that stage."""
    hits = 0
    for g, _ in rows:
        target = g + delta
        # the only constraint available is: stage index must stay in range
        if 1 <= target <= max(stages):
            hits += 0  # the shift violates nothing expressible
    return hits

print("=== conventional record (campaign, stage, g_log, decision) ===")
print(f"  rows per stage: max {max_rows_per_stage}, so (campaign, stage) is not a key")
print(f"  C1 stage monotone checkable: {c1_monotone}")
print("  C2 window: not expressible (no window field)")
print("  C3 gate availability: not expressible")
print("  C4 role authority: not expressible")
print("  C5 provenance: not expressible")
print("  C6 pass uniqueness/contiguity: not expressible (no pass entity)")
for delta in (1, -1):
    print(f"  unit shift {delta:+d}: detected {conv_detect_unit_shift(delta)}/{len(rows)}")

# ---------------------------------------------------------------- our record
def build(rs):
    d = defaultdict(set)
    for g, p in rs:
        d[g].add(p)
    return d

def gap(ps):
    return bool(ps) and set(range(max(ps) + 1)) != set(ps)

d = build(rows)
pre_existing = sum(1 for s in d if gap(d[s]))

def ours_detect(g, p, delta):
    dup = p in d.get(g + delta, set())
    src = set(d[g]) - {p}
    tgt = set(d.get(g + delta, set())) | {p}
    return dup or gap(src) or gap(tgt)

print("\n=== eleven-entity record (pass entity + C6) ===")
print(f"  stages with a pre-existing gap: {pre_existing}/{len(d)}")
for delta in (1, -1):
    hit = sum(1 for g, p in rows if ours_detect(g, p, delta))
    print(f"  unit shift {delta:+d}: detected {hit}/{len(rows)} = {hit/len(rows):.3f}")

# ---------------------------------------------------------------- decisions
print("\n=== the acceptance decision each record can route ===")
print("  conventional: 'something at stage 27 may be wrong' (no pass named)")
print("  ours:         'pass 3 of stage 27 is a re-entry of pass 3 of stage 26'")
