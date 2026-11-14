"""Measure the real capture order of the 169 stages from the frame timestamps.

Frames live in datasets/<stage>/Image_<yyyyMMddHHmmssSSS>.bmp. For each stage we take
the first and last frame timestamp, then ask whether the global capture order is the
stage order. If it is, the write-order assumption in the latency analysis is measured
rather than assumed.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import os
import re
from datetime import datetime

ROOT = str(ROOT_DIR / "datasets")
pat = re.compile(r"^Image_(\d{17})\.bmp$")

stages = []
for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d) or not name.isdigit():
        continue
    ts = []
    for f in os.listdir(d):
        m = pat.match(f)
        if m:
            ts.append(m.group(1))
    if not ts:
        continue
    ts.sort()
    stages.append((int(name), ts[0], ts[-1], len(ts)))

print(f"stage directories with frames: {len(stages)}")
print(f"total frames: {sum(s[3] for s in stages)}\n")

def parse(t):
    return datetime.strptime(t, "%Y%m%d%H%M%S%f")

# order by first capture time
by_first = sorted(stages, key=lambda s: s[1])
order = [s[0] for s in by_first]
inv = sum(1 for a, b in zip(order, order[1:]) if b < a)
print(f"stage index inversions when ordered by first frame time: {inv} of {len(order)-1}")
print(f"first stage in time: {order[0]}, last: {order[-1]}")
print(f"date span: {by_first[0][1][:8]} to {by_first[-1][2][:8]}\n")

# do stage intervals overlap in time?
overlaps = 0
for (g1, a1, b1, _), (g2, a2, b2, _) in zip(by_first, by_first[1:]):
    if parse(a2) < parse(b1):
        overlaps += 1
print(f"consecutive stage capture intervals that overlap: {overlaps} of {len(by_first)-1}")

# inactive gaps, which mark session boundaries
gaps = []
for (g1, a1, b1, _), (g2, a2, b2, _) in zip(by_first, by_first[1:]):
    dt = (parse(a2) - parse(b1)).total_seconds() / 60.0
    if dt > 60:
        gaps.append((g1, g2, round(dt / 60, 1)))
print(f"\ngaps over one hour between consecutive stages: {len(gaps)}")
for g1, g2, h in gaps[:10]:
    print(f"   after stage {g1} -> {g2}: {h} h")

print("\nfirst 8 stages in capture order:")
for g, a, b, n in by_first[:8]:
    print(f"   stage {g:3d}  {a} -> {b}  ({n} frames)")
