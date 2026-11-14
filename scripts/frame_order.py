"""Do the archived frame timestamps support the stage-by-stage writing order?

Only the release subsample carries capture timestamps. If the stage index rises with
capture time in every archived frame, the sorted order used in the latency analysis is
supported by observation rather than assumed, at least on the frames we still have.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import os
import re

D = str(ROOT_DIR / "paper_AEI_v1/release/release_frames_subsample")
pat = re.compile(r"_g(\d+)_Image_(\d{14,17})")

rows = []
for name in sorted(os.listdir(D)):
    m = pat.search(name)
    if not m:
        continue
    stage = int(m.group(1))
    ts = m.group(2)
    rows.append((ts, stage, name))

rows.sort()
print(f"{len(rows)} timestamped frames\n")
print(f"{'timestamp':22s} {'stage':>5}  file")
for ts, stage, name in rows:
    print(f"{ts:22s} {stage:5d}  {name[:60]}")

stages = [s for _, s, _ in rows]
drops = sum(1 for a, b in zip(stages, stages[1:]) if b < a)
print(f"\nstage index decreases between consecutive capture times: {drops} times")
print(f"stage range {min(stages)} to {max(stages)}")
first, last = rows[0][0][:8], rows[-1][0][:8]
print(f"dates span {first} to {last}")
