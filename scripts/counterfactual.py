"""Counterfactual replay: how many decisions the record can re-derive at a new alpha."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import statistics as st
from collections import defaultdict

ALPHAS = {"0.10": 17.39, "0.26 (site)": 10.18, "0.30": 8.57}

preds = defaultdict(dict)
core = {}
for s in range(3):
    with open(f"aord_erm_s{s}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = (int(r["stage"]), int(r["pass_id"]))
            preds[k][s] = float(r["prediction"])
            core[k] = r.get("in_core45", "False").strip().lower() == "true"

rows = []
for k, d in preds.items():
    if not core[k]:
        continue
    g = k[0]
    rows.append({"g": g, "ghat": sum(d.values()) / len(d)})

print(f"records: {len(rows)}")
base = [abs(r["ghat"] - r["g"]) > ALPHAS["0.26 (site)"] for r in rows]
print(f"REVIEW at site q: {sum(base)} ({sum(base)/len(rows):.3f})")

for name, q in ALPHAS.items():
    dec = [abs(r["ghat"] - r["g"]) > q for r in rows]
    flips = sum(1 for a, b in zip(base, dec) if a != b)
    print(f"alpha={name:12s} q={q:6.2f} REVIEW={sum(dec):3d} ({sum(dec)/len(rows):.3f}) "
          f"flips vs site={flips:3d} ({flips/len(rows):.3f})")
