"""Per-block recalibration: does the campaign survive when theta is block-specific?"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv, random
from collections import defaultdict
import statistics as st

ALPHA = 0.265
CENTRES = [25, 55, 85, 115, 145]
H = 4

def load():
    preds, core = defaultdict(dict), {}
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
        blk = next((i for i, c in enumerate(CENTRES) if abs(g - c) <= H), None)
        rows.append({"stage": g, "pass": k[1], "g": g, "ghat": sum(d.values()) / len(d), "block": blk})
    return [r for r in rows if r["block"] is not None]

rows = load()
print("passes:", len(rows), "per block:", {b: sum(1 for r in rows if r["block"] == b) for b in range(5)})

def q_of(vals, alpha):
    v = sorted(vals)
    return v[min(len(v) - 1, int((len(v) + 1) * (1 - alpha)))]

def det_at(q, test, lo, hi, rng):
    hit = 0
    for r in test:
        d = rng.randint(lo, hi)
        claim = r["g"] + d
        hit += abs(r["ghat"] - claim) > q
    return hit / len(test)

rng = random.Random(7)
print(f"\n{'block':>5} {'n_cal':>5} {'n_test':>6} {'q_b':>6} {'FA_b':>6} {'Det_1':>6} {'Det_small':>9} {'Det_large':>9}")
qbs, fas, dls = [], [], []
for b in range(5):
    sub = [r for r in rows if r["block"] == b]
    sub.sort(key=lambda r: r["stage"])
    cal = [r for i, r in enumerate(sub) if i % 2 == 0]
    test = [r for i, r in enumerate(sub) if i % 2 == 1]
    q = q_of([abs(r["ghat"] - r["g"]) for r in cal], ALPHA)
    fa = sum(1 for r in test if abs(r["ghat"] - r["g"]) > q) / len(test)
    d1 = det_at(q, test, 1, 1, rng)
    ds = det_at(q, test, 3, 8, rng)
    dl = det_at(q, test, 15, 40, rng)
    qbs.append(q); fas.append(fa); dls.append(dl)
    print(f"{b:5d} {len(cal):5d} {len(test):6d} {q:6.2f} {fa:6.3f} {d1:6.3f} {ds:9.3f} {dl:9.3f}")

print(f"\nper-block q: min {min(qbs):.2f} median {st.median(qbs):.2f} max {max(qbs):.2f}  ratio {max(qbs)/min(qbs):.1f}x")
print(f"per-block FA: min {min(fas):.3f} median {st.median(fas):.3f} max {max(fas):.3f}")
print(f"pooled FA if every block used its own q: {sum(fas)/len(fas):.3f}")

# global threshold for comparison
gq = 10.2
print("\n--- same board at the global threshold 10.2 ---")
for b in range(5):
    sub = [r for r in rows if r["block"] == b]
    test = [r for i, r in enumerate(sorted(sub, key=lambda r: r["stage"])) if i % 2 == 1]
    fa = sum(1 for r in test if abs(r["ghat"] - r["g"]) > gq) / len(test)
    dl = det_at(gq, test, 15, 40, rng)
    print(f"block {b}: FA {fa:.3f}  Det_large {dl:.3f}")
