"""Is delegation over subsets a real problem, or does it collapse to the card?

The verifier emits REVIEW on a fraction FA of passes whatever class is being sought,
so the review load is shared across classes: running it at all costs N*FA*c_FA, and
adding a class to its scope costs no extra reviews. If that holds, the optimal
delegated set is exactly the set of classes that pass their own bar, and there is no
subset search to do.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
N = 877
CMISS = 45.0
CFA = 1.57
FA = 0.265
TAU = 0.475

CLASSES = [("large", 0.987), ("transposition", 0.922), ("small", 0.372),
           ("off-by-three", 0.291), ("off-by-two", 0.274), ("off-by-one", 0.265)]

def total_cost(delegated, p_total, shares):
    """Shared review load: one review serves every delegated class at once."""
    run = N * FA * CFA if delegated else 0.0
    miss = 0.0
    for (name, det), w in zip(CLASSES, shares):
        p_c = p_total * w
        if name in delegated:
            miss += p_c * N * (1 - det) * CMISS
        else:
            miss += p_c * N * CMISS
    return run + miss

shares = [1 / len(CLASSES)] * len(CLASSES)
print(f"{'p (total)':>10} {'manual':>10} {'all feasible':>13} {'verifier worth it?':>19}")
for p in (0.005, 0.0125, 0.02, 0.03, 0.05, 0.10):
    feasible = {n for n, d in CLASSES if d - FA >= TAU}
    c_man = total_cost(set(), p, shares)
    c_all = total_cost(feasible, p, shares)
    print(f"{p*100:9.2f}% {c_man:10.0f} {c_all:13.0f} {'yes' if c_all < c_man else 'no':>19}")

print("\nincremental value of adding one class to a running verifier")
for name, det in CLASSES:
    p_c = 0.0125 / len(CLASSES)
    print(f"  {name:15s} Det={det:.3f}  dC = {-p_c * N * det * CMISS:+8.1f} min")

print("\nso: every class with Det>0 is worth adding once the verifier runs,")
print("and the feasible set is exactly the card. No subset search.")
