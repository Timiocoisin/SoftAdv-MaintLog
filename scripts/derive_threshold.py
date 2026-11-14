"""Derive the injectability margin from the site's cost ratio.

Deploy or qualify a class only if the expected cost of using the verifier is
below the cost of not using it:

    p * N * (1 - Det) * c_miss + N * FA * c_FA  <  p * N * c_miss

which rearranges to

    Det > FA * c_FA / (p * c_miss)

and, writing the Youden margin tau = Det - FA,

    tau > FA * (c_FA / (p * c_miss) - 1).
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()

FA = 0.265
C_FA = 1.57          # blended action cost, minutes (Table 12)
C_MISS = (30.0, 45.0, 60.0)   # elicited range, minutes
P = (0.005, 0.0125, 0.025, 0.05, 0.10, 0.20)

print("required Det = FA * c_FA / (p * c_miss), with FA = 0.265\n")
header = f"{'p':>7} " + " ".join(f"c_miss={c:>4.0f}" for c in C_MISS) + "   tau at c_miss=45"
print(header)
for p in P:
    row = f"{p*100:6.2f}% "
    for c in C_MISS:
        need = FA * C_FA / (p * c)
        row += f"{need:>12.3f} "
    tau = FA * C_FA / (p * 45.0) - FA
    print(row + f"   {tau:>8.3f}")

print("\nthe margin at each prevalence, c_miss = 45 min")
for p in (0.010, 0.0118, 0.0125, 0.015):
    tau = FA * C_FA / (p * 45.0) - FA
    print(f"  p = {p*100:5.2f}%  ->  tau = {tau:6.3f}")

print("\nclasses at p = 1.25%, c_miss = 45")
need = FA * C_FA / (0.0125 * 45.0)
for name, det in (("uniform large", 0.987), ("digit transposition", 0.922),
                  ("uniform small", 0.372), ("off-by-three", 0.291),
                  ("off-by-two", 0.274), ("off-by-one", 0.265)):
    verdict = "qualified" if det > need else "excluded"
    print(f"  {name:20s} Det={det:.3f}  need>{need:.3f}  {verdict}")
