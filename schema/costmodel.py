"""Cost model: current practice vs SoftAdv vs baselines, per campaign.

All inputs are the paper's existing numbers; the wrong-log prevalence p is
unknown and is therefore reported as a break-even, not assumed.
"""
N = 877            # passes per campaign
FA_S, DET_S = 0.265, 0.987      # SoftAdv
FA_R, DET_R = 0.265, 0.265      # random re-check, matched budget
FA_B, DET_B = 0.662, 0.795      # bright-ratio estimator


def cost(p, fa, det, c_fa, c_miss):
    return (1 - p) * N * fa * c_fa + p * N * (1 - det) * c_miss


def current(p, c_miss):
    return p * N * c_miss


def breakeven(fa, det, c_fa, c_miss):
    num = fa * c_fa
    return num / (num + det * c_miss)


print("=== break-even prevalence (%) ===")
print("c_fa\\c_miss", *[f"{m:>8d}" for m in (30, 45, 60)])
for c_fa in (1, 2, 3):
    row = []
    for c_miss in (30, 45, 60):
        row.append(f"{100*breakeven(FA_S, DET_S, c_fa, c_miss):8.2f}")
    print(f"{c_fa:>11d}", *row)

print("\n=== expected cost per campaign (min), c_fa=2, c_miss=45 ===")
print(f"{'p':>6} {'current':>8} {'softadv':>8} {'random':>8} {'bright':>8} {'save%':>7}")
for p in (0.005, 0.01, 0.02, 0.05, 0.10):
    c0 = current(p, 45)
    cs = cost(p, FA_S, DET_S, 2, 45)
    cr = cost(p, FA_R, DET_R, 2, 45)
    cb = cost(p, FA_B, DET_B, 2, 45)
    save = 100 * (c0 - cs) / c0
    print(f"{100*p:5.1f}% {c0:8.0f} {cs:8.0f} {cr:8.0f} {cb:8.0f} {save:6.0f}%")

print("\n=== passes needed to break even (p*, c_fa=2, c_miss=45) ===")
ps = breakeven(FA_S, DET_S, 2, 45)
print(f"p* = {100*ps:.2f}%  -> {ps*N:.1f} wrong logs per campaign")

print("\n=== sensitivity: saving vs current at p=2% ===")
for c_fa in (1, 2, 3):
    for c_miss in (30, 45, 60):
        c0 = current(0.02, c_miss)
        cs = cost(0.02, FA_S, DET_S, c_fa, c_miss)
        print(f"c_fa={c_fa} c_miss={c_miss}: current={c0:.0f} softadv={cs:.0f} save={c0-cs:.0f} min")
