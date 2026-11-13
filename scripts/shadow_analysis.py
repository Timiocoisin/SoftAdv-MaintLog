import csv
from collections import Counter, defaultdict

rows = list(csv.DictReader(open("shadow_passes.csv", encoding="utf-8")))
print("rows:", len(rows))
print("probe values:", Counter(r["probe"] for r in rows))
nat = [r for r in rows if r["probe"].strip() in ("0", "")]
pro = [r for r in rows if r["probe"].strip() not in ("0", "")]
print(f"natural {len(nat)}, probes {len(pro)}")

rev = [r for r in nat if r["d"].upper() == "REVIEW"]
print(f"\nnatural REVIEW: {len(rev)}/{len(nat)} = {len(rev)/len(nat):.3f}")
print("actions among REVIEW:", Counter(r["a"] for r in rev))
print("supervisor_action among REVIEW:", Counter(r["supervisor_action"] for r in rev))

print("\n=== REVIEW by action x supervisor_action ===")
tab = defaultdict(Counter)
for r in rev:
    tab[r["a"]][r["supervisor_action"]] += 1
for a in sorted(tab):
    c = tab[a]
    n = sum(c.values())
    print(f"{a}: n={n:2d} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))

print("\n=== change rate by action ===")
for a in sorted(tab):
    c = tab[a]
    n = sum(c.values())
    ch = c.get("change", 0)
    print(f"{a}: change {ch}/{n} = {ch/n:.3f}")

print("\n=== recheck ===")
rc = [r for r in rev if r["recheck_done"].strip() in ("1", "True", "true", "Y")]
print(f"recheck among REVIEW: {len(rc)}/{len(rev)} = {len(rc)/len(rev):.3f}")
mins = [float(r["recheck_min"]) for r in rc if r["recheck_min"].strip()]
if mins:
    mins.sort()
    print(f"recheck minutes: median {mins[len(mins)//2]:.1f}, range {min(mins)}-{max(mins)}")

print("\n=== probes ===")
for r in pro[:12]:
    print(r["pass_id"], r["d"], r["a"], r["supervisor_action"], r["notes"][:40])
