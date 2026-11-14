"""End-to-end cases for the release pipeline, from a site file to a release record."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release import build  # noqa: E402

BASE = {
    "site": "Changshu unit 2, exterior grinding",
    "date": "2026-11-14",
    "operator": "Xian-ming Huang",
    "prevalence": {"value": 0.013, "source": "assumed"},
    "cost": {"c_fa_min": 1.57, "c_miss_min": 45.0, "source": "elicited"},
    "classes": [
        {"class": "large", "det": 0.987, "fa": 0.265, "n": 234, "source": "measured"},
        {"class": "half", "det": 1.000, "fa": 0.265, "n": 234, "source": "measured"},
        {"class": "mirror", "det": 0.897, "fa": 0.265, "n": 234, "source": "measured"},
        {"class": "digit", "det": 0.922, "fa": 0.265, "n": 234, "source": "measured"},
        {"class": "small", "det": 0.372, "fa": 0.265, "n": 234, "source": "measured"},
        {"class": "off-one", "det": 0.265, "fa": 0.265, "n": 234, "source": "measured"},
    ],
}


def case(name, site, expect):
    rec, errs = build(site)
    got = rec["release"] if rec else "ERROR"
    ok = got == expect
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got} (expected {expect})")
    if rec:
        print(f"        bar {rec['bar']['value']}, delegated {rec['delegated']}, "
              f"manual {len(rec['manual'])}, unmeasured {len(rec['not_assessable'])}")
    for e in errs:
        print("        error:", e)
    return ok


results = []

# 1. a complete site: four classes clear, two do not, nothing unmeasured
results.append(case("complete site", BASE, "RELEASED"))

# 2. a site that could not measure one class: the card is still valid, the release is partial
partial = json.loads(json.dumps(BASE))
partial["classes"][2] = {"class": "mirror", "reason": "no stage blocks to hold out"}
results.append(case("one class unmeasured", partial, "PARTIAL"))

# 3. no prevalence and no bar: the bar cannot be derived, so nothing is released
nobar = json.loads(json.dumps(BASE))
del nobar["prevalence"]
results.append(case("no prevalence, no bar", nobar, "NOT RELEASED"))

# 4. a site whose bar is supplied explicitly instead of derived
given = json.loads(json.dumps(BASE))
del given["prevalence"]
given["bar"] = 0.45
results.append(case("site supplies its own bar", given, "RELEASED"))

# 5. a detector that clears no class
weak = json.loads(json.dumps(BASE))
for c in weak["classes"]:
    c["det"] = c["fa"] + 0.01
results.append(case("no class clears the bar", weak, "NOT RELEASED"))

# 6. a site whose prevalence makes the bar impossible
lowp = json.loads(json.dumps(BASE))
lowp["prevalence"] = {"value": 0.001, "source": "assumed"}
results.append(case("bar above every attainable margin", lowp, "NOT RELEASED"))

print(f"\n{sum(results)}/{len(results)} cases behaved as specified")
sys.exit(0 if all(results) else 1)
