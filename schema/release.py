"""Turn a site's measurements into a hold-point release record.

The paper's protocol says a class may be delegated only if it clears a cost-derived
bar, and that a row which cannot be run must be recorded as not assessable rather than
as a failed one. This script carries that from prose to a document a quality plan can
file:

    site.json  ->  bar  ->  per-class verdicts  ->  card  ->  validator  ->  release.json

Release outcomes are three, matching the protocol's grammar:

    RELEASED      every declared class was measured, the card is valid, at least one
                  class is qualified
    PARTIAL       the card is valid and at least one class is qualified, but at least
                  one declared class could not be measured
    NOT RELEASED  no class qualifies, the card is invalid, or the bar cannot be derived
                  because the prevalence or the miss cost is missing

Exit codes 0 / 2 / 1 for RELEASED / PARTIAL / NOT RELEASED, so a wrapper can branch.

Standard library only. Usage:

    python release.py site.json --out release.json [--bar 0.475]
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_card import check  # noqa: E402

REQUIRED_KEYS = ("site", "classes")


def derive_bar(prevalence, c_fa, c_miss, fa_reference):
    """Eq. (14) of the manuscript: tau = FA * (cFA / (p * cMiss) - 1)."""
    if None in (prevalence, c_fa, c_miss, fa_reference):
        return None, "prevalence or cost missing, so the bar cannot be derived"
    if not (0 < prevalence <= 1):
        return None, f"prevalence {prevalence} is not a fraction in (0, 1]"
    if c_miss <= 0 or c_fa <= 0:
        return None, "costs must be positive"
    tau = fa_reference * (c_fa / (prevalence * c_miss) - 1.0)
    return tau, None


def build(site, bar_override=None):
    errors = []
    for k in REQUIRED_KEYS:
        if k not in site:
            errors.append(f"site.json is missing {k!r}")
    if errors:
        return None, errors

    classes = site["classes"]
    if not isinstance(classes, list) or not classes:
        return None, ["site.json declares no classes"]

    # the false-review rate is a property of the verifier, not of a class; take the
    # largest reported value so a class is never qualified on another class's FA
    measured = [c for c in classes if c.get("det") is not None and c.get("fa") is not None]
    fa_ref = max((c["fa"] for c in measured), default=None)

    prevalence = (site.get("prevalence") or {}).get("value")
    cost = site.get("cost") or {}
    c_fa, c_miss = cost.get("c_fa_min"), cost.get("c_miss_min")

    if bar_override is not None:
        bar, bar_error, derived = float(bar_override), None, False
    else:
        bar, bar_error = derive_bar(prevalence, c_fa, c_miss, fa_ref)
        derived = bar is not None
        if bar is None and site.get("bar") is not None:
            bar, bar_error, derived = float(site["bar"]), None, False

    rows = []
    for c in classes:
        name = c.get("class", "?")
        det, fa = c.get("det"), c.get("fa")
        if det is None or fa is None:
            rows.append({
                "class": name,
                "verdict": "not_assessable",
                "reason": c.get("reason", "no measurement supplied for this class"),
            })
            continue
        margin = det - fa
        if bar is None:
            rows.append({"class": name, "verdict": "not_assessable",
                         "reason": "no bar available, so the margin cannot be judged",
                         "det": det, "margin": margin})
            continue
        rows.append({
            "class": name,
            "verdict": "qualified" if margin > bar else "excluded",
            "det": det, "margin": margin,
        })

    card = {"campaign_classes": [c.get("class") for c in classes], "classes": rows}
    card_errors = check(card, bar if bar is not None else 0.0)

    qualified = [r["class"] for r in rows if r["verdict"] == "qualified"]
    manual = [r["class"] for r in rows if r["verdict"] == "excluded"]
    unmeasured = [r["class"] for r in rows if r["verdict"] == "not_assessable"]

    if bar is None:
        outcome, reason = "NOT RELEASED", bar_error
    elif card_errors:
        outcome, reason = "NOT RELEASED", "; ".join(card_errors)
    elif not qualified:
        outcome, reason = "NOT RELEASED", "no class clears the bar"
    elif unmeasured:
        outcome, reason = "PARTIAL", "one or more declared classes could not be measured"
    else:
        outcome, reason = "RELEASED", "every declared class was measured and at least one qualifies"

    record = {
        "release": outcome,
        "reason": reason,
        "site": site.get("site"),
        "date": site.get("date") or datetime.now(timezone.utc).date().isoformat(),
        "bar": {
            "value": None if bar is None else round(bar, 4),
            "derived": derived,
            "formula": "tau = FA * (cFA / (p * cMiss) - 1)",
            "prevalence": prevalence,
            "prevalence_source": (site.get("prevalence") or {}).get("source"),
            "c_fa_min": c_fa,
            "c_miss_min": c_miss,
            "cost_source": cost.get("source"),
            "fa_reference": fa_ref,
        },
        "delegated": qualified,
        "manual": manual,
        "not_assessable": unmeasured,
        "card": card,
        "validation": {"valid": not card_errors, "errors": card_errors},
        "operator": site.get("operator"),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["hash"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return record, []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site")
    ap.add_argument("--out", default="-")
    ap.add_argument("--bar", type=float, default=None)
    a = ap.parse_args(argv)

    with open(a.site, encoding="utf-8") as fh:
        site = json.load(fh)
    record, errors = build(site, a.bar)
    if record is None:
        print("NOT RELEASED")
        for e in errors:
            print("  -", e)
        return 1

    text = json.dumps(record, indent=2, sort_keys=True)
    if a.out == "-":
        print(text)
    else:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"{record['release']}: {record['reason']}  ->  {a.out}")

    return {"RELEASED": 0, "PARTIAL": 2}.get(record["release"], 1)


if __name__ == "__main__":
    sys.exit(main())
