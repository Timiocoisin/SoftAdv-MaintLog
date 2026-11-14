"""Validate a qualification card. Standard library only.

A card records, class by class, one of three outcomes. The validator enforces the
grammar that the paper claims, so an unmeasured row cannot be recorded as a pass:

  * every class the campaign declared must appear on the card, so a row cannot be
    dropped instead of measured;
  * a verdict is one of qualified, excluded or not_assessable, and "pass" is not one;
  * qualified and excluded must agree with the margin and the bar, so a verdict
    cannot contradict the measurement that produced it;
  * a not_assessable row must carry a reason, because the outcome asserts that the
    row could not be run, not that it was run and disliked.

Usage:  python check_card.py card.json [--bar 0.475]
Exit code 0 when the card is valid, 1 otherwise, with the reasons on stdout.
"""
import json
import sys

VERDICTS = {"qualified", "excluded", "not_assessable"}


def check(card, bar):
    """Return a list of error strings. Empty list means the card is valid."""
    errors = []
    if not isinstance(card, dict):
        return ["card is not an object"]

    classes = card.get("classes")
    if not isinstance(classes, list) or not classes:
        return ["card has no classes"]

    declared = card.get("campaign_classes")
    if not isinstance(declared, list) or not declared:
        errors.append("campaign_classes is missing, so a dropped row cannot be detected")
    else:
        seen = {row.get("class") for row in classes if isinstance(row, dict)}
        missing = [c for c in declared if c not in seen]
        if missing:
            errors.append(f"classes declared by the campaign but absent from the card: {missing}")

    for i, row in enumerate(classes):
        where = f"row {i}"
        if not isinstance(row, dict):
            errors.append(f"{where}: not an object")
            continue
        name = row.get("class", f"<unnamed {i}>")
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"{where} ({name}): verdict {verdict!r} is not one of {sorted(VERDICTS)}")
            continue

        if verdict == "not_assessable":
            if not row.get("reason"):
                errors.append(f"{where} ({name}): not_assessable without a reason")
            if row.get("det") is not None or row.get("margin") is not None:
                errors.append(f"{where} ({name}): not_assessable carries a measurement")
            continue

        margin = row.get("margin")
        if margin is None:
            errors.append(f"{where} ({name}): {verdict} without a margin")
            continue
        above = margin > bar
        if verdict == "qualified" and not above:
            errors.append(f"{where} ({name}): qualified at margin {margin} against bar {bar}")
        if verdict == "excluded" and above:
            errors.append(f"{where} ({name}): excluded at margin {margin} against bar {bar}")

    return errors


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    bar = 0.475
    if "--bar" in argv:
        bar = float(argv[argv.index("--bar") + 1])
    with open(argv[1], encoding="utf-8") as fh:
        card = json.load(fh)
    errors = check(card, bar)
    if errors:
        print("INVALID")
        for e in errors:
            print("  -", e)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
