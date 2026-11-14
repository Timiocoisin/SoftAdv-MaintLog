"""Five cases for the card validator. Every case states what it must provoke."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_card import check  # noqa: E402

BAR = 0.475
DECLARED = ["large", "half", "mirror", "digit", "small", "off-three", "off-two", "off-one"]

def card(rows):
    return {"campaign_classes": DECLARED, "classes": rows}

def row(name, verdict, margin=None, det=None, reason=None):
    r = {"class": name, "verdict": verdict}
    if margin is not None:
        r["margin"] = margin
    if det is not None:
        r["det"] = det
    if reason is not None:
        r["reason"] = reason
    return r

CASES = []

# 1. a card that measures everything and refuses two rows
full = [
    row("large", "qualified", 0.722, 0.987),
    row("half", "qualified", 0.735, 1.000),
    row("mirror", "qualified", 0.632, 0.897),
    row("digit", "qualified", 0.657, 0.922),
    row("small", "excluded", 0.107, 0.372),
    row("off-three", "excluded", 0.026, 0.291),
    row("off-two", "excluded", 0.009, 0.274),
    row("off-one", "excluded", 0.000, 0.265),
]
CASES.append(("complete card, two verdicts per class", card(full), True))

# 2. an unmeasured row recorded as a pass
passy = [dict(r) for r in full]
passy[2] = row("mirror", "pass")
CASES.append(("unmeasured row recorded as pass", card(passy), False))

# 3. a verdict that contradicts its own margin
liar = [dict(r) for r in full]
liar[4] = row("small", "qualified", 0.107, 0.372)
CASES.append(("qualified verdict below the bar", card(liar), False))

# 4. not_assessable without a reason
mute = [dict(r) for r in full]
mute[2] = row("mirror", "not_assessable")
CASES.append(("not_assessable with no reason", card(mute), False))

# 5. a class the campaign declared, silently dropped from the card
short = [r for r in full if r["class"] != "mirror"]
CASES.append(("declared class omitted from the card", card(short), False))

# 6. a legitimate not_assessable row: transfer card where blocks cannot be held out
transfer = [
    row("large", "not_assessable", reason="no stage blocks to hold out"),
    row("half", "not_assessable", reason="no out-of-envelope probe"),
]
CASES.append(("transfer card with two unmeasured rows",
              {"campaign_classes": ["large", "half"], "classes": transfer}, True))

failures = 0
for name, c, expected in CASES:
    errs = check(c, BAR)
    got = len(errs) == 0
    ok = got == expected
    if not ok:
        failures += 1
    print(f"{'PASS' if ok else 'FAIL'}  expected {'valid' if expected else 'invalid'}: {name}")
    for e in errs:
        print("        ", e)

print(f"\n{len(CASES) - failures}/{len(CASES)} cases behaved as specified")
sys.exit(1 if failures else 0)
