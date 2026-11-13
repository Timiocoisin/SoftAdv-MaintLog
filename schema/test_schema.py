"""Self-test for the decision-record schema and validator."""
import json
import copy
import sys

sys.path.insert(0, ".")
from validate import validate  # noqa: E402

with open("decision-record-1.0.json", encoding="utf-8") as fh:
    SCHEMA = json.load(fh)
print("schema parsed, top-level keys:", len(SCHEMA["properties"]))

GOOD = {
    "schema_version": "1.0",
    "record_id": "DR-000085",
    "asset": {"asset_id": "WW-A-01", "component_type": "water-wall panel"},
    "campaign": {"campaign_id": "CAMP-2025-11",
                 "window_start": "2025-11-03T08:00:00+08:00",
                 "window_end": "2025-11-16T18:00:00+08:00", "planned_stages": 169},
    "stage": 85,
    "pass": {"index": 2, "window_start": "2025-11-09T10:11:00+08:00",
             "window_end": "2025-11-09T10:18:00+08:00"},
    "log": {"value": 85, "writer_id": "W-07",
            "written_at": "2025-11-09T10:12:00+08:00", "sheet_id": "S-085"},
    "evidence": [{"evidence_id": "E-0851",
                  "frame_uri": "file:///frames/pass0085/f012.jpg",
                  "estimate": 97.0, "gate": "usable", "confidence": 0.81,
                  "captured_at": "2025-11-09T10:14:00+08:00"}],
    "decision": "REVIEW",
    "threshold": 10,
    "rationale": {"rule_id": "R2", "evidence_refs": ["E-0851"],
                  "margin": 2.0, "expected_cost": 2.0},
    "override": {"response": "change", "operator_id": "SUP-02",
                 "decided_at": "2025-11-09T10:41:00+08:00",
                 "reason": "sheet re-read against panel"},
    "provenance": {"rule_version": "v1.0", "operator_id": "SUP-02",
                   "recorded_at": "2025-11-09T10:41:05+08:00",
                   "prev_hash": "0" * 64},
}

cases = []
cases.append(("valid record", GOOD, None))

bad = copy.deepcopy(GOOD)
bad["evidence"][0]["gate"] = "unknown"
cases.append(("bad gate", bad, None))

bad = copy.deepcopy(GOOD)
bad["override"]["operator_id"] = "W-07"
cases.append(("writer overrides own log (C4)", bad, None))

bad = copy.deepcopy(GOOD)
bad["rationale"]["evidence_refs"] = []
cases.append(("no evidence refs (C5)", bad, None))

bad = copy.deepcopy(GOOD)
bad["stage"] = 70
bad["decision"] = "PASS"
bad["override"] = None
cases.append(("accepted stage moves back (C1)", bad, 85))

bad = copy.deepcopy(GOOD)
bad["evidence"][0]["captured_at"] = "2025-11-09T09:55:00+08:00"
cases.append(("evidence before log (C2)", bad, None))

for name, rec, prev in cases:
    errs = validate(rec, prev)
    status = "ACCEPTED" if not errs else "REJECTED: " + "; ".join(errs)
    print(f"{name:34s} -> {status}")
