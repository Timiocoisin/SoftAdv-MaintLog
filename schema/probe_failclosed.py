"""Out-of-envelope probe for the fail-closedness row.

The question the protocol row asks is whether the framework refuses to emit
acceptance evidence when the evidence is unusable. We probe the record layer,
where acceptance is committed: construct decision records whose evidence is
degraded or absent, and check that every one is rejected.
"""
import copy
import json
import sys

sys.path.insert(0, ".")
from validate import validate  # noqa: E402

with open("decision-record-1.0.json", encoding="utf-8") as fh:
    SCHEMA = json.load(fh)

# Case A vignette, aligned with the manuscript (g_log 85, g_hat 97, thr 10)
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

assert not validate(GOOD), "the reference record must be accepted"

probes = []

# 1. no evidence at all
p = copy.deepcopy(GOOD); p["evidence"] = []
probes.append(("evidence absent (empty array)", p))

# 2. gate absent, decision PASS (a silent acceptance on unusable evidence)
p = copy.deepcopy(GOOD); p["evidence"][0]["gate"] = "absent"; p["decision"] = "PASS"
p["rationale"]["rule_id"] = "R1"; p["override"] = None
probes.append(("gate absent but decision PASS", p))

# 3. gate weak, decision PASS
p = copy.deepcopy(GOOD); p["evidence"][0]["gate"] = "weak"; p["decision"] = "PASS"
p["rationale"]["rule_id"] = "R1"; p["override"] = None
probes.append(("gate weak but decision PASS", p))

# 4. estimate outside the campaign range (stage beyond G)
p = copy.deepcopy(GOOD); p["evidence"][0]["estimate"] = 420.0
probes.append(("estimate outside the campaign (420)", p))

# 5. evidence captured outside the pass window (late frame)
p = copy.deepcopy(GOOD); p["evidence"][0]["captured_at"] = "2025-11-09T23:59:00+08:00"
probes.append(("evidence captured outside the pass window", p))

# 6. no operator identity on an accepted record
p = copy.deepcopy(GOOD); p["provenance"]["operator_id"] = ""
probes.append(("accepted record with no operator", p))

# 7. confidence below the support floor
p = copy.deepcopy(GOOD); p["evidence"][0]["confidence"] = 0.0
p["decision"] = "PASS"; p["rationale"]["rule_id"] = "R1"; p["override"] = None
probes.append(("zero-confidence evidence, decision PASS", p))

rejected = 0
for name, rec in probes:
    errs = validate(rec, previous_stage=85)
    ok = bool(errs)
    rejected += ok
    print(f"{'REJECTED' if ok else 'ACCEPTED':9s} {name:42s} {errs[0] if errs else ''}")

print(f"\nfail-closed rate: {rejected}/{len(probes)} = {rejected/len(probes):.3f}")
