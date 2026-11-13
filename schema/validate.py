"""Validator for the handover decision record (schema v1.0).

Checks the JSON Schema constraints that matter for acceptance and the five
integrity constraints C1-C5 of the information model.  Standard library only,
so it runs wherever the maintenance system runs.
"""
import json
import re
from datetime import datetime

DECISIONS = {"PASS", "REVIEW"}
ACTIONS = {"R1", "R2", "R3", "R4", "R5"}
GATES = {"usable", "weak", "absent"}
RESPONSES = {"follow", "change", "dismiss"}
REQUIRED = ["schema_version", "record_id", "asset", "campaign", "stage",
            "log", "evidence", "decision", "rationale", "provenance"]


def _ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def check_schema(rec):
    errors = [f"missing field {k}" for k in REQUIRED if k not in rec]
    if rec.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if not re.fullmatch(r"DR-[0-9]{6}", rec.get("record_id", "")):
        errors.append("record_id must match DR-000000")
    if not isinstance(rec.get("stage"), int) or rec["stage"] < 0:
        errors.append("stage must be a non-negative integer")
    if rec.get("decision") not in DECISIONS:
        errors.append("decision must be PASS or REVIEW")
    log = rec.get("log", {})
    if not isinstance(log.get("value"), int) or log.get("value", -1) < 0:
        errors.append("log.value must be a non-negative integer")
    for key in ("writer_id", "written_at", "sheet_id"):
        if not log.get(key):
            errors.append(f"log.{key} is required")
    ev = rec.get("evidence") or []
    if not ev:
        errors.append("evidence must contain at least one item")
    for item in ev:
        if item.get("gate") not in GATES:
            errors.append("evidence.gate must be usable, weak or absent")
        conf = item.get("confidence")
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            errors.append("evidence.confidence must lie in [0, 1]")
    rat = rec.get("rationale", {})
    if rat.get("rule_id") not in ACTIONS:
        errors.append("rationale.rule_id must be R1-R5")
    if not rat.get("evidence_refs"):
        errors.append("rationale must cite at least one evidence item")
    prov = rec.get("provenance", {})
    if not re.fullmatch(r"v[0-9]+\.[0-9]+", prov.get("rule_version", "")):
        errors.append("provenance.rule_version must look like v1.0")
    if not prov.get("operator_id"):
        errors.append("provenance.operator_id is required")
    return errors


def check_constraints(rec, previous_stage=None):
    """C1-C5 of the information model, applied to one record."""
    errors = []
    log, ev, rat = rec["log"], rec["evidence"], rec["rationale"]

    # C1 stage monotonicity against the previous accepted record
    if previous_stage is not None and rec["decision"] == "PASS":
        if rec["stage"] < previous_stage:
            errors.append("C1 violated: accepted stage moves backwards")

    # C2 evidence captured inside the pass window (two-sided when the window is present)
    win = rec.get("pass") or {}
    for item in ev:
        cap = _ts(item["captured_at"])
        if "window_start" in win and cap < _ts(win["window_start"]):
            errors.append("C2 violated: evidence falls before the pass window")
        if "window_end" in win and cap > _ts(win["window_end"]):
            errors.append("C2 violated: evidence falls after the pass window")
        if "window_start" not in win and cap < _ts(log["written_at"]):
            errors.append("C2 violated: evidence predates the log entry")

    # C3 gate availability must be recorded, not inferred, and it governs the decision
    if not all("gate" in item for item in ev):
        errors.append("C3 violated: gate quality not recorded")
    if rec["decision"] == "PASS":
        if any(item.get("gate") != "usable" for item in ev):
            errors.append("C3 violated: PASS on evidence whose gate is not usable")
        if any(not float(item.get("confidence", 0)) > 0.0 for item in ev):
            errors.append("C3 violated: PASS on evidence with no reported support")
    planned = rec.get("campaign", {}).get("planned_stages")
    if isinstance(planned, int) and planned > 0:
        for item in ev:
            est = item.get("estimate")
            if isinstance(est, (int, float)) and not 0 <= float(est) <= planned:
                errors.append(
                    f"C3 violated: estimate {est} outside the campaign's {planned} stages")

    # C4 only the supervisor may override
    over = rec.get("override")
    if isinstance(over, dict):
        if over.get("response") not in RESPONSES:
            errors.append("C4 violated: unknown override response")
        if over.get("operator_id") == log["writer_id"]:
            errors.append("C4 violated: writer cannot override own entry")

    # C5 provenance completeness
    if not (prov_complete(rec)):
        errors.append("C5 violated: decision lacks rule, evidence or operator")
    return errors


def prov_complete(rec):
    rat, prov = rec["rationale"], rec["provenance"]
    return bool(rat.get("rule_id") and rat.get("evidence_refs")
                and prov.get("rule_version") and prov.get("operator_id"))


def validate(rec, previous_stage=None):
    errors = check_schema(rec) + check_constraints(rec, previous_stage)
    return errors
