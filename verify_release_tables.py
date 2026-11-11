#!/usr/bin/env python3
"""Lightweight check that submission protocol tables are present and readable."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "release_tables"

REQUIRED = [
    "alert_operating_points.csv",
    "protocol_sensitivity_summary.csv",
    "brightness_drift_curve.csv",
    "proxy_split_summary.csv",
    "visual_heuristic_family.csv",
    "temporal_consistency.json",
    "odd_even_summary.json",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (TAB / name).is_file()]
    if missing:
        print("FAIL: missing tables:", ", ".join(missing))
        return 1

    # Smoke-read a few files
    with (TAB / "alert_operating_points.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("FAIL: alert_operating_points.csv is empty")
        return 1

    with (TAB / "temporal_consistency.json").open(encoding="utf-8") as f:
        json.load(f)

    print(f"PASS: {len(REQUIRED)} required release tables present and readable.")
    print("No model was trained; raw frames/checkpoints are outside this release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
