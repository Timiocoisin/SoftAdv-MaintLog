"""Export the flat-table baseline rows for the 234-pass core and test what it can and cannot do."""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = ROOT_DIR
OUT = ROOT / "outputs/softadv_protocol_20261106"
CORE_C = {25, 55, 85, 115, 145}
H = 4
GMAX = 169
CORE = sorted({g for c in CORE_C for g in range(c - H, c + H + 1) if 1 <= g <= GMAX})
THR = 10.0

def load_seedavg(pattern, seeds=(0, 1, 2)):
    bags = {}
    for s in seeds:
        path = next(OUT.glob(pattern.format(seed=s)))
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (int(row["stage"]), int(row["pass_id"]))
            bags.setdefault(key, []).append(float(row["prediction"]))
    return pd.DataFrame(
        [{"stage": k[0], "pass_id": k[1], "ghat": float(np.mean(v))} for k, v in bags.items()]
    )

feat = pd.read_csv(ROOT / "outputs/idea_validation_20261106/pass_features.csv")
te = feat[feat.stage.isin(CORE)].copy()
audit = load_seedavg("auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv")
core = audit[audit.stage.isin(CORE)].copy()
core = core.merge(te[["stage", "pass_id", "bright_40"]], on=["stage", "pass_id"])
assert len(core) == 234, len(core)

thr_gate = float(np.quantile(core["bright_40"], 0.10))
core["weak_gate"] = core["bright_40"] <= thr_gate
core["g_log"] = core["stage"].astype(float)          # correct-log board
core["gap"] = (core["ghat"] - core["g_log"]).abs()
core["alert"] = (core["gap"] > THR) | core["weak_gate"]

# action priority, as in the paper
a = np.full(len(core), "Accept", dtype=object)
a[core["alert"].to_numpy() & ~core["weak_gate"].to_numpy()] = "RecheckLog"
a[core["weak_gate"].to_numpy()] = "InspectGate"
a[core["alert"].to_numpy() & (core["gap"].to_numpy() <= THR) & ~core["weak_gate"].to_numpy()] = "Escalate"
core["action"] = a

rule = np.where(core["weak_gate"], "R3", np.where(core["gap"] > THR, "R2", "R1"))
core.loc[(core["action"] == "Escalate"), "rule"] = "R4"
core["rule"] = rule

flat = core.rename(columns={"pass_id": "pass_index"})[
    ["stage", "pass_index", "g_log", "ghat", "weak_gate", "gap", "alert", "action", "rule"]
].copy()
flat.insert(0, "campaign", "caseA")
flat["thr"] = THR
out = ROOT / "paper_AEI_v1/release/release_tables/flat_baseline_rows.csv"
flat.to_csv(out, index=False)

n = len(flat)
thr_only = int((flat["gap"] > THR).sum())
with_gate = int(flat["alert"].sum())
forced = with_gate - thr_only
weak = int(flat["weak_gate"].sum())
print(f"rows={n}")
print(f"threshold reviews = {thr_only} ({thr_only/n:.4f})")
print(f"with gate        = {with_gate} ({with_gate/n:.4f})")
print(f"forced by gate   = {forced} ({forced/n:.4f})")
print(f"weak gate        = {weak} ({weak/n:.4f})")
print("actions:", dict(flat["action"].value_counts()))
print("rules:  ", dict(flat["rule"].value_counts()))

# what a flat table does NOT do: it accepts a corrupted pair
corrupt = flat.copy()
i = 5
corrupt.loc[i, "g_log"] = corrupt.loc[i, "g_log"] + 1
dup = corrupt.duplicated(subset=["stage", "pass_index"]).any()
mono = (corrupt.sort_values(["stage", "pass_index"])["g_log"].diff() < 0).any()
print(f"\nflat table accepts the corrupted row: duplicate={dup}, monotone violation={mono}")
print(f"wrote {out}")
