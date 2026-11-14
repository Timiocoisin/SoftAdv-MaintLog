# SoftAdv-MaintLog

### A handover-record specification and acceptance protocol for maintenance-log verification

[![License: MIT](https://img.shields.io/badge/License-MIT-2B5E83.svg)](LICENSE)

This repository accompanies a manuscript that treats maintenance-log verification as a
**specification problem**: what a handover record must contain for an acceptance
decision to be auditable, and what a verifier must demonstrate before it may advise on
one.

The laboratory detector included here is a demonstrator, not the contribution. When it
is placed under the protocol it **fails the injectability row** on off-by-one slips, and
the protocol blocks deployment. We report that failure as a result.

## What is in the repository

| Path | Contents |
|---|---|
| `schema/` | Decision-record schema (JSON Schema 2020-12), validator, self-test, cost model |
| `scripts/` | Analysis scripts: conformal calibration, block-wise coverage, counterfactual replay, shadow-run analysis, structured error injection |
| `release_tables/` | Protocol, operating-point and board tables referenced by the manuscript |
| `training/` | MetalField operator, Audit-OrdNet dual-stream reader, labels, metrics, helpers |
| `experiments/` | Stage-gap / proxy / MetalField runners (repo-relative paths) |
| `docs/` | Scope, non-claims, reproducibility notes |

Not included: raw laboratory frames, feature caches, trained checkpoints, manuscript
sources.

## Running the analysis scripts

Every script in `scripts/` reads the campaign data from a root directory that you set
with one environment variable, so nothing is tied to the machine the analysis was
originally run on:

```bash
export SOFTADV_ROOT=/path/to/WaterWallSensing     # Windows: $env:SOFTADV_ROOT="..."
python scripts/derive_threshold.py                # no data needed
python scripts/conventional_baseline.py           # needs outputs/idea_validation_20261106/pass_features.csv
```

The scripts expect this layout under `SOFTADV_ROOT`:

| Path | Needed by |
|---|---|
| `outputs/idea_validation_20261106/pass_features.csv` | record checks, class boards |
| `outputs/idea_validation_20261106/predictions.csv` | estimator comparison |
| `outputs/softadv_protocol_20261106/*_predictions.csv` | per-seed and seed-averaged boards |
| `datasets/<stage>/Image_<timestamp>.bmp` | capture order (only the filenames are read) |

Scripts that need this data exit with a clear message if `SOFTADV_ROOT` does not point
at it. The derivation scripts (`derive_threshold.py`, `delegation_sep.py`) run with no
data at all.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# the record validator is standard-library only
python schema/test_schema.py       # one record accepted, five violations rejected
python schema/costmodel.py         # break-even prevalence and per-campaign cost
python verify_release_tables.py
```

## The record, in one table

Nine entities — asset, campaign, stage, log entry, evidence, decision, action, override,
rationale — with five integrity constraints:

| Constraint | Check |
|---|---|
| C1 | accepted stage index is non-decreasing within a campaign |
| C2 | evidence is captured inside the pass window |
| C3 | gate quality is recorded, not inferred |
| C4 | only a supervisor may override, never the writer |
| C5 | a decision carries a rule version, at least one evidence reference and an operator |

`schema/decision-record-1.0.json` fixes the shape; `schema/validate.py` adds what JSON
Schema cannot express (cross-record monotonicity, role authority, time ordering).

## The protocol

Four pass/fail rows and five deployment gates. A failed row triggers a pause: the
verifier stops advising rather than degrading silently.

| Row | Pass condition |
|---|---|
| Injectability | `Det − FA ≥ 0.50` on the site's own error pattern |
| Budget controllability | false-review rate non-increasing in the threshold |
| Fail-closedness | `REVIEW ≥ 0.90` on an out-of-envelope probe |
| Task applicability | an ordered `g_log` exists |

## What the demonstrator does, reported honestly

| Quantity | Value |
|---|---|
| Large-offset detection at 26.5% review load | 0.987 |
| Off-by-one slip detection at the same load | **0.24** |
| Digit-transposition detection | 0.85 |
| Calibrated review threshold (conformal, α=0.265) | 10.2 |
| Block-wise coverage when one stage block is held out | **0.00–1.00** |
| Break-even wrong-log prevalence (c2 min, 45 min) | 1.25% |
| Cost per campaign | 8.25 h |

The off-by-one result and the block-wise coverage spread are the two findings that
motivate the protocol: a single global threshold does not transport across stage
blocks, and uniform small-offset injection overstates detection.

## Scope and non-claims

- One laboratory specimen, one plant; no multi-site validation.
- Detection is scored on injected log errors; the true error distribution of the plant is
  unknown.
- No blinded supervisor baseline; the shadow run reports behaviour under advice, not
  unaided performance.
- The demonstrator is not a production interlock and never writes the log.

See `docs/` and `DATA_NOTICE.md`.

## Citation

No peer-reviewed identifier is assigned yet. Cite the repository URL and a commit hash.

## License

Code under the MIT License (see `LICENSE`). Numeric tables are supplied for research
verification; see `DATA_NOTICE.md`.
