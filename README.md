# SoftAdv-MaintLog

### Soft-advisory consistency checking of maintenance-log stages under slow appearance change

[![License: MIT](https://img.shields.io/badge/License-MIT-2B5E83.svg)](LICENSE)

**Laboratory research code for a human-in-the-loop soft-advisory loop:** recover a camera stage estimate \(\hat g\), compare it with a handwritten maintenance-log stage \(g\), expose an inspectable MetalField bright-metal gate, and raise site-tunable OK/ALERT exceptions **without** interlocking grinders.

This repository replaces the older [PASS-Ord](https://github.com/Timiocoisin/PASS-Ord) CNN/RF progress stack. The transferable question is soft-advisory **log–camera consistency** under near-duplicate stages; exterior boiler water-wall grinding is the concrete laboratory case (one fixed-pose specimen). **Not claimed:** multi-wall field deployment or a new backbone SOTA.

## What is included

| Path | Contents |
|------|----------|
| `training/ordprog_net.py` | MetalField operator + Audit-OrdNet dual-stream reader (ResNet-18 capacity-matched) |
| `training/*.py` | Labels, metrics, frame/feature helpers used by the boards |
| `experiments/` | Hard stage-gap / proxy / MetalField experiment runners (paths rewritten to be repo-relative) |
| `release_tables/` | Protocol and operating-point CSVs/JSON mirrored for ISWA submission assessment |
| `docs/` | Scope, non-claims, and reproducibility notes |

## What is **not** included

- Raw laboratory frames, feature caches, or trained checkpoints
- Manuscript sources / PDFs / cover letters
- Historical PASS-Ord CNN–RF nested-protocol code and retained prediction dumps
- Large exploratory `run_*.py` one-offs not needed for the soft-advisory boards

## Soft-advisory loop (conceptual)

```text
frame ──► MetalField gate (inspectable) ──► reader ──► ĝ
handwritten log g ─────────────────────────────────────┘
                         │
                         ▼
              |ĝ − g| > thr  or  weak gate  →  ALERT (re-check)
              else                           →  OK (monitor)
```

Default laboratory operating points treat alert rates as **tunable soft-advisory thresholds**, not as a production interlock.

## Quick start (tables + package import)

```bash
git clone https://github.com/Timiocoisin/SoftAdv-MaintLog.git
cd SoftAdv-MaintLog
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python verify_release_tables.py
python -c "from training.ordprog_net import VARIANTS; print(sorted(VARIANTS)[:5])"
```

Full image-based training/evaluation requires the laboratory frames (available from the corresponding author on reasonable request; public release planned upon manuscript acceptance). Experiment scripts under `experiments/` expect that local data layout and are research runners, not a one-click production pipeline.

## Protocol sketch

- One laboratory specimen trajectory: **169** logged stages, **17,511** frames, **877** passes.
- Primary ranking uses a **hard stage-gap hold** (near-duplicate neighbours excluded) rather than a soft image split.
- Primary matched claim in the manuscript: MetalField-backed readout vs plain RGB classification under that hold; Audit-OrdNet is the dual-stream carrier of the gate, not an architecture novelty claim.

See `release_tables/README.md` for which CSV/JSON files support which boards.

## Citation / status

No peer-reviewed publication identifier is assigned here yet. When citing this implementation, use the repository URL and a specific commit hash. This release makes **no** claim of journal acceptance.

## License

Source code authored for this project is released under the [MIT License](LICENSE). Numeric tables in `release_tables/` are supplied for research verification; see [DATA_NOTICE.md](DATA_NOTICE.md).
