# Artefacts for the QREI submission

**Paper:** *When may a verifier advise? A competence test for maintenance records, and the record
it needs* (submitted to *Quality and Reliability Engineering International*).

This file maps what the paper reports onto what this repository contains, and says plainly what is
and is not reproducible from the released files.

> **Note on two papers.** This repository also carries the artefacts of a companion measurement
> study on the same grinding campaign. `release_tables/headline/PAPER_TABLES.md` documents **that**
> paper (estimator comparisons, spatial holdout, corruption), not this one. The mapping for this
> paper is the table below.

## Reproducible from the released files

| Paper element | Artefact | Command |
|---|---|---|
| Acceptance bound for the early–late mirror: point **0.6325**, pass-resampling lower bound **0.568**, cluster-resampling lower bound **0.489–0.500** | `analysis/qrei_acceptance_bounds.py` + `data/scores_per_pass.csv` | `python analysis/qrei_acceptance_bounds.py` |
| Operating-characteristic curve and the power numbers (Section 4.5, Table 5, Figure 3): $s_{pass}=0.032$, $s_{cluster}=0.070$, $\mu_{50}=0.563/0.638$, $\mu_{80}=0.590/0.697$ | `analysis/qrei_oc_curve.py` + the same scores file | `python analysis/qrei_oc_curve.py` |
| Per-class margins $\mu_c=\mathrm{Det}_c-\mathrm{FA}$ (Table 5 of the supplement; detection board) | `data/scores_per_pass.csv`, column `flagged` | counts per class |
| Cost weights, break-even operating point: $c_{FA}=2$, $c_{miss}=45$, $thr^{\star}=8$, $thr=10$ | `release_tables/softadv_guarantees.json` | — |
| Decision card operating point: $n=234$, $\mathrm{FA}=0.265$, $\mathrm{Det}_{large}=0.987$ | `release_tables/decision_quality_card.csv` | — |

The scores file has 2,106 rows: 234 passes × 9 error classes, with the written index after
injection, the fused estimate, the threshold and the flag. It is the same file that accompanies the
submission.

## Present in the repository, not re-run here

| Element | Artefact |
|---|---|
| Record schema (11 entities) | `schema/decision-record-1.0.json` |
| Constraint validator and self-tests | `schema/validate.py`, `schema/test_schema.py`, `schema/probe_failclosed.py` |
| Cost model and card checker | `schema/costmodel.py`, `schema/check_card.py`, `schema/release.py` |
| Protocol tables | `release_tables/*.csv`, `release_tables/case_a2_protocol_rows.csv` |

## Not reproducible from the released files

| Element | Status |
|---|---|
| Duplicate-index and gap checks at **86.8%** and **97.9%** detection on complete records (Section 7.x) | These come from a **separate slip construction**, not from the injected offsets in `scores_per_pass.csv`. The slip log is available from the corresponding author. |
| Full laboratory frames and model checkpoints | Available to editors and reviewers; to be deposited on publication. |

## What changed relative to the submitted version

The bootstrap bound in the manuscript is quoted as **0.498**, which is one attained value: across
bootstrap seeds the cluster bound ranges from **0.489 to 0.500**, so it never clears the 0.50
margin. The script prints the value for the seed it was run with and records the range in its
`note` field.
