---
title: "CRCC Full Experiment Pipeline"
description: "Implement and run all experiments for Class-Risk-Constrained Cleaning (CRCC) on 3 tabular datasets × 2 model families × 5 seeds × 6 cleaning methods (180 training runs) as specified in tabular-class-risk-capped-label-cleaning-proposal.md."
status: completed
priority: P1
branch: ""
tags: ["ml", "noisy-labels", "tabular", "experiment"]
blockedBy: []
blocks: []
created: "2026-05-19T16:10:32.990Z"
createdBy: "ck:plan"
source: skill
---

# CRCC Full Experiment Pipeline

## Overview

End-to-end implementation of the CRCC (Class-Risk-Constrained Cleaning) experiment described in `docs/tabular-class-risk-capped-label-cleaning-proposal.md`. Extends the existing smoke test (`scripts/run-crcc-smoke-test.py`) into a full modular experiment pipeline covering three real tabular datasets, two model families, oracle deletion, CRCC-M ablation, and lambda grid search. Produces metric tables, plots, and a written report.

**Runtime:** `/home/than-minh/miniconda3/bin/python3`

**Datasets (cached locally in `data/` as Parquet — no network needed):**
- Pima Indians Diabetes — `data/pima.parquet`, 768×8, 500/268 → subsample to 85/15
- Credit-G — `data/credit-g.parquet`, 1000×20, 700/300 → subsample to 85/15
- Sick/Thyroid — `data/sick.parquet`, 3772×29, 3541/231 → preserve natural ~6.1% imbalance

**Scale:** 3 datasets × 2 models × 5 seeds × 10 method variants = 300 rows in CSV

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Environment and Dependency Validation](./phase-01-environment-and-dependency-validation.md) | Complete |
| 2 | [Data Loading and Preprocessing Module](./phase-02-data-loading-and-preprocessing-module.md) | Complete |
| 3 | [Scoring and Selection Modules](./phase-03-scoring-and-selection-modules.md) | Complete |
| 4 | [Evaluator and Experiment Orchestrator](./phase-04-evaluator-and-experiment-orchestrator.md) | Complete |
| 5 | [Run Full Experiments](./phase-05-run-full-experiments.md) | Complete |
| 6 | [Results Analysis and Visualization](./phase-06-results-analysis-and-visualization.md) | Complete |
| 7 | [Experiment Report](./phase-07-experiment-report.md) | Complete |

## Key Files

| File | Role |
|------|------|
| `data/pima.parquet`, `data/credit-g.parquet`, `data/sick.parquet` | Locally cached datasets (no network needed) |
| `scripts/data-loader.py` | Load from local parquet, imbalance induction, noise injection |
| `scripts/scoring.py` | Out-of-fold cross-entropy suspiciousness scorer |
| `scripts/selectors.py` | All 7 cleaning method selectors |
| `scripts/evaluator.py` | Retrain + 5-metric evaluation |
| `scripts/run-full-experiment.py` | Orchestrator — all combos, saves CSV |
| `scripts/plot-results.py` | 3 plots + summary table |
| `outputs/full-experiment-results.csv` | Raw results (all 180+ runs) |
| `outputs/summary-table.csv` | Mean ± std by dataset/model/method |
| `outputs/plots/` | 3 key figures |
| `docs/experiment-report.md` | Concise written report |

## Dependencies

None — self-contained project.
