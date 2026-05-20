---
title: "CRCC Research Paper Completion"
description: "Complete literature foundation, dataset correction, pipeline restructure, and experiment suite for CRCC — Springer LNCS conference submission"
status: completed
priority: P1
branch: ""
tags: ["research", "ml", "paper"]
blockedBy: []
blocks: []
created: "2026-05-20T09:04:18.577Z"
createdBy: "ck:plan"
source: skill
---

# CRCC Research Paper Completion

## Overview

End-to-end completion of the CRCC (Class-Risk-Constrained Cleaning) research paper pipeline.

Current state: 3 small medical datasets, lambda insensitivity confirmed, oracle paradox identified, repo structure not modular, no literature foundation, no statistical tests, no ablation depth.

Target: 5 KEEL/UCI benchmark datasets (used by comparative literature), full pipeline/ module structure, noise/budget ablation curves, Wilcoxon tests, oracle paradox analysis, 25-paper literature review with gap analysis.

## Phases

| Phase | Name | Status | Effort | Depends On |
|-------|------|--------|--------|------------|
| 1 | [Research Foundation](./phase-01-research-foundation.md) | Complete | 2h | — |
| 2 | [Dataset Correction](./phase-02-dataset-correction.md) | Complete | 2h | — |
| 3 | [Pipeline Restructure + Refactor](./phase-03-pipeline-restructure.md) | Complete | 3h | Phase 2 |
| 4 | [Noise Rate Ablation](./phase-04-noise-rate-ablation.md) | Complete | 2h | Phase 3 |
| 5 | [Budget Ablation](./phase-05-budget-ablation.md) | Complete | 2h | Phase 3 |
| 6 | [Statistical Significance Tests](./phase-06-statistical-significance-tests.md) | Complete | 1h | Phases 4,5 |
| 7 | [Oracle Paradox Analysis](./phase-07-oracle-paradox-analysis.md) | Complete | 2h | Phase 3 |

## Key Constraints

- All Python files ≤ 200 lines; split into pipeline/ submodules if exceeded
- snake_case for all Python filenames
- No network calls during experiments — datasets pre-cached as `.parquet` in `data/`
- All experiment code in `pipeline/` (modules); `scripts/` = entry-point runners only
- Fixed random seeds (13, 29, 47, 61, 83) for reproducibility
- Datasets must appear in comparative literature (KEEL/UCI benchmarks)

## Target Datasets (Literature-Verified)

| Dataset | Rows | Features | Minority % | Source | Papers Using |
|---------|------|----------|------------|--------|--------------|
| Pima Indian Diabetes | 768 | 8 | 35% | UCI | Chawla 2002, He & Garcia 2009 |
| German Credit | 1000 | 20 | 30% | UCI | He & Garcia 2009, Fernández 2018 |
| Yeast | 1484 | 8 | 46% | KEEL | ADASYN, Krawczyk 2016, Fernández 2018 |
| Ecoli | 336 | 7 | 14% | KEEL | ADASYN, Krawczyk 2016, Fernández 2018 |
| Phoneme | 5404 | 5 | 29% | KEEL | KEEL paper, Fernández 2018 |

Sick (current) = replaced by Yeast + Ecoli (better literature coverage). MAGIC + Bank Marketing dropped (no imbalanced tabular ML citations).

## Dependencies

None (self-contained plan)
