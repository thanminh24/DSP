# Codebase Summary

CRCC (Class-Risk-Constrained Cleaning) experiment pipeline for imbalanced tabular ML. Implements a lightweight post-detection label-cleaning rule that prevents over-deleting clean minority-class samples. The pipeline is modularized into a `pipeline/` package with all modules under 200 lines. Backed by a 25-paper research foundation document.

## Project Structure

```
DSP/
├── data/                        # Cached Parquet datasets (5 datasets)
│   ├── pima.parquet             # Pima Indians Diabetes (768 rows, 8 numeric)
│   ├── credit-g.parquet         # German Credit (1,000 rows, 20 mixed)
│   ├── yeast.parquet            # Yeast (1,484 rows, 8 numeric)
│   ├── ecoli.parquet            # Ecoli (336 rows, 7 numeric)
│   └── phoneme.parquet          # Phoneme (5,404 rows, 5 numeric)
├── pipeline/                    # Modular experiment pipeline (all files ≤200 LOC)
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py            # BaseExperimentConfig dataclass (51 lines)
│   │   └── experiment.py        # Single-experiment run logic (190 lines)
│   ├── data/
│   │   └── loaders.py           # Dataset loading, imbalance, noise injection (104 lines)
│   ├── cleaning/
│   │   └── selectors.py         # Deletion strategies: 7 methods (142 lines)
│   ├── scoring/
│   │   └── oof_loss.py          # Out-of-fold cross-entropy scoring (39 lines)
│   └── evaluation/
│       └── metrics.py           # Metric computation after retraining (53 lines)
├── scripts/                     # Experiment entry points (thin orchestration)
│   ├── run_full_experiment.py   # Main experiment: 5 datasets x 2 models x 5 seeds x 10 methods
│   ├── run_noise_ablation.py    # 3 noise levels ablation
│   ├── run_budget_ablation.py   # 5 budget levels ablation
│   ├── run_statistical_tests.py # Wilcoxon + Cohen's d significance tests
│   ├── run_oracle_paradox_analysis.py  # Oracle vs CRCC-P comparison
│   ├── run_mild_imbalance_experiment.py # Milder imbalance variant
│   └── run_crcc_smoke_test.py   # Quick smoke test
├── outputs/                     # Generated outputs (CSV + plots)
│   ├── full-experiment-results.csv
│   ├── noise-ablation-results.csv / noise-ablation-summary.csv
│   ├── budget-ablation-results.csv / budget-ablation-summary.csv
│   ├── statistical-tests-results.csv
│   ├── oracle-paradox-analysis.csv
│   └── plots/
├── docs/                        # Documentation
│   ├── research-foundation.md   # 25-paper literature review + gap analysis (117 lines)
│   ├── experiment-report.md     # Full results + analysis
│   ├── codebase-summary.md      # This file
│   └── tabular-class-risk-capped-label-cleaning-proposal.md
└── requirements.txt
```

## Pipeline Architecture

```
pipeline/core/config.py          # Single source of truth: datasets, seeds, noise rates, lambda grid
        │
        ▼
scripts/run_full_experiment.py   # Orchestrator: 5 datasets x 2 models x 5 seeds x 10 methods
  ├── pipeline/data/loaders.py
  │     ├── load_dataset()              # Load from data/*.parquet
  │     ├── induce_imbalance()          # Subsample to 85/15 ratio
  │     └── inject_noise()              # Class-dependent: 30% min→maj, 10% maj→min
  ├── pipeline/scoring/oof_loss.py      # 5-fold stratified CV → cross-entropy loss
  ├── pipeline/cleaning/selectors.py    # Apply deletion strategy (7 methods)
  └── pipeline/evaluation/metrics.py    # Retrain clean model + compute metrics
        │
        ▼
scripts/run_*.py                 # Post-hoc analyses: ablations, statistical tests, oracle paradox
```

## Deletion Strategies (in pipeline/cleaning/selectors.py)

| Method | Description |
|--------|-------------|
| No Cleaning | Returns empty array -- baseline |
| Random | Random uniform selection |
| Global Top-Loss | Top-k by suspiciousness (highest loss first) |
| Class-Proportional | Within-class top-k, budget proportional to class frequency |
| CRCC-P | Risk-adjusted score + proportional per-class deletion caps |
| CRCC-M | Same as CRCC-P but minority cap halved (factor 0.5) |
| Oracle | Uses ground-truth noise labels (upper bound -- not practical) |

## Metrics (in pipeline/evaluation/metrics.py)

- `balanced_accuracy` — Balanced accuracy on clean test set
- `macro_f1` — Macro-averaged F1 score
- `minority_recall` — Recall for minority class
- `noise_precision_deleted` — Fraction of deleted samples that are truly mislabeled
- `clean_minority_deletion_rate` (CMDR) — Fraction of deleted samples that are clean minority instances (harm metric; 0 is ideal)

## Key Design Decisions

- **Separation of detection and intervention**: Suspiciousness scoring is independent of the deletion strategy. Any scorer can be plugged into any selector.
- **Caps dominate over risk adjustment**: Under 85/15 imbalance, class-proportional caps alone achieve ~88% mean CMDR reduction. Lambda (risk-adjusted ranking) provides negligible additional benefit because the minority cap is hit before any minority sample needs score adjustment -- confirmed across all 5 datasets.
- **Post-detection framing**: CRCC does not change how samples are scored -- it changes which scored samples get deleted given a fixed budget.
- **One-class safety guard**: When a dataset has a very small minority count, the cap naturally prevents over-deletion (edge case handling confirmed on ecoli with n_minority <= budget_count).

## Datasets

| Name | Source | Rows | Features | Minority Label | Minority % (natural) |
|------|--------|------|----------|----------------|----------------------|
| Pima | KEEL/UCI | 768 | 8 numeric | tested_positive | ~35% |
| Credit-G | KEEL/UCI | 1,000 | 20 mixed | bad | ~30% |
| Yeast | KEEL/UCI | 1,484 | 8 numeric | MIT | ~31% |
| Ecoli | KEEL/UCI | 336 | 7 numeric | im | ~10% |
| Phoneme | KEEL/UCI | 5,404 | 5 numeric | nasal | ~29% |

All datasets are cached as Parquet files in `data/`, sourced from KEEL/UCI repositories and selected based on a 25-paper literature review (see `docs/research-foundation.md`). All datasets are subsampled to 85/15 majority/minority ratio before noise injection, except where natural ratio already achieves that.

## Experiment Scripts

| Script | Purpose | Scale |
|--------|---------|-------|
| `run_full_experiment.py` | Main experiment: all methods at default noise/budget | 5 datasets x 2 models x 5 seeds x 10 methods = 500 rows |
| `run_noise_ablation.py` | Robustness across low/medium/high noise rates | 3 noise levels x 5 datasets x 2 models x 5 seeds = 150 rows |
| `run_budget_ablation.py` | CMDR-vs-budget curve across 5 budget levels | 5 budgets x 5 datasets x 2 models x 5 seeds = 250 rows |
| `run_statistical_tests.py` | Wilcoxon signed-rank + Cohen's d (global vs CRCC-P CMDR) | Paired tests across all dataset/model combos |
| `run_oracle_paradox_analysis.py` | Analysis of why CRCC-P outperforms oracle deletion | 5 datasets x 2 models x 5 seeds = 50 combos |
| `run_mild_imbalance_experiment.py` | Variant with milder (30/70) imbalance ratio | Supplementary experiment |
| `run_crcc_smoke_test.py` | Quick sanity check | Single dataset, reduced scale |

## Running the Pipeline

```bash
# 1. Full experiment (~10-15 min)
python scripts/run_full_experiment.py

# 2. Ablation studies
python scripts/run_noise_ablation.py
python scripts/run_budget_ablation.py

# 3. Statistical analysis
python scripts/run_statistical_tests.py
python scripts/run_oracle_paradox_analysis.py

# 4. Quick smoke test (~30s)
python scripts/run_crcc_smoke_test.py
```

Outputs go to `outputs/`: CSV result files and summary tables.
