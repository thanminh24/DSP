# Codebase Summary

## Overview

NoiSyn (Noise-Aware Out-of-Fold Synthesis): addresses hidden minority-class label noise in
imbalanced tabular classification. Combines Confidence-Weighted Majority Suppression (CWMS)
and Minority-Side Boundary Synthesis (MSBS) without modifying any labels.

Key finding: NoiSyn outperforms class-proportional reweighting by +3.47 pp BA for LR
(p=6.08e-15) and beats IW-SMOTE — the strongest competitor with public code — by +1.84 pp
(p=0.023). Gains are significant for LR, SVM, LightGBM; neutral/negative for RF/ET
(bootstrap aggregation already robust to noise). Shuffled-score ablation confirms OOF
ordering is load-bearing across all 5 compatible model families.

## Active Modules

### Core Pipeline (`pipeline/`)
- `pipeline/data/loaders.py`: cached OpenML dataset loading, imbalance induction via
  minority undersampling, synthetic class-dependent label noise injection
- `pipeline/data/encoding.py`: target-encode categorical columns, return numeric arrays
- `pipeline/scoring/balanced_oof.py`: class-balanced 5-fold OOF P(minority|x) scores
  (self-family scoring: scorer uses same model family as final predictor)
- `pipeline/scoring/oof_loss.py`: standard OOF cross-entropy loss for deletion baselines
- `pipeline/augmentation/msbs.py`: Minority-Side Boundary Synthesis — interpolates
  synthetic minority samples near the corrupted boundary using suspiciousness-weighted seeds
- `pipeline/augmentation/relabeling.py`: top-k majority-pool relabeling (historical; not
  used in primary method)
- `pipeline/augmentation/synthesis.py`: Confidence-Guided Minority Synthesis (unused in
  active sweep; candidate for cleanup)
- `pipeline/baselines/soft_weighting.py`: CWMS sample weights (standard + boosting variant)
- `pipeline/baselines/cleanlab_baselines.py`: cleanlab find_label_issues wrapper
- `pipeline/baselines/confidence_relabeling.py`: unbalanced OOF confidence relabeling baseline
- `pipeline/cleaning/selectors.py`: global top-loss, class-proportional, oracle deletion
- `pipeline/evaluation/metrics.py`: deletion-method evaluation (BA, accuracy, recall, etc.)
- `pipeline/evaluation/augment_metrics.py`: augment-method evaluation (all metrics + n_synthetic)
- `pipeline/models/factories.py`: 9 model factories including `make_cwms_factory()` for
  balanced scorer instances (spw=1.0)
- `pipeline/core/config.py`: shared experiment configuration (datasets, seeds, noise rates)

## Method Variants (in CSV column `method`)

| CSV value | Paper name | Description |
|-----------|-----------|-------------|
| `cwms_msbs` | NoiSyn | Full method: CWMS weights + MSBS synthesis with self-family OOF scorer |
| `cwms_msbs_shuffled` | NoiSyn-Shuffled | OOF scores permuted within majority pool (ablation) |
| `cwms_msbs_knn` | NoiSyn-kNN | k-NN majority ratio scorer instead of model OOF (scorer agnosticism study) |
| `cwms_msbs_crossfamily` | NoiSyn-CrossFamily | HGB OOF scorer applied to LR predictor (cross-family study) |
| `cwms` | CWMS only | Suppression weights without synthesis component |
| `msbs` | MSBS only | Synthesis without suppression weights |
| `class_proportional` | Class Prop. | Inverse-frequency sample weights, no synthesis |
| `smote` | SMOTE | Standard SMOTE, no noise awareness |
| `no_cleaning` | No Cleaning | Vanilla train on corrupted labels |
| `iw_smote` | IW-SMOTE | Zhang et al. 2022 (competitor, public code) |
| `sw_framework` | SW-approx | Xu et al. 2022 approximation (no public code) |

## Scripts (`scripts/`) — active

| Script | Purpose |
|--------|---------|
| `run_full_benchmark_solution.py` | **Primary** internal sweep (7 models, all metrics, 3 protocols, 10 seeds) |
| `run_competitor_headtohead.py` | External competitor sweep (LR, medium noise, IW-SMOTE, SW-approx) |
| `run_scorer_agnosticism_sweep.py` | kNN/cross-family scorer variants sweep |
| `analyze_full_benchmark.py` | **Primary** analysis: Wilcoxon tests, per-model tables, all metrics |
| `analyze_competitor_headtohead.py` | Competitor comparison analysis |
| `analyze_scorer_agnosticism.py` | Scorer agnosticism analysis |
| `_sweep_utils.py` | Shared sweep infrastructure (combo key, append-mode CSV, dispatchers) |
| `run_relabeling_viability_sweep.py` | Historical: original viability proof sweep |
| `download_datasets.py` | Fetch OpenML datasets |
| `validate_environment.py` | Check conda env, package versions |

## Outputs (`outputs/`)

| File | Contents | Rows |
|------|---------|------|
| `full-benchmark-solution.csv` | **PRIMARY** — 7 models, 5 datasets, 10 seeds, 3 protocols, all metrics | 8,250 |
| `competitor-headtohead.csv` | LR vs IW-SMOTE/SW-approx, medium noise | 300 |
| `scorer-agnosticism-sweep.csv` | kNN and cross-family scorer variants | varies |
| `cwms-msbs-deep-sweep.csv` | Legacy v2 sweep; superseded by full-benchmark-solution | — |
| `cwms-msbs-full-sweep.csv` | Legacy v1 sweep; reference only | — |
| `relabeling-viability-verdict.md` | Historical: why relabeling was discouraged | — |
| `archive/superseded-results.tar.gz` | Pre-pivot experiment results | — |

## Documentation (`docs/`)

| File | Contents |
|------|---------|
| `docs/paper-draft.md` | Full paper draft (NoiSyn, 8 sections, all tables) |
| `docs/results-reference.md` | Machine-verified numbers from CSVs (Tables 1, 1b, 2, 2b) |
| `docs/paper-outline.md` | Paper structure and section notes |
| `docs/reproducibility-guide.md` | Step-by-step reproduction instructions |
| `docs/codebase-summary.md` | This file |

## Environment

Python: `/home/than-minh/miniconda3/envs/dsp/bin/python`
Packages: sklearn 1.6.1, lightgbm 4.6.0, catboost 1.2.8, xgboost 3.1.2, pandas 2.3.3

Run primary sweep:
```bash
conda activate dsp
python scripts/run_full_benchmark_solution.py
python scripts/run_competitor_headtohead.py
```

Run analysis:
```bash
python scripts/analyze_full_benchmark.py
python scripts/analyze_competitor_headtohead.py
```

## Key Design Decisions

- **Self-family OOF scoring**: scorer is a balanced instance of the final model family,
  preventing cross-family misspecification and confirmation bias
- **No label correction**: only sample weights and synthetic additions — irreversibility
  avoided in small-class settings
- **10% synthesis budget**: substantially smaller than typical SMOTE; allocates precisely
  at the corrupted boundary
- **Append-mode CSV writes**: sweeps write incrementally; crash recovery via combo key check
- **Dynamic scale_pos_weight**: computed from noisy training labels at runtime
- **Conda env**: Python 3.12, numpy 1.26.4 (cleanlab compat), scikit-learn 1.6.1

## Active Research Plans

- `plans/260522-2200-scorer-agnosticism-knn-crossfamily/` — kNN and cross-family scorer
  robustness study (in progress)

## Unresolved Questions

- Multi-class extension not explored
- Natural (non-injected) label noise evaluation not included
- Comparison against loss-correction methods (Patrini 2017) not included
