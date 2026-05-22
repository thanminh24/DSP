# Codebase Summary

## Overview

Confidence-guided OOF relabeling for hidden-minority label noise in imbalanced tabular
classification. Replaces the earlier CRCC deletion framing as the primary contribution.

Key finding: class-balanced out-of-fold confidence scoring + top-k relabeling preserves
minority feature evidence better than deletion-based cleaning. Verified across 8 model
families (LR, calibrated LR, random forest, extra trees, HGB, XGBoost, LightGBM, CatBoost).

## Active Modules

### Core Pipeline (`pipeline/`)
- `pipeline/data/loaders.py`: cached OpenML dataset loading, imbalance induction via
  minority undersampling, synthetic class-dependent label noise injection
- `pipeline/data/encoding.py`: target-encode categorical columns, return numeric arrays
- `pipeline/scoring/balanced_oof.py`: class-balanced 5-fold OOF P(minority|x) scores
- `pipeline/scoring/oof_loss.py`: standard OOF cross-entropy loss for deletion baselines
- `pipeline/augmentation/relabeling.py`: top-k majority-pool relabeling (Type A) and
  random relabeling control
- `pipeline/baselines/cleanlab_baselines.py`: cleanlab find_label_issues wrapper
- `pipeline/baselines/confidence_relabeling.py`: unbalanced OOF confidence relabeling baseline
- `pipeline/cleaning/selectors.py`: global top-loss, class-proportional, and oracle deletion selectors
- `pipeline/evaluation/metrics.py`: deletion-method evaluation (noise precision, minority deletion rate)
- `pipeline/evaluation/augment_metrics.py`: relabeling-method evaluation (relabel correctness, n_relabeled)
- `pipeline/models/factories.py`: 8 model factories with dynamic scale_pos_weight and GPU support
- `pipeline/core/config.py`: shared experiment configuration (datasets, seeds, noise rates, lambda grid)

### Scripts (`scripts/`)
- `scripts/run_relabeling_viability_sweep.py`: main viability grid sweep (11 methods, 5 noise protocols)
  Supports incremental CSV append + resume from checkpoint. Flags: `--quick`, `--gpu`
- `scripts/combine_relabeling_results.py`: merges per-model CSVs into canonical combined file
- `scripts/analyze_relabeling_statistics.py`: paired Wilcoxon tests, bootstrap CIs, verdict output
- `scripts/generate_figures.py`: 3 publication PNG figures from combined CSV
- `scripts/audit_relabeling_protocol.py`: static leakage and protocol audit

### Outputs (`outputs/`)
- `outputs/relabeling-viability-{model}.csv`: per-model sweep results
- `outputs/relabeling-all-results-combined.csv`: canonical combined result file
- `outputs/relabeling-statistical-tests.csv`: paired test results
- `outputs/relabeling-viability-verdict.md`: final verdict document
- `outputs/plots/`: 3 publication figures (PNG)

### Documentation (`docs/`)
- `docs/confidence-guided-relabeling-report.md`: full technical report with leakage audit
- `docs/paper-outline.md`: 8-section paper outline
- `docs/reproducibility-guide.md`: step-by-step reproduction instructions
- `docs/codebase-summary.md`: this file

## Active Research Plans

- `plans/260521-0000-confidence-guided-relabeling-viability-proof/plan.md` — original viability proof
- `plans/260521-2130-phase79-completion-boosting-stats-paper/plan.md` — boosting coverage + paper packaging

## Key Design Decisions

- **Append-mode CSV writes**: sweeps write incrementally; crash recovery via `_combo_key()` identity check
- **Dynamic scale_pos_weight**: computed from noisy training labels at runtime (not hardcoded)
- **GPU for boosting**: XGBoost GPU provides 277x speedup over CPU for small tabular data
- **Conda env**: Python 3.12 with numpy 1.26.4 (cleanlab compat), scikit-learn 1.6.1 (CatBoost compat)

## Unresolved Questions

- Weak-supervision transfer: only 1/5 WRENCH datasets show positive transfer
- Multi-class extension not explored
- Comparison against loss-correction methods (Patrini 2017, FixMatch) not included
