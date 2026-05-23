# Codebase Summary

## Overview

NoiSyn (Noise-Aware Out-of-Fold Synthesis): addresses hidden minority-class label noise in
imbalanced tabular classification. Combines Confidence-Weighted Majority Suppression (CWMS)
and Minority-Side Boundary Synthesis (MSBS) without modifying any labels.

Key findings (Phase 3–5 complete, 15 datasets, hardened):
- **LR**: +3.16 pp BA, Stouffer Z=9.31, p≈0, 9/15 datasets individually sig (CONFIRMED, final)
- **SVM**: +0.37 pp BA, Z=1.24, p=0.11 — NOT significant at 15-dataset scale
- **HGB/LGB**: near-zero overall (noise-level interaction: gains at low IR, losses at high IR)
- **CatBoost**: -1.11 pp BA (negative, consistent)
- **RF**: -4.37 pp BA, ET: -3.79 pp BA (strongly harmful; OOF signal diluted by bootstrap)
- **IR=0.30 sensitivity**: LR +1.74 pp (Z=5.16, significant); SVM +4.05 pp (Z=8.08, significant)
- RF/ET component ablation confirms CWMS suppression is the primary harm source for bootstrap
  ensemble models. Failure-mode analysis confirms method is limited to hidden-minority asymmetric
  noise only (not symmetric or reverse-asymmetric). Clean-data ablation: +2.62 pp LR even 
  without noise injection, confirming boundary-aware synthesis as a dual mechanism.

Statistical methodology: per-dataset Wilcoxon signed-rank (30 pairs per dataset = 10 seeds
× 3 protocols) + Stouffer's Z combination. Replaces deprecated aggregate Wilcoxon on i.i.d.
pairs assumption.

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

| Script | Purpose | Status |
|--------|---------|--------|
| `run_full_benchmark_solution.py` | **PRIMARY** sweep (9 models, 15 datasets, all metrics, 3 protocols, 10 seeds); `--ratio` for IR sensitivity | ✓ Complete |
| `run_expanded_competitor_headtohead.py` | Phase 5 competitor comparison (LR+SVM+HGB × 3 protocols × 15 datasets) | ✓ Complete |
| `run_rfet_ablation_sweep.py` | RF/ET component ablation (cwms-only vs msbs-only) | ✓ Complete |
| `run_failure_mode_sweep.py` | Failure-mode protocols (symmetric + reverse-asymmetric) | ✓ Complete |
| `run_clean_data_ablation.py` | Clean-data ablation (zero noise, LR+SVM) | ✓ Complete |
| `run_iw_smote_lamda_sweep.py` | IW-SMOTE λ sensitivity (λ ∈ {10,20,30,50,100}) | ✓ Complete |
| `run_scorer_agnosticism_sweep.py` | kNN/cross-family scorer variants sweep | ✓ Complete |
| `analyze_full_benchmark.py` | **PRIMARY** analysis: per-dataset Wilcoxon + Stouffer (Table 1) | ✓ Complete |
| `analyze_expanded_competitor_headtohead.py` | Table 2: 3-model × 3-protocol expanded comparison | ✓ Complete |
| `analyze_rfet_ablation.py` | Table 3: RF/ET ablation decomposition | ✓ Complete |
| `analyze_failure_mode.py` | Table 4: failure mode (symmetric/reverse-asymmetric) | ✓ Complete |
| `analyze_ir_sweep.py` | Table 6: IR=0.15 vs IR=0.30 sensitivity | ✓ Complete |
| `analyze_scorer_agnosticism.py` | Scorer agnosticism robustness (kNN, cross-family) | ✓ Complete |
| `_sweep_utils.py` | Shared sweep infrastructure (combo key, append-mode CSV, dispatchers) | ✓ Active |
| `download_datasets.py` | Fetch OpenML datasets (15 benchmarks) | ✓ Complete |
| `validate_environment.py` | Check conda env, package versions | ✓ Active |
| `run_competitor_headtohead.py` | Historical: external competitor sweep (LR, medium noise) | Archived |
| `run_relabeling_viability_sweep.py` | Historical: original viability proof sweep | Archived |

## Outputs (`outputs/`)

| File | Contents | Rows | Status |
|------|---------|------|--------|
| `full-benchmark-solution.csv` | ORIGINAL — 9 models, 5 datasets, 10 seeds, 3 protocols | 8,250 | Archived |
| `full-benchmark-solution-v2.csv` | **PRIMARY (Phase 3, FINAL)** — 9 models, 15 datasets, 10 seeds, 3 protocols | 24,750 | Complete |
| `full-benchmark-ir030-solution.csv` | **Phase 4 (FINAL)** — same as v2 but target_ratio=0.30 | 8,250 | Complete |
| `competitor-headtohead.csv` | LR vs competitors, 5 datasets, medium noise only | 300 | Archived |
| `competitor-headtohead-expanded.csv` | **Phase 5 (FINAL)** — LR+SVM+HGB vs competitors, 15 datasets, 3 protocols | 8,100 | Complete |
| `rfet-ablation-sweep.csv` | RF/ET ablation: cwms-only vs msbs-only (Phase 2) | 1,500 | Complete |
| `failure-mode-sweep.csv` | Symmetric + reverse-asymmetric failure modes (Phase 2) | 400 | Complete |
| `clean-data-ablation.csv` | Zero-noise clean-data ablation (Phase 2) | 400 | Complete |
| `iw-lamda-sweep.csv` | IW-SMOTE λ sensitivity (Phase 2) | 700 | Complete |
| `scorer-agnosticism-sweep.csv` | kNN and cross-family scorer variants | 2,700 | Complete |

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

Run primary sweep (Phase 3 — 15 datasets):
```bash
conda activate dsp
python scripts/run_full_benchmark_solution.py --gpu  # outputs/full-benchmark-solution-v2.csv
python scripts/run_full_benchmark_solution.py --gpu --ratio 0.30  # Phase 4: IR=0.30
python scripts/run_expanded_competitor_headtohead.py  # Phase 5: 3 models × 3 protocols × 15 datasets
```

Run Phase 2 ablation sweeps:
```bash
python scripts/run_rfet_ablation_sweep.py
python scripts/run_failure_mode_sweep.py
python scripts/run_clean_data_ablation.py
python scripts/run_iw_smote_lamda_sweep.py
```

Run analysis:
```bash
python scripts/analyze_full_benchmark.py --input outputs/full-benchmark-solution-v2.csv
python scripts/analyze_expanded_competitor_headtohead.py
python scripts/analyze_rfet_ablation.py
python scripts/analyze_failure_mode.py
python scripts/analyze_ir_sweep.py
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

## Completed Research Plans

- `plans/260523-1200-noisyn-paper-hardening/` — 6-phase hardening to reach PR/KBS tier
  **STATUS: COMPLETE (Phases 1-6 delivered)**
  - Phase 1: Quick fixes (PR-AUC, per-dataset stats, binary label assertion) ✓
  - Phase 2: Fast sweeps (RF/ET ablation, IW-SMOTE λ, clean-data, failure modes) ✓
  - Phase 3: Dataset expansion to 15, re-run benchmark (24,750 rows) ✓
  - Phase 4: IR=0.30 sweep (8,250 rows, LR/SVM/HGB sensitivity analysis) ✓
  - Phase 5: Expanded external comparison (LR+SVM+HGB × 3 protocols × 15 datasets) ✓
  - Phase 6: Paper rewrite with hardened tables and honest framing ✓

## Paper Status

**NoiSyn Hardening Complete (Phases 1-6)**

Final results submitted to peer review. All 6 phases delivered with verified findings:
- Statistical methodology: per-dataset Wilcoxon + Stouffer Z (not aggregate i.i.d.)
- 15-dataset benchmark: 24,750 rows, 10 seeds, 3 noise protocols
- 9 model families tested: LR (confirmed), SVM (marginal), HGB/LGB (neutral), CatBoost/RF/ET (negative)
- IR sensitivity: LR gains hold across IR=0.15 and IR=0.30 (both significant)
- Honest framing: method limited to hidden-minority asymmetric noise, not symmetric or reverse

Key tables finalized:
- Table 1: Per-model + per-dataset Wilcoxon p-values, 15-dataset hardened stats
- Table 2: Expanded competitor headtohead (LR+SVM+HGB vs IW-SMOTE, SW-approx)
- Table 3: RF/ET component ablation (confirms CWMS is harm source)
- Table 4: Failure mode validation (symmetric/reverse-asymmetric show negative/neutral)
- Table 5: Clean-data ablation (synthesis contributes +2.62pp even zero noise)
- Table 6: IR=0.30 sensitivity (LR +1.74pp, SVM +4.05pp, both significant)

## Unresolved Questions

None — paper hardening complete with all statistical validations passed.
