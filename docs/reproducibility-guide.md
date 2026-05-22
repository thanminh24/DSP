# Reproducibility Guide

## Environment

```bash
conda create -n dsp python=3.12 -y
conda activate dsp
pip install -r requirements.txt
```

Key dependencies: scikit-learn 1.6.1, numpy 1.26.4, xgboost 3.1.2, lightgbm 4.6.0,
catboost 1.2.8, cleanlab 2.7.1.

GPU acceleration: `--gpu` flag on sweep scripts enables CUDA for XGBoost/LightGBM/CatBoost.
Not recommended for small datasets (< 1000 rows) due to transfer overhead.

## Data

Datasets are auto-downloaded from OpenML on first run and cached under `data/`:
`pima`, `credit-g`, `yeast`, `ecoli`, `phoneme`.

## Quick Reproduction (Core Claim)

Minimal reproduction proving the core claim (LR model, quick grid, ~15 min):

```bash
conda activate dsp
python scripts/run_relabeling_viability_sweep.py lr --quick
python scripts/combine_relabeling_results.py
python scripts/analyze_relabeling_statistics.py
python scripts/generate_figures.py
```

Output files:
- `outputs/relabeling-all-results-combined.csv` — combined result rows
- `outputs/relabeling-statistical-tests.csv` — paired Wilcoxon tests
- `outputs/relabeling-viability-verdict.md` — verdict + per-model win rates
- `outputs/plots/relabeling-main-table.png` — per-model Δ BA/recall
- `outputs/plots/relabeling-precision-vs-random.png` — relabel precision comparison
- `outputs/plots/relabeling-operating-condition.png` — noise protocol heatmap

## Full Sweep (All 8 Model Families)

```bash
conda activate dsp

# Tree models (~5 min each)
python scripts/run_relabeling_viability_sweep.py lr --quick
python scripts/run_relabeling_viability_sweep.py calibrated_lr --quick
python scripts/run_relabeling_viability_sweep.py extra_trees --quick
python scripts/run_relabeling_viability_sweep.py random_forest --quick
python scripts/run_relabeling_viability_sweep.py hgb --quick

# Boosting models (~30-120 min each, CPU)
python scripts/run_relabeling_viability_sweep.py xgboost --quick
python scripts/run_relabeling_viability_sweep.py lightgbm --quick
python scripts/run_relabeling_viability_sweep.py catboost --quick

# Combine + analyze + figures
python scripts/combine_relabeling_results.py
python scripts/analyze_relabeling_statistics.py
python scripts/generate_figures.py
```

Expected runtime: ~4-6 hours total on CPU (8 cores). GPU reduces boosting time by ~40%.

## Grid Dimensions (Quick Mode)

Per model: 5 datasets × 10 seeds × 3 noise protocols × 1 budget (0.10) × 1 ratio (0.15)
= 150 combos × 11 methods = 1650 rows.

## Full Grid (Publication-Grade)

Remove `--quick` flag: 5 datasets × 20 seeds × 5 noise protocols × 3 budgets × 2 ratios
= 3000 combos × 11 methods = 33,000 rows per model. Expected ~10-20 hours per model.

## Output Schema

| Column | Description |
|--------|-------------|
| dataset | Dataset name (pima, credit-g, yeast, ecoli, phoneme) |
| model | Model family |
| seed | Random seed (10 or 20 values) |
| noise_protocol | Noise type (hidden_minority_low/medium/high, reverse_asymmetric, symmetric) |
| mn_to_maj | Minority→majority noise rate |
| maj_to_min | Majority→minority noise rate |
| budget | Cleaning/relabeling budget (fraction of training set) |
| target_ratio | Imbalance target ratio |
| method | Cleaning/relabeling method name |
| balanced_accuracy | Test-set balanced accuracy |
| minority_recall | Test-set minority class recall |
| macro_f1 | Test-set macro F1 |
| relabel_correctness | Fraction of relabeled samples that were true minority |
