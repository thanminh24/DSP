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

Minimal reproduction proving the core claim (~2h, 6 models, medium protocol):

```bash
conda activate dsp
python scripts/run_cwms_msbs_deep_sweep.py --medium-only
python scripts/analyze_cwms_msbs_deep_results.py
```

Output files:
- `outputs/cwms-msbs-deep-sweep.csv` — 1,350 rows (6 models × 5 datasets × 10 seeds)
- Statistical summary printed to stdout (BA, recall, precision, F1, Wilcoxon p-values)

## Full Sweep (All 3 Noise Protocols)

```bash
conda activate dsp
python scripts/run_cwms_msbs_deep_sweep.py
python scripts/analyze_cwms_msbs_deep_results.py
```

Expected runtime: ~4h on CPU (8 cores). GPU reduces boosting time by ~40%.

## Grid Dimensions

Per sweep: 6 models (5 with full methods, 1 with 2 methods) × 5 datasets × 10 seeds
× 3 noise protocols × 1 budget (0.10) × 1 ratio (0.15) = ~4,300 rows.

## Output Schema

| Column | Description |
|--------|-------------|
| dataset | Dataset name (pima, credit-g, yeast, ecoli, phoneme) |
| model | Model family |
| seed | Random seed (10 values) |
| noise_protocol | Noise type (hidden_minority_low/medium/high) |
| mn_to_maj | Minority→majority noise rate |
| maj_to_min | Majority→minority noise rate |
| budget | Budget fraction of training set |
| target_ratio | Imbalance target ratio |
| method | Method name (no_cleaning, class_proportional, msbs, cwms, cwms_msbs) |
| balanced_accuracy | Test-set balanced accuracy |
| accuracy | Test-set accuracy |
| macro_f1 | Macro-averaged F1 |
| weighted_f1 | Weighted F1 |
| minority_recall | Test-set minority class recall |
| minority_precision | Test-set minority class precision |
| majority_recall | Test-set majority class recall |
| n_synthetic | Number of MSBS synthetic samples |
| macro_f1 | Test-set macro F1 |
| relabel_correctness | Fraction of relabeled samples that were true minority |
