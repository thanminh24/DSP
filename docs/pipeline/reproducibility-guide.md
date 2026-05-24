# Reproducibility Guide

## Environment

```bash
conda create -n dsp python=3.12 -y
conda activate dsp
pip install -r requirements.txt
```

Key dependencies: scikit-learn 1.6.1, numpy 1.26.4, xgboost 3.1.2, lightgbm 4.6.0,
catboost 1.2.8. GPU acceleration: `--gpu` flag enables CUDA for XGBoost/LightGBM/CatBoost.

## Data (15 datasets)

Download all 15 datasets from OpenML and cache under `data/`:

```bash
python scripts/download_datasets.py
```

Datasets: pima, credit-g, yeast, ecoli, phoneme, breast_cancer, ilpd, blood, haberman,
ionosphere, vehicle_bus, glass_float, abalone, spambase, kc1.

## Full Reproduction (paper results)

### Table 1 — Internal Benchmark (9 models × 15 datasets × 3 protocols × 10 seeds)

```bash
# ~24h on GPU; resume-safe if interrupted
python scripts/run_full_benchmark_solution.py --gpu
python scripts/analyze_full_benchmark.py --input outputs/full-benchmark-solution-v2.csv
```

### Table 2 — External Comparison (LR+SVM+HGB × 3 protocols × 15 datasets)

```bash
python scripts/run_expanded_competitor_headtohead.py
python scripts/analyze_expanded_competitor_headtohead.py
```

### Table 3 — RF/ET Component Ablation

```bash
python scripts/run_rfet_ablation_sweep.py
python scripts/analyze_rfet_ablation.py
```

### Table 4 — Failure Mode Analysis (symmetric + reverse-asymmetric noise)

```bash
python scripts/run_failure_mode_sweep.py
python scripts/analyze_failure_mode.py
```

### Table 5 — Clean-Data Ablation

```bash
python scripts/run_clean_data_ablation.py
# No dedicated analysis script; read CSV directly or add to analyze_full_benchmark.py
```

### Table 6 — IR=0.30 Sensitivity

```bash
python scripts/run_full_benchmark_solution.py --gpu --ratio 0.30
python scripts/analyze_ir_sweep.py
```

## Output Schema (full-benchmark-solution-v2.csv)

| Column | Description |
|--------|-------------|
| dataset | Dataset name (one of 15) |
| model | Model family (lr, svm, hgb, lightgbm, catboost, random_forest, extra_trees, xgboost, calibrated_lr) |
| seed | Random seed (13, 17, 23, 29, 31, 37, 41, 43, 47, 53) |
| noise_protocol | hidden_minority_low / hidden_minority_medium / hidden_minority_high |
| mn_to_maj | Minority→majority noise rate |
| maj_to_min | Majority→minority noise rate |
| budget | Budget fraction of training set (0.10) |
| target_ratio | Imbalance target minority ratio (0.15 or 0.30) |
| method | no_cleaning, class_proportional, smote, cwms, msbs, cwms_msbs, cwms_msbs_shuffled |
| balanced_accuracy | Test-set balanced accuracy (primary metric) |
| pr_auc | Precision-Recall AUC (from predict_proba or decision_function) |
| minority_recall | Test-set minority class recall |
| minority_precision | Test-set minority class precision |
| majority_recall | Test-set majority class recall |
| g_mean | Geometric mean (√(minority_recall × majority_recall), computed in analysis) |
| n_synthetic | Number of MSBS synthetic samples added |
| error | Non-null only for failed rows (excluded by analysis scripts) |

## Statistical Methodology

Per-dataset Wilcoxon signed-rank test (one-sided) + Stouffer's Z combination:

```python
from scripts.analyze_full_benchmark import per_dataset_wilcoxon_stouffer
result = per_dataset_wilcoxon_stouffer(df, method_a="cwms_msbs", method_b="class_proportional")
# result["stouffer_z"], result["stouffer_p"], result["n_datasets_significant"]
```

Within each dataset: 30 pairs (10 seeds × 3 protocols). Stouffer combination is valid
because datasets are independent replication units.

## Method Key

| CSV value | Paper name | Description |
|---|---|---|
| `cwms_msbs` | NoiSyn | Full method: CWMS + MSBS with self-family OOF scorer |
| `cwms_msbs_shuffled` | NoiSyn-Shuffled | OOF scores permuted within majority pool |
| `cwms` | CWMS-only | Suppression weights, no synthesis |
| `msbs` | MSBS-only | Synthesis, no suppression weights |
| `class_proportional` | Class Prop. | Inverse-frequency weights, no synthesis |
| `smote` | SMOTE | Standard SMOTE, noise-unaware |
| `no_cleaning` | No Cleaning | Vanilla train on corrupted labels |
| `iw_smote` | IW-SMOTE | Zhang et al. 2022 (competitor, real code) |
| `sw_framework` | SW-approx | Xu et al. 2022 approximation |
