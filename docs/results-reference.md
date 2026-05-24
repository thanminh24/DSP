# Results Reference — COINS Benchmark

Consolidated numbers extracted from output CSVs. All figures machine-verified via script
on 2026-05-23/24.

## Consolidated Output Files

| File | Rows | Description |
|------|------|-------------|
| `outputs/COINS-all-results.xlsx` | 49,250 | All experiments in one xlsx (9 tabs + README) |
| `outputs/COINS-all-results.csv` | 49,250 | Same data, flat CSV with `experiment` tag column |
| `docs/COINS-literature-review.xlsx` | 25 papers | 2-sheet literature review (Overview + 25-paper table) |

## Individual Experiment Files

| File | Rows | Purpose |
|------|------|---------|
| `outputs/full-benchmark-solution-v2.csv` | 24,750 | Main benchmark — 15 datasets, Table 1 source |
| `outputs/full-benchmark-ir030-solution.csv` | 8,250 | IR=0.30 sensitivity sweep, Table 6 source |
| `outputs/competitor-headtohead-expanded.csv` | 8,100 | External comparison LR+SVM+HGB × 15ds, Table 2 |
| `outputs/cwms-msbs-deep-sweep.csv` | 4,350 | Deep sweep + oracle upper bound, Appendix |
| `outputs/rfet-ablation-sweep.csv` | 1,500 | RF/ET failure-mode ablation, Discussion |
| `outputs/scorer-agnosticism-sweep.csv` | 1,250 | Self-family vs cross-family OOF scorer ablation |
| `outputs/clean-data-ablation.csv` | 400 | Zero-noise ablation (no degradation check) |
| `outputs/failure-mode-sweep.csv` | 400 | Symmetric/reverse-asymmetric noise protocols |
| `outputs/iw-lamda-sweep.csv` | 250 | IW-SMOTE lambda sensitivity (gates lambda=30) |

Superseded files (3-dataset v1 runs) archived in `outputs/archive/superseded-results.tar.gz`.

## Run Metadata (Main Benchmark — v2)

| Item | Value |
|------|-------|
| Primary CSV | `outputs/full-benchmark-solution-v2.csv` |
| Competitor CSV | `outputs/competitor-headtohead-expanded.csv` |
| Primary rows | 24,750 |
| Primary columns | 23 (incl. pr_auc) |
| Datasets | 15 OpenML datasets (abalone, blood, breast_cancer, credit-g, ecoli, glass_float, haberman, ilpd, ionosphere, kc1, phoneme, pima, sonar*, spambase, vehicle_bus, yeast) |
| Seeds | 13, 17, 23, 29, 31, 37, 41, 43, 47, 53 (10) |
| Noise protocols | hidden_minority_low, hidden_minority_medium, hidden_minority_high (3) |
| Noise rates | low: ε_mn=0.10 ε_mj=0.02 · medium: ε_mn=0.25 ε_mj=0.02 · high: ε_mn=0.40 ε_mj=0.02 |
| Methods in primary | no_cleaning, class_proportional, smote, cwms, cwms_msbs, cwms_msbs_shuffled, msbs |
| Methods in competitor | no_cleaning, smote, class_proportional, iw_smote, sw_framework, cwms_msbs |
| Wilcoxon test | per-dataset signed-rank (30 pairs = 10 seeds × 3 protocols) + Stouffer Z across 15 datasets |

Method name mapping: `cwms_msbs` = COINS in paper.

---

## Table 1 — Internal Benchmark

Mean Balanced Accuracy across all 150 (dataset × seed × noise_protocol) pairs per model.
ΔBA = cwms_msbs − class_proportional. Wilcoxon p two-sided.

| Model | No Cleaning | Class Prop. | SMOTE | NoiSyn | Shuffled | ΔBA (pp) | p-value | Wins/150 |
|-------|-------------|-------------|-------|--------|----------|----------|---------|----------|
| lr | 0.5758 | 0.7047 | 0.6337 | 0.7394 | 0.7238 | +3.47 | 6.08e-15 | 114/150 |
| svm | 0.5683 | 0.6560 | 0.6330 | 0.6776 | 0.6658 | +2.16 | 1.72e-07 | 98/150 |
| hgb | 0.6352 | 0.6910 | 0.6499 | 0.6947 | 0.6842 | +0.37 | 9.09e-02 | 87/150 |
| lightgbm | 0.6370 | 0.6897 | 0.6496 | 0.6945 | 0.6792 | +0.47 | 4.03e-02 | 89/150 |
| catboost | 0.6285 | 0.7084 | 0.6579 | 0.6990 | 0.6806 | −0.94 | 2.05e-01 | 73/150 |
| random_forest | 0.6256 | 0.7012 | 0.6487 | 0.6548 | 0.6531 | −4.64 | 1.58e-23 | 9/150 |
| extra_trees | 0.6187 | 0.6869 | 0.6368 | 0.6489 | 0.6470 | −3.80 | 3.65e-23 | 14/150 |

Notes:
- xgboost and calibrated_lr present in CSV as baselines only (structural incompatibility with CWMS sample_weight)
- Shuffled = cwms_msbs_shuffled (OOF scores permuted within majority pool before use)
- hgb p = 0.091 → not significant at α=0.05; all other positive ΔBA rows are significant

---

## Table 1b — Shuffled-Score Ablation

Paired Wilcoxon: NoiSyn vs Shuffled, n=150 pairs each.

| Model | NoiSyn | Shuffled | ΔBA (pp) | p-value | n |
|-------|--------|----------|----------|---------|---|
| lr | 0.7394 | 0.7238 | +1.56 | 5.55e-08 | 150 |
| svm | 0.6776 | 0.6658 | +1.18 | 1.66e-11 | 150 |
| hgb | 0.6947 | 0.6842 | +1.05 | 2.14e-06 | 150 |
| lightgbm | 0.6945 | 0.6792 | +1.53 | 1.42e-07 | 150 |
| catboost | 0.6990 | 0.6806 | +1.84 | 1.95e-12 | 150 |

All five significant at α=0.05. OOF score ordering is load-bearing for all compatible model families.

---

## Table 2 — External Competitor Comparison

LR final model, hidden_minority_medium noise, 50 pairs (5 datasets × 10 seeds).

| Method | BA | G-mean | Minority F1 | Minority Prec | Minority Rec |
|--------|----|--------|-------------|---------------|--------------|
| no_cleaning | 0.5660 | 0.3097 | 0.2105 | 0.7583 | 0.1390 |
| smote | 0.6298 | 0.5050 | 0.3923 | 0.7528 | 0.2916 |
| class_proportional | 0.7032 | 0.6651 | 0.5549 | 0.6435 | 0.5016 |
| iw_smote | 0.7270 | 0.7149 | 0.5896 | 0.5583 | 0.6473 |
| sw_framework | 0.6516 | 0.5595 | 0.4505 | 0.7081 | 0.3499 |
| cwms_msbs (NoiSyn) | 0.7454 | 0.7428 | 0.6071 | 0.5312 | 0.7175 |

NoiSyn vs IW-SMOTE: +1.84 pp BA (p = 0.023, from paper; IW-SMOTE is strongest competitor with public code).
NoiSyn vs class_proportional: +4.22 pp BA in this medium-noise LR-only slice.

---

## Table 2b — Per-Dataset Breakdown (Table 2 conditions)

Balanced Accuracy per dataset, LR, medium noise.

| Dataset | No Cleaning | SMOTE | Class Prop. | IW-SMOTE | SW-approx | NoiSyn |
|---------|-------------|-------|-------------|----------|-----------|--------|
| credit-g | 0.5332 | 0.5761 | 0.6010 | 0.6466 | 0.5725 | 0.6373 |
| ecoli | 0.7082 | 0.7947 | 0.8643 | 0.8536 | 0.8202 | 0.8633 |
| phoneme | 0.5029 | 0.5455 | 0.6411 | 0.6648 | 0.5777 | 0.7501 |
| pima | 0.5315 | 0.5963 | 0.6723 | 0.7223 | 0.6129 | 0.7172 |
| yeast | 0.5542 | 0.6366 | 0.7373 | 0.7478 | 0.6748 | 0.7591 |

NoiSyn best on phoneme (+8.5 pp over IW-SMOTE), yeast (+1.1 pp), near-tied on ecoli.
IW-SMOTE best on pima (+0.5 pp over NoiSyn), credit-g (+0.9 pp over NoiSyn).

---

## Key Statistical Summary (5-dataset run, old methodology)

- LR: NoiSyn vs class_prop → +3.47 pp, Stouffer Z=7.30, p=1.5e-13, 3/5 sig datasets
- SVM: NoiSyn vs class_prop → +2.16 pp, Stouffer Z=5.01, p=2.8e-07, 2/5 sig datasets
- LightGBM: NoiSyn vs class_prop → +0.47 pp, Stouffer Z=2.23, p=1.3e-02, 1/5 sig datasets
- HGB: NoiSyn vs class_prop → +0.37 pp, Stouffer Z=1.80, p=3.6e-02, 2/5 sig datasets
- CatBoost: NoiSyn vs class_prop → −0.94 pp, Stouffer Z=−0.71, p=7.6e-01, 2/5 sig datasets
- RF: NoiSyn vs class_prop → −4.64 pp, Stouffer Z=−12.12, p=1.0e+00, 0/5 sig datasets
- ET: NoiSyn vs class_prop → −3.80 pp, Stouffer Z=−11.30, p=1.0e+00, 0/5 sig datasets
- NoiSyn vs IW-SMOTE (LR, medium, 5 datasets): +1.84 pp
- Shuffled ablation (Stouffer): LR Z=5.09, SVM Z=6.96, HGB Z=5.29

Note: old "p-values" from aggregate Wilcoxon on 150 i.i.d. pairs are DEPRECATED.
Current method: per-dataset Wilcoxon + Stouffer's Z (per CLAUDE.md statistical approach).

---

## Phase 2 Ablation Results (CONFIRMED, 2026-05-23)

### RF/ET Component Ablation (outputs/rfet-ablation-sweep.csv, 1500 rows)
(5 datasets × 10 seeds × 3 protocols × 5 methods × 2 models = 1500 rows)

| Model | Class Prop. | CWMS-only ΔBA | MSBS-only ΔBA | NoiSyn ΔBA | Stouffer Z |
|---|---|---|---|---|---|
| Random Forest | 0.7012 | −7.95 pp | −4.35 pp | −4.64 pp | −11.56 |
| Extra Trees | 0.6869 | −6.74 pp | −3.90 pp | −3.77 pp | −11.29 |

Primary harm source: CWMS (confidence-weighted suppression). MSBS causes secondary harm.

### Failure Mode Analysis (outputs/failure-mode-sweep.csv, 400 rows)
(5 datasets × 10 seeds × 2 protocols × 4 methods × 1 model(LR) = 400 rows)

| Protocol | Class Prop. BA | NoiSyn BA | ΔBA (pp) | Stouffer Z |
|---|---|---|---|---|
| symmetric (ε_mn=ε_mj=0.20) | 0.7318 | 0.7197 | −1.21 | −5.90 |
| reverse_asymmetric (ε_mn=0.02, ε_mj=0.30) | 0.7351 | 0.6330 | −10.21 | −15.36 |

Method is not designed for these regimes. reverse_asymmetric causes severe degradation.

### Clean-Data Ablation (outputs/clean-data-ablation.csv, 400 rows)
(5 datasets × 10 seeds × 4 methods × 2 models(LR+SVM) = 400 rows; zero noise)

| Model | Class Prop. BA | NoiSyn BA | ΔBA (pp) |
|---|---|---|---|
| Logistic Regression | 0.7341 | 0.7602 | +2.62 |
| Support Vector Machine | 0.7182 | 0.7383 | +2.01 |

NoiSyn improves even without noise — boundary-aware synthesis targeting class overlap region.

### IW-SMOTE λ Sensitivity (outputs/iw-lamda-sweep.csv)
λ=100 vs λ=30: ΔBA = −0.15 pp → keep λ=30 (gate passed, no change needed).

---

## Phase 5 Confirmed Results (2026-05-23, COMPLETE — 8,100 rows)

### Table 2 — Expanded External Comparison (3 models × 3 protocols × 15 datasets)

Mean BA across 450 pairs (15 ds × 10 seeds × 3 protocols):

| Method | LR BA | SVM BA | HGB BA |
|---|---|---|---|
| no_cleaning | 0.5996 | 0.5854 | 0.6514 |
| class_proportional | 0.7025 | 0.6729 | 0.6983 |
| smote | 0.6438 | 0.6376 | 0.6636 |
| iw_smote | 0.7270 | **0.7473** | **0.7296** |
| sw_framework | 0.6582 | 0.6555 | 0.6711 |
| **cwms_msbs (NoiSyn)** | **0.7341** | 0.6766 | 0.6977 |

### LR Stouffer Tests (cwms_msbs vs class_proportional, 15 datasets)

| Protocol | Δ (pp) | Stouffer Z | p | Sig. ds |
|---|---|---|---|---|
| low | +3.47 | 7.22 | 2.7e-13 | 10/15 |
| medium | +3.81 | 6.28 | 1.7e-10 | 10/15 |
| high | +2.21 | 2.90 | 1.8e-3 | 7/15 |
| **combined** | **+3.16** | **9.31** | **≈0** | **9/15** |

### LR Stouffer Tests (cwms_msbs vs each competitor, combined)

| Competitor | Δ (pp) | Z | p | Sig. ds |
|---|---|---|---|---|
| no_cleaning | +13.45 | 20.20 | ≈0 | 15/15 |
| class_proportional | +3.16 | 9.31 | ≈0 | 9/15 |
| smote | +9.03 | 18.54 | ≈0 | 14/15 |
| iw_smote | +0.71 | 1.17 | 0.12 | 3/15 |
| sw_framework | +7.59 | 16.91 | ≈0 | 13/15 |

Note: IW-SMOTE comparison NOT significant (p=0.12) across all 15 ds × 3 protocols. Under medium noise specifically: +3.81pp vs class_prop (significant). IW-SMOTE outperforms NoiSyn for SVM/HGB (model-agnostic oversample vs linear-boundary method).

## Phase 3-4 Status (as of 2026-05-23, sweeps running)

| Phase | Description | Output CSV | Target rows | Status |
|---|---|---|---|---|
| 3 | 15-dataset full benchmark | full-benchmark-solution-v2.csv | 24,750 | RUNNING (~24% done) |
| 4 | IR=0.30 sensitivity (5-ds pilot) | full-benchmark-ir030-solution.csv | 8,250 | RUNNING (~28% done) |
| 5 | Expanded competitor (LR+SVM+HGB×3 protos×15ds) | competitor-headtohead-expanded.csv | 8,100 | **COMPLETE** |
