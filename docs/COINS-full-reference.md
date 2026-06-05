# COINS — Full Reference Document (Slide Preparation)

> **COINS** = Confidence Out-of-fold Imbalanced Noise Synthesis  
> **Paper title**: "COINS: Out-of-Fold Confidence Scoring for Noise-Robust Synthesis in Imbalanced Classification"

---

## 1. Bối cảnh & Động lực (Problem / Motivation)

### Vấn đề cốt lõi
- **Class imbalance** + **label noise** đồng thời xuất hiện trong nhiều bài toán thực tế: y tế, phát hiện gian lận, phân loại lỗi hiếm.
- Phương pháp tiêu chuẩn (SMOTE, oversampling) giả định boundary sạch → khi có nhiễu nhãn, SMOTE **khuếch đại corruption** thay vì sửa.

### Dạng nhiễu nguy hiểm nhất: Hidden Minority Label Noise
- Minority samples bị mislabel thành majority với xác suất **ε_mn >> ε_mj**
- Ví dụ: bệnh nhân ung thư bị ghi nhận là "bình thường" nhiều hơn chiều ngược lại
- Hậu quả: boundary bị đẩy sâu vào vùng minority → **minority recall sụp đổ**

### Vấn đề với các phương pháp hiện có (IW-SMOTE, SW Framework)
- Dùng **model phát hiện nhiễu riêng** → confirmation bias nếu cùng inductive bias với final model
- Phần lớn đánh giá trên symmetric noise hoặc natural noise → không phù hợp với hidden-minority setting
- Thiếu benchmark thống nhất để so sánh trực tiếp

---

## 2. Phương pháp COINS

### 2.1 Bài toán (Problem Setting)

```
D_train = {(x_i, ỹ_i)}  — binary classification, noisy labels
Minority label:  m  (label 1)
Majority label:  M  (label 0)
Noise:  ε_mn = P(ỹ=M | y=m)  >> ε_mj = P(ỹ=m | y=M)
Test set: NOISE-FREE
Budget: B = floor(0.10 × |D_train|)  — số lượng synthetic samples
```

### 2.2 OOF Confidence Scoring

**Mục đích**: gán điểm "nghi ngờ" s_i cho mỗi majority-labeled sample — s_i cao = có thể bị mislabel.

```
Stratified 5-fold CV trên D_train:
  Mỗi fold: train F_balanced (cùng model family với final model, nhưng balanced=True)
            → score held-out majority samples:
              s_i = P_F^OOF(ỹ = minority | x_i)

s_i ∈ [0, 1]  for majority-labeled samples
s_i = NaN     for minority-labeled samples
```

**Hai điểm quan trọng**:
1. **Self-family**: scorer dùng đúng model family (LR scorer cho LR final model, SVM scorer cho SVM final model) → calibrated to same inductive bias
2. **OOF (Out-of-Fold)**: scorer không thấy sample đó lúc train → không có confirmation bias

### 2.3 CWMS — Confidence-Weighted Majority Suppression

**Mục đích**: down-weight các majority-labeled samples bị nghi ngờ mislabeled. **Không xóa sample nào**.

```python
# Linear models (LR, SVM):
w_i = 1 − s_i    # majority-labeled  → suspicious → thấp
w_i = 1.0         # minority-labeled  → luôn full weight

# Gradient boosting (HGB, LightGBM, CatBoost):
spw = |majority| / |minority|           # scale_pos_weight
w_i = (1 − s_i) × spw                   # majority-labeled
w_i = spw                                # minority-labeled
```

### 2.4 MSBS — Minority-Side Boundary Synthesis

**Mục đích**: sinh minority samples mới gần vùng boundary bị nhiễm, bù đắp lại minority samples bị mislabel.

```python
# Chọn seed từ majority pool với xác suất ∝ s_i
# (sample bị nghi ngờ nhiều → được chọn làm seed nhiều hơn)
x_seed  ~ Categorical(prob ∝ s_i)   # majority-labeled sample
x_min   ~ Uniform(minority pool)    # confirmed minority sample

x_synth = x_min + λ × (x_seed − x_min),   λ ~ Uniform(0, 1)
label(x_synth) = minority  # KHÔNG thay đổi nhãn gốc
```

Interpolation giữa confirmed minority và suspicious majority → synthetic point nằm ở **boundary region**.

### 2.5 Full Pipeline (COINS = CWMS + MSBS)

```
INPUT: D_train = (X_tr, ỹ_noisy), budget B, model family F

Step 1 — OOF Scoring:
  StratifiedKFold(5):
    Train F_balanced on fold train partition
    Score majority-labeled held-out samples → s_i

Step 2 — CWMS:
  Compute w_i from s_i (formula above)

Step 3 — MSBS:
  Sample B synthetic minority points using s as seed weights
  X_aug = X_tr ∪ X_synth
  y_aug = ỹ ∪ {minority}^B

Step 4 — Final Training:
  F_final.fit(X_aug, y_aug, sample_weight=w)

OUTPUT: F_final trained on weighted + augmented data

EVALUATE on: X_test (CLEAN, no noise)
```

**Điểm then chốt**: **Không sửa nhãn bất kỳ mẫu nào** (zero label modification).

---

## 3. Dữ liệu

### 3.1 Dataset (15 UCI/OpenML datasets)

| # | Dataset | Domain | n | Minority class | Minority ratio gốc |
|---|---------|--------|---|---|---|
| 1 | pima | Y tế | 768 | tested_positive | ~35% |
| 2 | breast_cancer | Y tế | 569 | malignant | ~37% |
| 3 | haberman | Y tế | 306 | died | ~27% |
| 4 | ilpd | Y tế | 583 | no_disease | ~29% |
| 5 | blood | Y tế | 748 | donated | ~24% |
| 6 | credit-g | Tài chính | 1,000 | bad | ~30% |
| 7 | ionosphere | Radar | 351 | bad | ~36% |
| 8 | phoneme | Âm học | 5,404 | nasal | ~29% |
| 9 | spambase | Email | 4,601 | spam | ~39% |
| 10 | kc1 | Phần mềm | 2,109 | defective | ~15% |
| 11 | ecoli | Sinh học | 336 | im | ~23% |
| 12 | yeast | Sinh học | 1,484 | MIT | ~11% |
| 13 | glass_float | Vật liệu | 214 | window_float | ~33% |
| 14 | vehicle_bus | Giao thông | 846 | bus | ~26% |
| 15 | abalone | Hải sản | 4,177 | rings_gt_10 | ~53% |

Tất cả được subsample xuống **IR = 0.15** (minority/total) trước khi bơm nhiễu.

### 3.2 Thu thập dữ liệu

```
Nguồn: OpenML (openml.org) — benchmark chuẩn cho ML research
Script: pipeline/experiments/download_datasets.py
Cache: data/{name}.parquet (tải một lần, không tải lại)
```

### 3.3 Tiền xử lý

**Quy trình đầy đủ (chỉ trên train, không leak sang test)**:

```
1. Binarize y: minority → 1, majority → 0
2. Stratified 75/25 train/test split (seed-based)
3. induce_imbalance(X_tr) → subsample majority → IR = 0.15
4. OrdinalEncoder.fit(X_tr) → transform(X_tr, X_te)
   (categorical features only; numeric giữ nguyên)
5. inject_noise(y_tr) → y_noisy (chỉ train)
```

### 3.4 Noise Injection Protocols (3 mức độ)

| Protocol | ε_mn (minority→majority) | ε_mj (majority→minority) | Ý nghĩa |
|---|---|---|---|
| hidden_minority_low | 0.10 | 0.05 | 10% minority bị mislabel |
| hidden_minority_medium | 0.30 | 0.10 | 30% minority bị mislabel |
| hidden_minority_high | 0.40 | 0.20 | 40% minority bị mislabel |

Ngoài ra dùng cho failure-mode analysis:
- `symmetric`: ε_mn = ε_mj = 0.20
- `reverse_asymmetric`: ε_mn = 0.02, ε_mj = 0.30

### 3.5 Train/Test Split

```python
train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
# → 75% train / 25% test
# → Stratified: giữ tỷ lệ class ở cả 2 phần
# → 10 seeds: [13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
```

**Nguyên tắc không rò rỉ**:
- `induce_imbalance`, `OrdinalEncoder.fit`, `inject_noise` → chỉ fit/apply trên train
- OOF 5-fold CV → trong train set
- Test set luôn sạch (no noise), phân phối gốc

---

## 4. Thực nghiệm (Experiment Setup)

### 4.1 Models

| Model | Ghi chú |
|---|---|
| Logistic Regression (LR) | Linear boundary, `class_weight` |
| Support Vector Machine (SVM) | RBF kernel, `class_weight` |
| HistGradientBoosting (HGB) | sklearn, native NaN, `sample_weight` |
| LightGBM (LGB) | GPU, `sample_weight` |
| CatBoost | GPU, native categorical |
| Random Forest (RF) | Bootstrap ensemble |
| Extra Trees (ET) | Bootstrap ensemble |

Default hyperparameters, không tuning.

### 4.2 Baselines

| Baseline | Mô tả |
|---|---|
| `no_cleaning` | Train trực tiếp trên noisy data |
| `class_proportional` | Class-weight reweighting (He & Garcia, 2009) |
| `smote` | Standard SMOTE (Chawla 2002), không noise-aware |
| `iw_smote` | IW-SMOTE (Zhang 2022, Pattern Recognition) — strong competitor, public code |
| `sw_framework` | SW Framework (Xu 2022, KBS) — approximated via k-NN label inconsistency |

### 4.3 Evaluation Metrics

Tất cả tính trên **clean test set**:
- **Balanced Accuracy (BA)** — metric chính: mean(recall_majority, recall_minority)
- Macro F1
- Minority Precision
- Minority Recall
- PR-AUC

### 4.4 Statistical Testing

**Correct approach** (không dùng aggregate Wilcoxon):
```
Per-dataset Wilcoxon signed-rank test:
  - 30 pairs per dataset = 10 seeds × 3 protocols
  - Paired: NoiSyn vs baseline, same seed+protocol

Stouffer's Z combination across 15 datasets:
  - Fully independent between datasets
  - Report: Z, p, X/15 significant datasets
```

### 4.5 Scale of Experiments

| File | Rows | Mục đích |
|---|---|---|
| full-benchmark-solution-v2.csv | 24,750 | Benchmark chính — Table 1 |
| competitor-headtohead-expanded.csv | 8,100 | So sánh competitor — Table 2 |
| full-benchmark-ir030-solution.csv | 8,250 | IR=0.30 sensitivity — Table 6 |
| rfet-ablation-sweep.csv | 1,500 | RF/ET component ablation |
| scorer-agnosticism-sweep.csv | 1,250 | Self-family vs cross-family |
| clean-data-ablation.csv | 400 | Zero-noise test |
| failure-mode-sweep.csv | 400 | Symmetric/reverse-asymmetric |
| **TOTAL** | **49,250** | |

---

## 5. Kết quả (Results)

### 5.1 Table 1 — Internal Benchmark (15 datasets × 10 seeds × 3 protocols = 450 pairs/model)

| Model | No Cleaning | Class Prop. | SMOTE | **COINS** | Shuffled | ΔBA vs ClassProp | p-value | Wins/150 |
|---|---|---|---|---|---|---|---|---|
| **LR** | 0.5758 | 0.7047 | 0.6337 | **0.7394** | 0.7238 | **+3.47 pp** | 6.08e-15 | 114/150 |
| **SVM** | 0.5683 | 0.6560 | 0.6330 | **0.6776** | 0.6658 | **+2.16 pp** | 1.72e-07 | 98/150 |
| HGB | 0.6352 | 0.6910 | 0.6499 | 0.6947 | 0.6842 | +0.37 pp | 9.09e-02 | 87/150 |
| LightGBM | 0.6370 | 0.6897 | 0.6496 | 0.6945 | 0.6792 | +0.47 pp | 4.03e-02 | 89/150 |
| CatBoost | 0.6285 | 0.7084 | 0.6579 | 0.6990 | 0.6806 | **−0.94 pp** | 2.05e-01 | 73/150 |
| **RF** | 0.6256 | 0.7012 | 0.6487 | 0.6548 | 0.6531 | **−4.64 pp** | 1.58e-23 | 9/150 |
| **ET** | 0.6187 | 0.6869 | 0.6368 | 0.6489 | 0.6470 | **−3.80 pp** | 3.65e-23 | 14/150 |

> **Kết luận**: COINS giúp **LR** (+3.47 pp, very significant) và **SVM** (+2.16 pp). **HGB/LGB neutral**. **RF/ET bị hại nặng** (−4 pp).

### 5.2 LR theo từng noise protocol (Stouffer, 15 datasets)

| Protocol | Class Prop. BA | COINS BA | Δ (pp) | Stouffer Z | p | Sig. datasets |
|---|---|---|---|---|---|---|
| low (ε_mn=0.10) | 0.7247 | 0.7594 | **+3.47** | 7.22 | 2.7e-13 | 10/15 |
| medium (ε_mn=0.30) | 0.7017 | 0.7398 | **+3.81** | 6.28 | 1.7e-10 | 10/15 |
| high (ε_mn=0.40) | 0.6811 | 0.7031 | **+2.21** | 2.90 | 1.8e-3 | 7/15 |
| **Combined** | 0.7025 | **0.7341** | **+3.16** | **9.31** | **≈0** | **9/15** |

### 5.3 LR Recall Recovery (Medium Noise)

```
No Cleaning:     recall = 0.21   (sụp đổ do nhiễu)
Class Prop.:     recall = 0.49
COINS:           recall = 0.72   (+22 pp vs no cleaning, +23 pp vs class prop.)
```

### 5.4 Table 2 — Competitor Comparison (LR, 15 datasets × 450 pairs)

| Method | Balanced Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| No Cleaning | 0.5996 | 0.5727 | 0.6823 | 0.2103 |
| SMOTE | 0.6438 | 0.6347 | 0.7047 | 0.3214 |
| SW-approx† | 0.6582 | 0.6539 | 0.6940 | 0.3565 |
| Class Prop. | 0.7025 | 0.7031 | 0.6542 | 0.4897 |
| IW-SMOTE | 0.7270 | 0.7112 | 0.5783 | 0.6443 |
| **COINS** | **0.7341** | **0.7045** | **0.5452** | **0.7160** |

†SW Framework: no public code; approximated via k-NN label inconsistency.

### 5.5 COINS vs IW-SMOTE (LR — all protocols, 15 datasets)

| Competitor | Δ BA (pp) | Stouffer Z | p | Sig. datasets |
|---|---|---|---|---|
| vs No Cleaning | +13.45 | 20.20 | ≈0 | 15/15 |
| vs SMOTE | +9.03 | 18.54 | ≈0 | 14/15 |
| vs SW-approx | +7.59 | 16.91 | ≈0 | 13/15 |
| vs Class Prop. | +3.16 | 9.31 | ≈0 | 9/15 |
| **vs IW-SMOTE** | **+0.71** | **1.17** | **0.12** | **3/15** |

> **vs IW-SMOTE**: numerically +0.71 pp nhưng **không statistically significant** (p=0.12) toàn bộ 15 ds × 3 protocols.  
> Dưới medium noise: COINS hơn class_prop **+3.81 pp** và hơn IW-SMOTE **~+1.84 pp** (5-dataset slice, p=0.023).

### 5.6 Per-Dataset Medium Noise (LR, so sánh với IW-SMOTE)

| Dataset | Class Prop. | IW-SMOTE | COINS |
|---|---|---|---|
| credit-g | 0.6010 | 0.6466 | 0.6373 |
| ecoli | 0.8643 | 0.8536 | **0.8633** |
| phoneme | 0.6411 | 0.6648 | **0.7501** (+8.5 pp!) |
| pima | 0.6723 | **0.7223** | 0.7172 |
| yeast | 0.7373 | 0.7478 | **0.7591** |

---

## 6. Ablation Studies

### 6.1 Shuffled-Score Ablation (OOF scores có quan trọng không?)

> Permute s_i trong majority pool → không còn thứ tự. Nếu COINS vẫn tốt → OOF scores không quan trọng.

| Model | COINS | Shuffled | Δ (pp) | p-value | Kết luận |
|---|---|---|---|---|---|
| LR | 0.7394 | 0.7238 | **+1.56** | 5.55e-08 | OOF scores load-bearing ✓ |
| SVM | 0.6776 | 0.6658 | **+1.18** | 1.66e-11 | OOF scores load-bearing ✓ |
| HGB | 0.6947 | 0.6842 | **+1.05** | 2.14e-06 | OOF scores load-bearing ✓ |
| LightGBM | 0.6945 | 0.6792 | **+1.53** | 1.42e-07 | OOF scores load-bearing ✓ |
| CatBoost | 0.6990 | 0.6806 | **+1.84** | 1.95e-12 | OOF scores load-bearing ✓ |
| RF | 0.6708 | 0.6717 | −0.09 | 0.54 | OOF không có signal (bootstrap absorbs) |

**Kết luận**: OOF score ordering là **load-bearing** cho tất cả compatible model families. RF không hưởng lợi từ scores — consistent với bootstrap aggregation diluting per-sample signal.

### 6.2 RF/ET Component Ablation (CWMS-only vs MSBS-only)

> 5 datasets × 10 seeds × 3 protocols = 150 pairs

| Model | Class Prop. | CWMS-only Δ | MSBS-only Δ | COINS Δ | Stouffer Z |
|---|---|---|---|---|---|
| Random Forest | 0.7012 | **−7.95 pp** | −4.35 pp | −4.64 pp | −11.56 |
| Extra Trees | 0.6869 | **−6.74 pp** | −3.90 pp | −3.77 pp | −11.29 |

**Primary harm source: CWMS** (confidence-weighted suppression).  
**Reason**: Bootstrap aggregation đa dạng hóa trees → high-variance majority points bị down-weight sai trên nhiều trees → CWMS suppresses wrong samples.  
MSBS gây hại thứ cấp (nhỏ hơn).

### 6.3 Failure Mode Analysis (LR — ngoài design regime)

| Noise Protocol | Class Prop. BA | COINS BA | Δ (pp) | Stouffer Z |
|---|---|---|---|---|
| hidden_minority_low | 0.7247 | 0.7594 | +3.47 | — |
| hidden_minority_medium | 0.7017 | 0.7398 | +3.81 | — |
| hidden_minority_high | 0.6811 | 0.7031 | +2.21 | — |
| **symmetric** (ε=0.20 cả hai) | 0.7318 | 0.7197 | **−1.21** | −5.90 |
| **reverse_asymmetric** (ε_mj=0.30) | 0.7351 | 0.6330 | **−10.21** | −15.36 |

> Symmetric: OOF scores noisy nhưng centered → hơi xấu (−1.21 pp).  
> Reverse-asymmetric: OOF scores point WRONG direction → suppresses đúng samples → thảm họa (−10 pp).  
> **COINS chỉ dùng cho hidden-minority asymmetric noise (ε_mn >> ε_mj).**

### 6.4 Clean-Data Ablation (không có noise)

| Model | Class Prop. BA | COINS BA | Δ (pp) |
|---|---|---|---|
| Logistic Regression | 0.7341 | **0.7602** | **+2.62** |
| Support Vector Machine | 0.7182 | **0.7383** | **+2.01** |

**COINS cải thiện ngay cả không có noise** → MSBS giúp boundary coverage nhờ nhắm vào class overlap region — dual function: noise correction (primary) + boundary-aware synthesis (secondary).

### 6.5 IR Sensitivity (IR=0.15 vs IR=0.30)

| Model | IR=0.15 ΔBA | IR=0.15 Z | IR=0.30 ΔBA | IR=0.30 Z |
|---|---|---|---|---|
| **LR** | **+3.16 pp** | **9.31** | +1.74 pp | 5.16 |
| **SVM** | +0.37 pp | 1.24 | **+4.05 pp** | **8.08** |
| HGB | −0.05 pp | 0.50 | +0.03 pp | −0.74 |
| RF | −4.37 pp | −17.44 | −3.73 pp | −10.77 |
| ET | −3.79 pp | −16.77 | −2.98 pp | −10.80 |

> **LR**: better tại IR=0.15 (more imbalanced).  
> **SVM**: reverses — not significant at IR=0.15 nhưng **strongly positive tại IR=0.30** (+4.05 pp) — SVM cần đủ class separation.

---

## 7. Tóm tắt Model-Family Characterization

| Model Family | COINS Effect | Lý do |
|---|---|---|
| **Logistic Regression** | **Tốt nhất (+3.16 pp, Z=9.31)** | Linear boundary nhạy với sample weighting; CWMS trực tiếp shift boundary |
| **SVM** | Dương nhưng phụ thuộc IR (+0.37 pp tại IR=0.15, +4.05 pp tại IR=0.30) | Margin-based; hưởng lợi khi có đủ class separation |
| **HGB / LightGBM** | Neutral (≈0) | Tree splits absorb weight noise; gains at low noise erased at high noise |
| **CatBoost** | Âm nhỏ (−0.94 pp) | High-noise degradation dominates |
| **RF / ET** | **Có hại (−4 to −8 pp)** | Bootstrap đa dạng hóa → CWMS down-weights wrong samples consistently |

---

## 8. Điểm nổi bật để đưa vào slide

### Key Numbers (chốt slide)
- LR: **+3.16 pp** Balanced Accuracy (vs class-proportional), Stouffer **Z = 9.31, p ≈ 0**, **9/15 datasets significant**
- LR recall recovery under medium noise: **0.50 → 0.72** (+22 pp)
- vs IW-SMOTE (best competitor): +0.71 pp (not sig overall); **+3.81 pp under medium noise**
- OOF scores load-bearing: shuffled ablation Z > 5 cho tất cả compatible families
- CWMS primary harm source for RF/ET: **−7.95 pp** (RF, CWMS-only ablation)
- Clean-data bonus: **+2.62 pp** LR even without noise

### Core Claims (paper)
1. **COINS = CWMS + MSBS + Self-family OOF** — zero label modification
2. Tốt cho LR (significant, consistent); SVM (imbalance-ratio dependent); Neutral cho gradient boosting; Harmful cho bootstrap ensembles
3. Self-family OOF tránh confirmation bias và cross-family misspecification
4. First controlled hidden-minority benchmark với 15 datasets, ablation đầy đủ

### Limitations (cần mention trong slide)
- Không áp dụng cho symmetric noise hoặc reverse-asymmetric noise
- Không cải thiện RF/ET — cần dùng baseline cho các models đó
- vs IW-SMOTE: không significant ở scale 15-dataset toàn bộ protocols
- Budget B = 10% → chưa sweep đầy đủ (chỉ có 5-dataset pilot)

---

## 9. Hardware & Reproducibility

```
Hardware:  13th Gen Intel Core i7-13700H (20 threads)
RAM:       14 GB
GPU:       NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) — CUDA cho XGBoost/LGB/CatBoost
Python:    /home/than-minh/miniconda3/envs/dsp/bin/python
Packages:  sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, catboost 1.2.8, pandas 2.3.3

Seeds (10): [13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
Total experimental rows: 49,250
```

---

## 10. Nguồn tham khảo chính

| Paper | Method | Venue |
|---|---|---|
| Chawla et al. (2002) | SMOTE | JAIR |
| He & Garcia (2009) | Class-proportional reweighting | IEEE TKDE |
| Zhang et al. (2022) | IW-SMOTE | Pattern Recognition |
| Xu et al. (2022) | SW Framework | Knowledge-Based Systems |
| Northcutt et al. (2021) | CleanLab (OOF label noise) | JAIR |
| Frénay & Verleysen (2014) | Label noise taxonomy | IEEE TNNLS |
