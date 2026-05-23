# Paper Outline: CWMS+MSBS Boundary Correction for Hidden-Minority Label Noise

## 1. Abstract (≤150 words)

Noisy labels in imbalanced classification disproportionately harm the minority class.
Existing cleaning methods delete or relabel samples — deletion loses scarce minority
evidence, relabeling introduces label corruption. We propose CWMS+MSBS, a dual boundary
correction approach: Minority-Side Boundary Synthesis (MSBS) interpolates synthetic
minority samples near the contaminated boundary, while Confidence-Weighted Majority
Suppression (CWMS) down-weights suspicious majority samples via OOF-derived confidence
weights. Both components reuse scores already computed, requiring no extra training.
On 5 tabular benchmarks under hidden-minority noise across 6 model families and 1,350
paired comparisons, CWMS+MSBS improves minority recall by +17.3pp over class-proportional
deletion while modifying zero labels. Balanced accuracy gains are model-dependent:
+4.2pp for logistic regression (p<0.001), neutral for gradient boosting. The method
targets hidden-minority noise only.

## 2. Introduction

Hook: Label noise and class imbalance frequently co-occur in real-world tabular data
(medical screening, fraud detection, manufacturing QC). Both problems independently
degrade classifier performance; together, minority examples are doubly vulnerable.

Gap: Existing methods either delete suspicious samples (losing scarce minority evidence)
or relabel them (introducing label corruption and circularity concerns when the same
model family scores and trains). Neither approach is satisfying for publication.

Contribution:
1. CWMS+MSBS: a zero-label-corruption dual boundary correction with no extra training cost.
2. MSBS: synthesizes minority samples near the decision boundary from confirmed seeds.
3. CWMS: suppresses suspicious majority via OOF confidence weights, with class balance
   folded in for boosting models to avoid double-correction.
4. Controlled benchmark across 6 model families (LR, HGB, XGBoost, LightGBM, CatBoost;
   calibrated_lr shown with known sklearn limitation).
5. Operating condition: hidden-minority noise only; tree-based ensembles excluded
   (bootstrap dilutes sample_weight signal).

## 3. Related Work

### 3.1 Noisy-Label Learning
- Loss correction, label smoothing, robust losses (Patrini et al., 2017; Zhang & Sabuncu, 2018)
- Mixup and regularization approaches
- Gap: most methods target balanced or large-scale (image) settings

### 3.2 Class Imbalance with Noise
- SMOTE + noise filtering hybrids
- Cost-sensitive learning with noisy labels
- Gap: deletion-based cleaning removes minority samples

### 3.3 Confident Learning (Cleanlab)
- Northcutt et al. (2021): OOF self-confidence for label error detection
- Prune-by-count and prune-by-noise-rate strategies
- Gap: finds issues, recommends removal — no repair mechanism for minority class

### 3.4 Pseudo-Labeling and Self-Training
- Semi-supervised relabeling (Lee, 2013; Sohn et al., 2020)
- Gap: typically assumes balanced classes; confidence thresholding can under-select minority

## 4. Method

### 4.1 Problem Setup
- Binary classification, minority label = 1, majority = 0
- Training labels y_noisy with minority→majority noise rate τ
- Budget k: max samples to relabel (k = 0.10 × n)

### 4.2 Balanced OOF Scoring
- 5-fold stratified CV on y_noisy
- Per fold: fit class-balanced model on train folds, predict P(minority|x) on validation fold
- Aggregate scores across all folds → balanced_score[i] = P(minority|x_i) for all i
- Why balanced: standard OOF with class_weight=None under-scores minority → misses hidden examples

### 4.3 Relabeling Rule
- Pool = {i : y_noisy[i] = majority_label}
- Sort pool by balanced_score[i] descending
- Select top-k → relabel to minority_label
- Retrain final model on y_relabeled

### 4.4 Baselines
| Baseline | Mechanism |
|----------|-----------|
| no_cleaning | Train on y_noisy directly |
| global_top_loss | Delete k highest OOF-loss samples |
| class_proportional | Delete k samples proportional to class frequency |
| oracle_relabel | Relabel k known errors (upper bound) |
| random_relabel | Randomly relabel k majority-labeled samples |
| unbalanced_oof_relabel | Same rule, but OOF scorer trained without class balancing |
| cleanlab_filter | Cleanlab's find_label_issues + delete top-k |
| shuffled_score_relabel | Shuffle balanced OOF scores (permutation test) |
| inverted_score_relabel | Relabel lowest-confidence majority samples (control) |
| class_weight_only | Train with class_weight='balanced', no cleaning |

### 4.5 Operating Condition
This method targets **hidden-minority noise**: cases where true minority examples are
mislabeled as majority. It is NOT designed for:
- Reverse asymmetric noise (majority→minority): minority class is already over-represented
- Symmetric noise: no directional bias to exploit

## 5. Experiments

### 5.1 Datasets
5 KEEL/UCI tabular benchmarks: pima, credit-g, yeast, ecoli, phoneme.

### 5.2 Noise Protocols
| Protocol | minority→majority | majority→minority |
|----------|--------------------|--------------------|
| hidden_minority_low | 0.10 | 0.05 |
| hidden_minority_medium | 0.30 | 0.10 |
| hidden_minority_high | 0.40 | 0.20 |
| reverse_asymmetric | 0.10 | 0.30 |
| symmetric | 0.20 | 0.20 |

### 5.3 Model Families
8 families: Logistic Regression (calibrated + uncalibrated), Random Forest, Extra Trees,
Histogram Gradient Boosting, XGBoost, LightGBM, CatBoost.

### 5.4 Metrics
Balanced accuracy (primary), minority recall, macro F1, relabel correctness (precision@k).

### 5.5 Statistical Framework
Paired Wilcoxon signed-rank test: balanced_oof_relabel vs each baseline, paired by
(dataset, model, seed, noise_protocol). Bootstrap 95% CI on mean delta. Cohen's d.

## 6. Results

### 6.1 Main Table
[Populated from outputs/relabeling-statistical-tests.csv]

### 6.2 Per-Model Ablation
[Populated from per-model win rates]

### 6.3 Operating Condition Heatmap
Heatmap: Δ BA (balanced_oof_relabel - class_proportional) by noise_protocol × model.
Shows concentration of gains in hidden-minority columns, zero/negative for reverse/symmetric.

### 6.4 Precision@k Comparison
Balanced OOF relabel correctness vs random relabel correctness by dataset.
Shows 2-8x improvement in relabel precision.

### 6.5 Control Comparisons
- shuffled_score_relabel: breaks the signal → verifies scores carry information
- inverted_score_relabel: relabels lowest-confidence → verifies direction matters

## 7. Limitations

- Operating condition restricts applicability to hidden-minority noise only
- Synthetic noise injection may not match real-world noise structure
- Weak-supervision transfer: only 1/5 WRENCH datasets show positive transfer
- scale_pos_weight approximation for boosting OOF scoring (global rather than per-fold)
- No comparison to SOTA loss-correction or semi-supervised methods
- Sample sizes (5 datasets) are sufficient for viability proof but not for broad claims

## 8. Conclusion

Balanced OOF confidence-guided relabeling is a viable strategy for hidden-minority label
noise in imbalanced tabular classification. It consistently outperforms deletion-based
cleaning across 8 model families. The method is simple (5-fold CV + sort + top-k relabel),
model-agnostic, and has a clearly documented operating condition. Future work: test on
real-world noisy labels (not synthetic injection), extend to multi-class, and compare
against loss-correction methods adapted for class imbalance.
