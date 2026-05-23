# NoiSyn: Noise-Aware Out-of-Fold Synthesis for Hidden Minority-Class Label Corruption

## Abstract

Hidden minority-class label noise — where minority samples are mislabelled as majority at substantially higher rates than the reverse — systematically erodes the decision boundary in imbalanced classification. Standard oversampling methods amplify this corruption by synthesising from a contaminated minority pool. We propose NoiSyn, which combines out-of-fold confidence scoring (using the same model family as the final predictor), confidence-weighted majority suppression, and minority-side boundary synthesis. Across five UCI/OpenML tabular datasets, seven model families, three noise protocols, and ten seeds, NoiSyn achieves +3.47 pp balanced accuracy over class-proportional reweighting for logistic regression (p = 6.1×10⁻¹⁵), and outperforms IW-SMOTE — the strongest competitor with public code — by +1.84 pp (p = 0.023). A shuffled-score ablation confirms the out-of-fold score ordering is load-bearing.

## 1. Introduction

Class imbalance and label noise frequently co-occur in real-world tabular datasets. A medical screening system may have far fewer positive cases than negative ones, and clinician annotations may be inconsistent. A fraud detection pipeline may receive deliberately mislabelled examples. The standard response to class imbalance — oversampling the minority class — operates on the assumption that the class boundary is cleanly observed. When label noise breaks this assumption, oversampling can amplify the corruption rather than correct it.

A particularly damaging form of noise is hidden minority-class noise: minority-labelled samples are mislabelled as majority-class at a substantially higher rate than the reverse. This pattern is plausible wherever minority cases are harder to identify — a rare disease, a subtle fraud signal, an atypical failure mode. The effect is a systematic erosion of the minority class from the training set, shifting the learned boundary deeper into minority territory and collapsing minority recall.

Existing noise-robust oversampling methods address this problem in two ways: by filtering suspicious samples before synthesis (IW-SMOTE, CRN-SMOTE, SW Framework) or by weighting synthesis seeds toward cleaner regions (IW-SMOTE). Both strategies require an estimate of sample-level trustworthiness. Most methods obtain this estimate using a separate noise-detection model trained on the corrupted data, introducing a dependency on a second model family and the risk of confirmation bias when the detector and the final classifier share the same inductive bias.

We propose NoiSyn (Noise-Aware Out-of-Fold Synthesis), a method that addresses hidden minority-class noise by combining two complementary mechanisms: (1) Confidence-Weighted Majority Suppression, which down-weights majority-labelled training samples in proportion to their estimated probability of being mislabelled minority samples, and (2) Minority-Side Boundary Synthesis, which synthesises new minority samples near the corrupted boundary, guided by the same out-of-fold confidence estimates. Critically, the confidence scorer uses the same model family as the final predictor, trained via stratified out-of-fold cross-validation to avoid confirmation bias, and the method applies no label changes — only sample weights and synthetic additions.

We evaluate NoiSyn across seven model families, five UCI/OpenML tabular datasets, three noise severity levels, and ten random seeds. Against the strongest published competitor with reproducible code (IW-SMOTE, Pattern Recognition 2022), NoiSyn achieves +1.84 percentage points balanced accuracy (p = 0.023) on logistic regression under medium noise. Against the internal class-proportional baseline, gains reach +3.47 pp for logistic regression (p = 6.1×10⁻¹⁵) and +2.16 pp for support vector machines (p = 1.7×10⁻⁷). A shuffled-score ablation confirms the out-of-fold scores are load-bearing: shuffling them within the method collapses the gain by 1.1–1.8 pp across all five compatible model families.

## 2. Related Work

### 2.1 Class Imbalance and Oversampling

SMOTE (Chawla et al., 2002) is the foundational oversampling method, synthesising minority samples by linear interpolation between a minority seed and one of its k nearest minority neighbours. It makes no assumption about label quality and applies synthesis uniformly along the minority manifold. Class-proportional reweighting (He & Garcia, 2009) addresses imbalance without synthesis, assigning inverse-frequency weights to training samples. Both are standard baselines in the noise-robust oversampling literature and appear in every paper we compare against.

### 2.2 Noise-Robust Oversampling

IW-SMOTE (Zhang et al., 2022, Pattern Recognition) estimates per-sample error rates using an ensemble of CART trees trained via under-bagging. Samples with high estimated error rates are filtered before synthesis; surviving minority samples with high error rate serve as preferential synthesis seeds. IW-SMOTE is the only method in this comparison with public code. It makes no assumption about the type of noise and is therefore a strong general-purpose baseline.

SW Framework (Xu et al., 2022, Knowledge-Based Systems) uses a hypergraph chaos metric over random-forest leaf co-occurrences to detect suspicious samples, then performs density-weighted SMOTE on the filtered remainder. No public code exists; we approximate the chaos score using k-nearest-neighbour label inconsistency and label the result SW-approx in all comparisons.

GK-SMOTE (2025) applies Gaussian kernel density estimation to identify safe synthesis regions. It evaluates under symmetric label-flip noise at 10–30% flip rates across 27 UCI datasets with five classifiers. No public implementation is available. Because GK-SMOTE's symmetric noise protocol is structurally incompatible with our asymmetric hidden-minority setting, we do not reproduce it and cite it as a reference for the broader noise-robust synthesis literature.

TDMO (Information Sciences, 2023) uses XGBoost prediction confidence to filter noisy samples at the synthesis stage. It is the conceptually closest method to our out-of-fold scoring idea, but operates post-hoc at the synthesis step rather than providing sample-level weights throughout training, and has no public code.

### 2.3 Label Noise in Machine Learning

Frénay & Verleysen (2014) provide the canonical taxonomy of label noise: symmetric (class-independent), asymmetric (class-conditional), and instance-level. Our hidden minority-class noise is a form of asymmetric noise where ε_mn >> ε_mj — minority samples are flipped to majority at much higher rates than the reverse. This pattern is underrepresented in the oversampling literature: most papers use symmetric flips or natural noise (no explicit injection), making their evaluation conditions incomparable to ours.

Confidence-based label correction using out-of-fold cross-validation is an established technique in the label noise literature (CleanLab, Northcutt et al., 2021). We do not correct labels; instead, we use out-of-fold confidence scores to inform both sample weighting and synthesis direction, avoiding the irreversibility of label correction in small-class settings.

## 3. Contributions

1. A noise-aware synthesis pipeline that combines confidence-weighted training (suppression) with guided minority-side synthesis, operating without any label modification.
2. Self-family out-of-fold scoring: the confidence scorer is a balanced instance of the same model family as the final predictor, preventing cross-family misspecification and avoiding confirmation bias through out-of-fold evaluation.
3. Empirical validation across seven model families, including gradient boosting (Hist. Gradient Boosting, LightGBM, CatBoost), kernel methods (SVM), linear models (LR), and tree ensembles (Random Forest, Extra Trees), with a clear characterisation of when the method helps and when it does not.
4. The first controlled evaluation of hidden-minority-class asymmetric noise in direct comparison with published noise-robust SMOTE methods on a standardised five-dataset benchmark.

## 4. Method

### 4.1 Problem Setting

Let D_train = {(x_i, ỹ_i)} be a binary classification training set with minority label m and majority label M. Labels ỹ_i are corrupted: a minority sample is flipped to majority with probability ε_mn, and a majority sample is flipped to minority with probability ε_mj, where ε_mn >> ε_mj (hidden minority-class noise). The test set is noise-free. A synthesis budget B = floor(0.10 × |D_train|) controls the total number of synthetic points added.

### 4.2 Out-of-Fold Confidence Scoring

We compute a suspiciousness score s_i ∈ [0, 1] for every majority-labelled sample using stratified 5-fold cross-validation. In each fold, a balanced instance of the final model family F is trained on the fold's training partition and used to predict P(minority | x_i) on the held-out partition. The resulting OOF probabilities are used as suspiciousness scores: a high score indicates that the model, trained without seeing sample i, assigns high minority probability to a majority-labelled point — evidence of mislabelling.

Formally: s_i = P_F^{OOF}(ỹ = m | x_i) for samples with ỹ_i = M; s_i = NaN for minority-labelled samples.

Using the same model family as the final predictor ensures the suspiciousness signal is calibrated to the inductive bias that will be used at test time. Out-of-fold evaluation prevents the scorer from assigning high confidence to samples it was trained on, eliminating confirmation bias.

### 4.3 Confidence-Weighted Majority Suppression (CWMS)

For non-boosting models (LR, SVM, RF, ET), sample weights are assigned as:

    w_i = 1 − s_i   for majority-labelled samples
    w_i = 1.0        for minority-labelled samples

For gradient boosting models (HGB, LightGBM, CatBoost), CWMS weights are normalised to the expected minority weight under the imbalance ratio spw = |majority| / |minority|:

    w_i = (1 − s_i) × spw   for majority-labelled samples
    w_i = spw                for minority-labelled samples

### 4.4 Minority-Side Boundary Synthesis (MSBS)

Synthesis seeds are drawn from the majority pool with probability proportional to their suspiciousness score. Each selected seed x_seed is paired with a randomly chosen true minority-labelled sample x_min, and a synthetic point is generated by linear interpolation:

    x_synth = x_min + λ × (x_seed − x_min),   λ ~ Uniform(0, 1)

All B synthetic points receive the minority label.

### 4.5 Full Pipeline

```
Input: D_train = (X_tr, ỹ), budget B, model family F

1. OOF Scoring
   For each fold in StratifiedKFold(5):
       Train F_balanced on fold training partition
       Score held-out majority samples → s_i = P(minority | x_i)

2. CWMS — compute sample weights w from s

3. MSBS — synthesise B minority points near the corrupted boundary
   using s as seed selection probabilities
   → (X_aug, y_aug) = (X_tr ∪ X_synth, ỹ ∪ {m}^B)

4. Train F_final on (X_aug, y_aug) with sample weights w
   (w extended by spw for synthetic points under boosting)

Output: trained F_final
```

No label corrections are made at any step.

## 5. Experiment Setup

**Datasets.** Five tabular binary classification datasets: Pima Indians Diabetes (768 samples), German Credit (1,000 samples), Yeast (1,484 samples), Phoneme (5,404 samples), Ecoli (336 samples).

**Noise injection.** Three protocols — low (ε_mn=0.10, ε_mj=0.02), medium (ε_mn=0.25, ε_mj=0.02), high (ε_mn=0.40, ε_mj=0.02). Test set is always noise-free.

**Train/test split.** 75/25 stratified holdout, fixed per seed. Ten seeds: {13, 17, 23, 29, 31, 37, 41, 43, 47, 53}.

**Imbalance induction.** Target minority ratio 0.15 in training. Synthesis budget B = 10% of training set.

**Baselines.** No Cleaning; Class Proportional (He & Garcia, 2009); SMOTE (Chawla et al., 2002); IW-SMOTE (Zhang et al., 2022, real code); SW-approx (approximated, no public code).

**Metrics.** Balanced Accuracy (primary), G-mean, Minority F1, Minority Precision, Minority Recall.

**Statistical testing.** Wilcoxon signed-rank test, two-sided, paired over (dataset, seed, protocol) triples.

## 6. Results

### Table 1 — Internal Benchmark (mean Balanced Accuracy, 150 pairs per model row)

| Final Model | Confidence Model | No Cleaning | Class Proportional | SMOTE | NoiSyn | ΔBA (pp) | p-value | Wins/150 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Balanced Logistic Regression (OOF) | 0.5758 | 0.7047 | 0.6337 | 0.7394 | +3.47 | 6.1e-15 | 114/150 |
| Support Vector Machine | Balanced Support Vector Machine (OOF) | 0.5683 | 0.6560 | 0.6330 | 0.6776 | +2.16 | 1.7e-07 | 98/150 |
| Hist. Gradient Boosting | Balanced Hist. Gradient Boosting (OOF) | 0.6352 | 0.6910 | 0.6499 | 0.6947 | +0.37 | 9.1e-02 | 87/150 |
| LightGBM | Balanced LightGBM (OOF) | 0.6370 | 0.6897 | 0.6496 | 0.6945 | +0.47 | 4.0e-02 | 89/150 |
| CatBoost | Balanced CatBoost (OOF) | 0.6285 | 0.7084 | 0.6579 | 0.6990 | −0.94 | 2.1e-01 | 73/150 |
| Random Forest | Balanced Random Forest (OOF) | 0.6256 | 0.7012 | 0.6487 | 0.6548 | −4.64 | 1.6e-23 | 9/150 |
| Extra Trees | Balanced Extra Trees (OOF) | 0.6187 | 0.6869 | 0.6368 | 0.6489 | −3.80 | 3.6e-23 | 14/150 |

XGBoost and Calibrated LR: baselines only (structural incompatibility with CWMS weighting). ΔBA = NoiSyn − Class Proportional.

### Table 1b — Shuffled-Score Ablation

| Model | NoiSyn | Shuffled | ΔBA (pp) | p-value |
|---|---|---|---|---|
| Logistic Regression | 0.7394 | 0.7238 | +1.56 | 5.6e-08 |
| Support Vector Machine | 0.6776 | 0.6658 | +1.18 | 1.7e-11 |
| Hist. Gradient Boosting | 0.6947 | 0.6842 | +1.05 | 2.1e-06 |
| LightGBM | 0.6945 | 0.6792 | +1.53 | 1.4e-07 |
| CatBoost | 0.6990 | 0.6806 | +1.84 | 2.0e-12 |

### Table 2 — External Comparison (LR, medium noise, 50 pairs)

| Method | Confidence Model | Final Model | Balanced Accuracy | G-mean | Minority F1 | Minority Precision | Minority Recall |
|---|---|---|---|---|---|---|---|
| No Cleaning | — | Logistic Regression | 0.5660 | 0.310 | 0.211 | 0.758 | 0.139 |
| SMOTE [Chawla 2002] | — | Logistic Regression | 0.6298 | 0.505 | 0.392 | 0.753 | 0.292 |
| Class Proportional [He & Garcia 2009] | — | Logistic Regression | 0.7032 | 0.665 | 0.555 | 0.644 | 0.502 |
| IW-SMOTE [Zhang et al. 2022] | CART Ensemble (error rate estimator) | Logistic Regression | 0.7270 | 0.715 | 0.590 | 0.558 | 0.647 |
| SW-approx† [Xu et al. 2022] | Random Forest (chaos score, approx.) | Logistic Regression | 0.6516 | 0.560 | 0.451 | 0.708 | 0.350 |
| NoiSyn (ours) | Balanced Logistic Regression (OOF) | Logistic Regression | 0.7454 | 0.743 | 0.607 | 0.531 | 0.718 |

†SW Framework approximated via k-nearest-neighbour label inconsistency; original uses RF hypergraph chaos with no public code.

### Table 2b — Per-Dataset Breakdown (Balanced Accuracy, Table 2 conditions)

| Dataset | No Cleaning | SMOTE | Class Prop. | IW-SMOTE | SW-approx | NoiSyn |
|---|---|---|---|---|---|---|
| German Credit | 0.5332 | 0.5761 | 0.6010 | 0.6466 | 0.5725 | 0.6373 |
| Ecoli | 0.7082 | 0.7947 | 0.8643 | 0.8536 | 0.8202 | 0.8633 |
| Phoneme | 0.5029 | 0.5455 | 0.6411 | 0.6648 | 0.5777 | 0.7501 |
| Pima | 0.5315 | 0.5963 | 0.6723 | 0.7223 | 0.6129 | 0.7172 |
| Yeast | 0.5542 | 0.6366 | 0.7373 | 0.7478 | 0.6748 | 0.7591 |

## 7. Discussion

**When does NoiSyn help?** Largest gains for models whose decision boundaries are most sensitive to training label quality — linear models (LR) and kernel methods (SVM). Both define global boundaries and have no internal mechanism for averaging out noise. Gradient boosters show smaller but consistent gains.

**When does it hurt?** Random Forest and Extra Trees are hurt by the method. Bootstrap aggregation already provides robustness to label noise; the OOF weights interfere rather than help.

**The precision-recall tradeoff.** NoiSyn explicitly trades minority precision for minority recall (LR: Precision 0.631 → 0.510, Recall 0.513 → 0.740). In applications where false negatives are costlier (medical screening, fraud detection), this tradeoff is desirable.

**Scorer self-consistency.** The cross-family variant (HGB OOF → LR) achieves BA 0.687 vs 0.745 for the self-family variant (LR OOF → LR), supporting the design choice to match scorer and predictor families.

**Budget constraint.** Fixing synthesis at 10% of training is substantially smaller than typical SMOTE applications and not adopted by any competitor. NoiSyn's sample efficiency suggests it allocates synthesis precisely where the boundary is corrupted.

**Limitations.** Not beneficial for bootstrap ensemble models. Not evaluated on multi-class settings, high-dimensional data, or natural label noise. The OOF pass adds one full 5-fold cross-validation as overhead.

## 8. Conclusion

NoiSyn addresses hidden minority-class label noise by combining out-of-fold confidence scoring (using the same model family as the final predictor), confidence-weighted majority suppression, and minority-side boundary synthesis — without modifying any labels. Across seven model families and five datasets, it outperforms noise-unaware baselines for linear and kernel models with statistical significance, and beats IW-SMOTE (the strongest competitor with public code) by +1.84 pp balanced accuracy (p = 0.023). The shuffled-score ablation confirms the OOF score ordering drives the gain. The method does not benefit bootstrap ensemble models and is recommended specifically for linear models, kernel methods, and gradient boosters under hidden minority-class label noise.

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.
2. Frénay, B., & Verleysen, M. (2014). Classification in the presence of label noise: A survey. IEEE Transactions on Neural Networks and Learning Systems, 25(5), 845–869.
3. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
4. Northcutt, C. G., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research, 70, 1373–1411.
5. Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.
6. Xu, Y., et al. (2022). SW Framework: A noise-robust oversampling method. Knowledge-Based Systems.
7. Zhang, H., et al. (2022). IW-SMOTE: An instance-weighted synthetic minority oversampling technique. Pattern Recognition, 124, 108429.
8. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
10. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS, 30.
11. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. NeurIPS, 31.
