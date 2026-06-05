# COINS Problem Setup: Attack and Confirmation Note

Generated: 2026-06-01

Purpose: preserve the paper-safe wording for the COINS problem definition slide and record the main reviewer attacks before paper writing.

## Confirmed Slide Contract

Use this version in slides/paper:

```text
Main problem:
Imbalanced binary tabular classification under hidden minority-class label noise.
True minority samples are mislabeled as majority, so minority evidence is buried inside the majority class.

Input:
- Feature matrix X
- Noisy binary labels y_tilde
- Minority label m, majority label M
- Model family F
- Synthesis budget B, about 10% of the noisy training set
- Stratified 5-fold cross-validation setting

Core signal:
s_i = P_F^OOF(y_tilde = m | x_i)
for majority-labeled samples only.

High s_i means:
"This majority-labeled sample looks minority-like."

Output:
- Sample weights for training
- Synthetic minority samples near suspicious boundary regions
- Final classifier trained without label modification

Success criteria:
Improve balanced accuracy and minority recall under hidden-minority noise.
Secondary metrics: macro F1 and minority precision.

Scope:
Strongest evidence: Logistic Regression under confirmed hidden-minority asymmetric noise.
Conditional: SVM, depending on imbalance ratio.
Not intended for reverse-asymmetric noise or bootstrap ensembles.
```

## Confirmation Against Current Project

| Slide item | Verdict | Project evidence |
|---|---|---|
| Imbalanced binary tabular classification | Confirmed | README problem setup defines binary minority/majority labels and 15 tabular UCI/OpenML datasets. `README.md:48-50`, `README.md:114` |
| Hidden minority-class label noise | Confirmed | Project assumption is `epsilon_mn >> epsilon_mj`, minority -> majority noise. `README.md:9`, `README.md:29`, `README.md:50` |
| Feature matrix `X` | Confirmed with caveat | Code loads raw dataframe, induces imbalance, then encodes train/test into numeric arrays. `pipeline/experiments/run_relabeling_viability_sweep.py:171-182` |
| Noisy labels `y_tilde` | Confirmed | `inject_noise()` returns `y_noisy` and `noisy_mask`. `pipeline/data/loaders.py:105-125` |
| Minority/majority labels | Confirmed | Code sets minority as the smaller observed class and majority as the opposite binary label. `pipeline/experiments/run_relabeling_viability_sweep.py:171-187` |
| Model family `F` | Confirmed | `make_model_factory()` creates LR, SVM, HGB, RF, ET, etc.; paper claim narrows to LR/SVM. `pipeline/models/factories.py` |
| Budget `B` | Confirmed with wording change | README says `floor(0.10 * train)`, code uses `round(budget * len(y_noisy))`. Use "about 10%" in slides. `README.md:50`, `pipeline/experiments/run_relabeling_viability_sweep.py:187` |
| Stratified 5-fold CV | Confirmed | Balanced OOF scorer uses `StratifiedKFold(n_splits=5)`. `pipeline/scoring/balanced_oof.py:34`; call site uses 5. `pipeline/experiments/run_relabeling_viability_sweep.py:205-207` |
| Core score `P_F^OOF(y_tilde=m|x_i)` | Confirmed | `balanced_oof_majority_scores()` extracts minority probability for majority-labeled samples and sets minority-labeled samples to NaN. `pipeline/scoring/balanced_oof.py:21-25`, `pipeline/scoring/balanced_oof.py:44-50` |
| High score interpretation | Confirmed with caveat | High score is evidence a majority-labeled point looks minority-like. It may be mislabeled or just near class overlap/boundary. Use "minority-like", not "definitely mislabeled". |
| Sample weights output | Confirmed | CWMS assigns `1 - score` to majority-labeled samples, minority samples remain weight 1.0 for LR/SVM path. `pipeline/baselines/soft_weighting.py:12-31` |
| Synthetic minority samples output | Confirmed | MSBS interpolates confirmed minority samples toward high-score majority neighbors and labels generated points as minority. `pipeline/augmentation/msbs.py:23-32`, `pipeline/augmentation/msbs.py:45-71` |
| Final classifier without label modification | Confirmed | `cwms_msbs` trains on augmented data and sample weights; no relabeling in this path. `pipeline/experiments/run_relabeling_viability_sweep.py:352-380`; README states zero label corrections. `README.md:83`, `README.md:106` |
| Success metrics | Confirmed | Metrics include balanced accuracy, macro F1, minority recall, minority precision, and PR-AUC. `README.md:118`, `pipeline/evaluation/augment_metrics.py:70-87` |

## Attack Surface and Safe Answers

| Attack | Why it matters | Safe answer |
|---|---|---|
| High `s_i` does not prove mislabeling | Boundary-overlap majority samples can also receive high minority probability. | Correct. COINS treats `s_i` as minority-likeness/suspiciousness, not proof. This is why it suppresses and synthesizes without changing labels. |
| Existing methods already use OOF/confidence for label noise | Confident Learning uses confidence and pruning/ranking. | COINS does not claim OOF detection is new. Novelty is using self-family OOF confidence as one signal for both majority suppression and minority-side synthesis, with zero label modification. |
| Noise-robust SMOTE already exists | GK-SMOTE, SMOTE-LOF, IW-SMOTE, CRN-SMOTE weaken a broad novelty claim. | COINS should not say prior oversampling ignores noise. Say prior methods mostly use geometry, density, clustering, or ensemble disagreement; COINS uses self-family OOF label-confidence under hidden-minority asymmetric noise. |
| IW-SMOTE is very close and strong | COINS is only numerically ahead of IW-SMOTE for LR across all protocols, not statistically significant. | Do not claim general superiority over IW-SMOTE. Claim clear gains over no-cleaning, SMOTE, class-proportional, SW-approx; IW-SMOTE remains strongest competitor. COINS's advantage is targeted LR recall/boundary repair and simpler OOF-guided integration. |
| Robust-GBDT attacks the need for COINS | Robust-GBDT handles noisy + imbalanced tabular data at the loss level and fits the tabular SOTA story. | Frame Robust-GBDT as a competing loss-level direction, not support for COINS. COINS is data-level boundary reconstruction for hidden-minority signal, strongest for LR/linear workflows. |
| GBDT/tree papers make LR look weak | Tree methods are often tabular defaults. | COINS is not a universal tabular classifier improvement. It targets interpretable/linear or margin workflows where OOF boundary scores are high-signal. |
| RF/ET results are harmful | Method fails for bootstrap ensembles. | State this explicitly. It strengthens honesty. Claim boundary: not for RF/ET. |
| Reverse-asymmetric noise is severe failure | If noise direction is wrong, COINS suppresses the wrong evidence. | State precondition: use only when hidden-minority noise is plausible or verified. Include failure-mode analysis in paper. |
| Synthetic noise may not equal real annotation noise | Current benchmark injects controlled noise. | Treat natural label noise as future work and avoid claiming real-world deployment proof. |
| "No label modification" but synthetic labels are assigned | A reviewer may call synthetic minority labels label creation. | Say no original training labels are changed. Synthetic samples are assigned minority labels by construction. |

## Literature Positioning

Use these papers as follows:

| Paper family | Role in paper | Positioning |
|---|---|---|
| Confident Learning | Foundational support | Supports OOF confidence for label-quality estimation; COINS extends from detection/pruning to suppression + synthesis. |
| Label-noise surveys | Problem framing | Supports symmetric/asymmetric/instance-dependent noise taxonomy. |
| GK-SMOTE, SMOTE-LOF, IW-SMOTE, CRN-SMOTE | Direct competitors | Show noisy oversampling is active; COINS differs by self-family OOF confidence and hidden-minority focus. |
| Imbalanced learning survey | Context | Establishes imbalanced learning as active, broad, application-relevant area. |
| Robust-GBDT | Opposing/competing evidence | Present as strong loss-level alternative for tree-based workflows, not as direct support. |
| Demsar/statistical comparison | Evaluation rigor | Supports multi-dataset nonparametric testing; COINS uses per-dataset Wilcoxon plus Stouffer aggregation. |

External primary-source anchors checked:

- Northcutt et al. 2021, Confident Learning, JAIR. https://doi.org/10.1613/jair.1.12125
- Miraj et al. 2025, GK-SMOTE. https://arxiv.org/abs/2509.11163
- Zhang et al. 2022, IW-SMOTE. https://doi.org/10.1016/j.knosys.2022.108919
- Asniar et al. 2022, SMOTE-LOF. https://doi.org/10.1016/j.jksuci.2021.01.014
- Chen et al. 2024, imbalanced learning survey. https://doi.org/10.1007/s10462-024-10759-6
- Luo et al. 2025, Robust-GBDT. https://doi.org/10.1007/s10115-025-02595-z
- Demsar 2006, statistical comparisons over multiple data sets. https://www.jmlr.org/papers/v7/demsar06a.html

## Paper-Safe Claims

Use:

```text
COINS addresses hidden minority-class asymmetric label noise in imbalanced binary tabular classification by combining self-family OOF confidence scoring, confidence-weighted majority suppression, and minority-side boundary synthesis without modifying original labels.
```

Use:

```text
The OOF score is interpreted as minority-likeness among majority-labeled samples, not as ground-truth proof of label error.
```

Use:

```text
COINS is most defensible for Logistic Regression under confirmed hidden-minority noise. It is not a universal noisy-imbalanced tabular method.
```

Avoid:

```text
COINS solves noisy imbalanced classification.
```

Avoid:

```text
High OOF minority probability means the sample is mislabeled.
```

Avoid:

```text
COINS beats IW-SMOTE.
```

Replace with:

```text
COINS is numerically ahead of IW-SMOTE for LR in the full benchmark, but this difference is not statistically significant across all protocols and datasets; IW-SMOTE remains the strongest general-purpose competitor.
```

## Final Verdict

The slide contract is valid after three wording changes:

1. Use `y_tilde` in the score formula.
2. Say high score means "minority-like", not "likely mislabeled" unless qualified.
3. Say budget is "about 10%" because docs and implementation differ on floor vs round.

This wording is safe enough to reuse in the capstone deck and later paper introduction/method sections.

## Unresolved Questions

- Should paper title use COINS everywhere and retire NoiSyn from `docs/paper/paper-draft.md`?
- Should Robust-GBDT be added as a formal baseline or only discussed as competing literature?
- Should natural/noisy real datasets be added before submission, or left as future work?
