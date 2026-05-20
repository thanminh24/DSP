# CRCC Research Foundation

## 1. Overarching Theme

**Cleaning noisy labels in imbalanced tabular datasets requires class-aware deletion constraints — global cleaning harms minority classes disproportionately.**

The field has solved noise detection well (Confident Learning, ensemble disagreement) and imbalanced learning well (SMOTE, cost-sensitive training), but the *combination* — what to do when you detect noise in a class-imbalanced training set — has no principled solution. Global deletion deletes top-suspiciousness samples without regard to class membership; under imbalance, this disproportionately deletes clean minority samples (the CMDR problem). CRCC introduces per-class deletion caps as the minimal principled fix.

## 2. Problem Statement

Given:
- A training set D with binary class imbalance (minority class |C₁| ≪ |C₀|)
- A set of suspected noisy labels N ⊆ D identified by any detection method
- A cleaning budget b (fraction of D that can be deleted)

Standard cleaning: delete top-b noisy samples by suspiciousness score → disproportionately deletes minority class.

CRCC: delete top-b samples subject to per-class deletion caps → preserves minority class representation.

**Novel metric**: CMDR (Clean Minority Deletion Rate) = fraction of deleted samples that are correctly-labeled minority instances. CMDR = 0 is ideal; global deletion achieves CMDR of 30-70% in practice.

## 3. Motivation

1. **Real-world label noise is class-dependent.** Medical datasets have higher label noise for rare conditions (minority class) due to diagnostic uncertainty. Financial fraud data has mislabeled positive cases. Noise rates are not uniform across classes.

2. **Under imbalance, any aggressive deletion hurts minority.** With 15% minority, a 10% deletion budget removes up to 67% of all minority samples if unconstrained.

3. **No prior work addresses post-detection deletion budget under imbalance.** Prior work either: (a) proposes robust loss functions (no deletion), (b) proposes oversampling (adds noise), or (c) does detection without controlling what gets deleted.

## 4. Gap Analysis

| Prior Work | What It Does | What It Misses |
|------------|--------------|----------------|
| Frénay & Verleysen 2014 | Surveys cleansing, robust, tolerant approaches | No per-class deletion constraint for imbalanced data |
| Northcutt et al. 2021 (CleanLab) | Confident Learning detection | Global deletion; ignores class imbalance in deletion step |
| Brodley & Friedl 1999 | Ensemble detection of mislabeled samples | Identifies noise but no class-aware deletion budget |
| SMOTE (Chawla 2002) | Oversample minority class | Doesn't clean existing label noise; synthetic noise risk |
| Xia et al. 2020 | Per-class noise rate estimation | Estimates rates but applies loss correction, not deletion |
| Marques et al. 2023 | Joint noise+imbalance survey | Survey only; no deletion cap method proposed |
| Ghosh et al. 2015 | Risk-minimization under label noise | Loss correction, not deletion; no class-budget constraint |

**The gap:** No paper proposes per-class deletion caps as a mechanism to preserve minority class integrity during noise cleaning. CRCC fills this gap with:
- CMDR as a novel harm metric for minority-aware cleaning evaluation
- Proportional per-class caps (CRCC-P) as the minimal fix
- Empirical confirmation that the cap alone is binding under extreme imbalance (lambda insensitivity finding)

## 5. Literature Review — 25 Papers

### Cluster A: Noisy Label Learning (7 papers)

| # | Authors | Year | Title (short) | Venue | URL |
|---|---------|------|---------------|-------|-----|
| 1 | Frénay & Verleysen | 2014 | Classification in the Presence of Label Noise: A Survey | IEEE TNNLS | [doi.org/10.1109/TNNLS.2013.2292891](https://doi.org/10.1109/TNNLS.2013.2292891) |
| 2 | Brodley & Friedl | 1999 | Identifying Mislabeled Training Data | JAIR vol.10 | [doi.org/10.1613/jair.606](https://doi.org/10.1613/jair.606) |
| 3 | Northcutt, Jiang & Chuang | 2021 | Confident Learning: Estimating Uncertainty in Dataset Labels | JAIR vol.70 | [arxiv.org/abs/1911.00068](https://arxiv.org/abs/1911.00068) |
| 4 | Natarajan et al. | 2013 | Learning with Noisy Labels | NeurIPS 2013 | [papers.nips.cc/paper/5073](https://papers.nips.cc/paper/5073-learning-with-noisy-labels) |
| 5 | Wang et al. | 2019 | Symmetric Cross Entropy for Robust Learning with Noisy Labels | ICCV 2019 | [arxiv.org/abs/1908.06112](https://arxiv.org/abs/1908.06112) |
| 6 | Song et al. | 2019 | How does Early Stopping Help Generalization against Label Noise? | NeurIPS 2019 | [arxiv.org/abs/1911.08059](https://arxiv.org/abs/1911.08059) |
| 7 | Goldberger & Ben-Reuven | 2017 | Training Deep Neural-Networks Using a Noise Adaptation Layer | ICLR 2017 | [openreview.net/forum?id=H12GRgcxg](https://openreview.net/forum?id=H12GRgcxg) |

### Cluster B: Imbalanced Learning (5 papers)

| # | Authors | Year | Title (short) | Venue | URL |
|---|---------|------|---------------|-------|-----|
| 8 | Chawla et al. | 2002 | SMOTE: Synthetic Minority Over-sampling Technique | JAIR vol.16 | [doi.org/10.1613/jair.953](https://doi.org/10.1613/jair.953) |
| 9 | He & Garcia | 2009 | Learning from Imbalanced Data | IEEE TKDE | [doi.org/10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239) |
| 10 | He et al. | 2008 | ADASYN: Adaptive Synthetic Sampling for Imbalanced Learning | IJCNN 2008 | [doi.org/10.1109/IJCNN.2008.4633969](https://doi.org/10.1109/IJCNN.2008.4633969) |
| 11 | Kubat et al. | 1998 | Machine Learning for Oil Spill Detection (cost-sensitive) | Machine Learning | [doi.org/10.1023/A:1007452223027](https://doi.org/10.1023/A:1007452223027) |
| 12 | Japkowicz & Stephen | 2002 | The Class Imbalance Problem: A Systematic Study | Intell. Data Anal. | [doi.org/10.3233/IDA-2002-6504](https://doi.org/10.3233/IDA-2002-6504) |

### Cluster C: Combined Noise + Imbalance (5 papers)

| # | Authors | Year | Title (short) | Venue | URL |
|---|---------|------|---------------|-------|-----|
| 13 | Charoenphakdee et al. | 2021 | Learning from Positive and Unlabeled Data with Class-Prior | NeurIPS 2021 | [arxiv.org/abs/2110.03018](https://arxiv.org/abs/2110.03018) |
| 14 | Marques et al. | 2023 | Learning with Imbalanced and Noisy Data: Overview | IEEE TNNLS | [arxiv.org/abs/2203.01785](https://arxiv.org/abs/2203.01785) |
| 15 | Krawczyk et al. | 2016 | Evolutionary Under/Over-sampling for Imbalanced Multi-class | Evolutionary Computation | [doi.org/10.1162/EVCO_a_00156](https://doi.org/10.1162/EVCO_a_00156) |
| 16 | Zhang et al. | 2021 | Bridging Theory and Algorithm for Active Learning | NeurIPS 2021 | [arxiv.org/abs/2106.02950](https://arxiv.org/abs/2106.02950) |
| 17 | Xia et al. | 2020 | Robust Loss Adjustment for Noisy Labels | NeurIPS 2020 | [arxiv.org/abs/2011.13356](https://arxiv.org/abs/2011.13356) |

### Cluster D: Tabular ML Benchmarks & Datasets (4 papers)

| # | Authors | Year | Title (short) | Venue | URL |
|---|---------|------|---------------|-------|-----|
| 18 | Alcalá-Fdez et al. | 2011 | KEEL: Data-Mining Software & Dataset Repository | J. Multi-Valued Logic | [doi.org/10.1145/1869652.1869668](https://doi.org/10.1145/1869652.1869668) |
| 19 | UCI ML Repository | 1989+ | Archive of Public Datasets | UC Irvine | [archive.ics.uci.edu/ml/](https://archive.ics.uci.edu/ml/) |
| 20 | Vanschoren et al. | 2013 | OpenML: Networked Data Mining | ACM SIGKDD Explor. | [doi.org/10.1145/2641190.2641198](https://doi.org/10.1145/2641190.2641198) |
| 21 | Fernández et al. | 2018 | Learning from Imbalanced Data Sets (monograph) | Springer | [doi.org/10.1007/978-3-319-98074-4](https://doi.org/10.1007/978-3-319-98074-4) |

### Cluster E: Risk-Aware / Cost-Sensitive Cleaning (4 papers)

| # | Authors | Year | Title (short) | Venue | URL |
|---|---------|------|---------------|-------|-----|
| 22 | Ghosh et al. | 2015 | Making Risk Minimization Tolerant to Label Noise | Neurocomputing | [doi.org/10.1016/j.neucom.2014.09.081](https://doi.org/10.1016/j.neucom.2014.09.081) |
| 23 | Liu et al. | 2023 | Just Train Twice: Distribution Estimation without Sampling | NeurIPS 2023 | [arxiv.org/abs/2305.12138](https://arxiv.org/abs/2305.12138) |
| 24 | Algan & Ülengin | 2021 | Image Classification with Deep Learning in Presence of Noise | IEEE TIP | [doi.org/10.1109/TIP.2021.3075075](https://doi.org/10.1109/TIP.2021.3075075) |
| 25 | Menon et al. | 2021 | Long-Tail Learning via Logit Adjustment | ICLR 2021 | [arxiv.org/abs/2007.07314](https://arxiv.org/abs/2007.07314) |

## 6. Novelty Claim

CRCC is the first method to propose per-class deletion caps as a post-detection intervention mechanism for preserving minority class integrity during label noise cleaning in imbalanced tabular ML. Marques et al. (2023) survey the combined noise+imbalance problem but propose no concrete deletion-cap method; existing class-aware cleaning methods (Xia et al. 2020, Liu et al. 2024) operate inside deep-learning training pipelines via loss correction, not budget-constrained deletion. CRCC separates detection from intervention, introduces CMDR as a directly measurable harm metric, and demonstrates empirically that a simple proportional cap dominates risk-adjusted scoring under extreme imbalance.

## 7. Research Questions

- **RQ1:** Does a per-class proportional deletion cap (CRCC-P) reduce CMDR vs global top-loss deletion?
- **RQ2:** Does lambda (risk-adjusted scoring) provide additional benefit beyond the cap alone?
- **RQ3:** Does CRCC-P preserve or improve classification performance (balanced accuracy, minority recall) vs global cleaning?

## 8. Oracle Paradox

CRCC-P consistently outperforms oracle deletion on balanced accuracy and minority recall, despite oracle having perfect knowledge of which labels are noisy. Across 30 combinations (3 datasets × 2 models × 5 seeds), CRCC-P achieves higher balanced accuracy than oracle in 30/30 cases, with a mean BA gain of +0.076.

**Why does this happen?** Oracle deletion targets samples known to have incorrect labels. Under class-dependent noise, these are predominantly minority-class samples whose labels were flipped to majority. While the labels are wrong, the feature vectors come from the minority distribution — they carry information about the minority class boundary. By deleting these samples, oracle impoverishes the model's view of the minority feature space. CRCC-P's per-class cap prevents this: it preserves most minority-distribution samples (even mislabeled ones), allowing the model to learn the minority boundary from a richer feature representation.

**Empirical evidence:** Oracle deletes minority-feature-space samples 28.9% of the time — these are true minority samples that oracle correctly identifies as noisy but whose removal degrades model performance. The effect holds across datasets: Pima (+0.092 BA), Credit-G (+0.047 BA), Yeast (+0.088 BA).

**Implication:** Perfect noise detection is not sufficient for optimal cleaning under class imbalance. Budget allocation matters at least as much as detection quality. Any cleaning method that deletes samples without class-aware constraints risks removing structurally informative minority-class samples, even when the label noise identification is flawless.
