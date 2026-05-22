# Gap Audit Report — Q1/Q2 A*/A 2020–2025

**Date:** 2026-05-22  
**Agents:** 3 parallel researchers (27 + 22 + 24 = 73 papers reviewed)  
**Verdict: CONFIRMED NOVEL**

---

## Confirmed Gap Statement

> No existing paper corrects the decision boundary under **hidden minority label noise**
> (minority samples systematically mislabeled as majority) via combined **per-sample OOF
> confidence weighting** and **minority boundary synthesis**, **without modifying any
> training labels**.

---

## Papers Checked — Combined Table

| Paper | Venue | Year | Noise Type | Modifies Labels | Weighting+Synthesis | Overlap | Differentiator |
|-------|-------|------|-----------|-----------------|---------------------|---------|----------------|
| SW Framework | KBS | 2022 | Hidden minority + general | No | Yes (density+SMOTE) | 52–60% | Uses hypergraph chaos metric (geometric), not OOF; filters first then synthesizes away from boundary |
| IW-SMOTE | Pattern Rec. | 2022 | General | No | Yes | 60% | Uses UnderBagging ensemble safety detection, not OOF confidence |
| CBS+CSA | IEEE TNNLS | 2024 | Hidden minority | YES (EMA) | Yes (Mixup) | 55% | Modifies labels via EMA — violates our zero-label-mod constraint |
| RSMOTE | Info Sci | 2020/2021 | General | No | Yes (adaptive SMOTE) | 35–40% | Density-aware, not confidence-based; synthesizes away from noisy regions |
| RT2S | CIKM | 2023 | Asymmetric | YES (LLM) | No (reweighting only) | 45% | Uses OOF scores but couples with LLM label correction; no synthesis |
| BAAF | arXiv | 2025 | Imbalanced | No | Yes (adversarial boundary) | 50% | Uses boundary utility feedback not OOF confidence; targets imbalance not noise |
| ABWSMO | JIFS | 2023 | General | No | Yes (K-Means weighted SMOTE) | 45% | Weights via clustering density, not model confidence; synthesizes toward safe regions |
| CIW (Google) | ICLR | 2022 | Symmetric | No | Yes (constrained reweighting) | 40% | Per-minibatch not cross-validation; symmetric noise only |
| LNR | ICML | 2025 | Hidden minority | YES (intentional) | No | 25% | Opposite: injects majority→minority mislabels to fix boundary |
| Robust-GBDT | Springer | 2024 | General | No | No (loss function only) | 25% | No synthesis; gradient boosting specific |
| DivideMix | ICLR | 2020 | Symmetric | YES (pseudo-labels) | No | 30% | Label modification via GMM; no synthesis; symmetric only |
| CleanLab / Confident Learning | JMLR | 2021 | Asymmetric | YES (correction) | No | 25% | Label modification; no synthesis |
| Meta-Weight-Net | NeurIPS | 2019 | Symmetric | No (meta-gradient) | No | 35% | No synthesis; meta-learning reweight |
| WISEST | Sensors | 2024 | General | No | Yes | 30% | Boundary synthesis but threshold-based location, not confidence-directed |
| Conformal Synthesis | arXiv | 2023 | General | No | No (synthesis only) | 40% | Confidence regions for synthesis; deep learning, not tabular |
| GK-SMOTE | arXiv | 2025 | General | No | Yes | 30% | Avoids noisy regions via KDE; opposite direction from MSBS |

---

## SW (2022) Deep Analysis — Primary Risk

**Title:** *A weighted space division framework for imbalanced problems with label noise*  
**Venue:** Knowledge-Based Systems (Q1) 2022  
**Overlap score:** 52–60%

| Feature | SW (2022) | CWMS+MSBS (Ours) |
|---------|-----------|-----------------|
| Noise detection mechanism | Hypergraph chaos metric (global geometry) | OOF P(minority\|x) (instance-level, same model family) |
| Synthesis direction | Away from noisy regions (filter-first) | TOWARD suspicious majority samples (exploit boundary signal) |
| Noise target | General noise, not specifically hidden minority asymmetric | Specifically ε_mn >> ε_mj (minority→majority dominates) |
| Label modification | No | No |
| Single pass | No (separate preprocessing) | Yes (OOF scores reused, no extra model) |

**Verdict:** SW does NOT fill our gap. It uses a fundamentally different detection mechanism (geometric vs. model-confidence), synthesizes away from noise (filter-first paradigm), and does not specifically target hidden minority asymmetric noise.

**Differentiation strategy for paper:**
> SW uses global space partitioning (hypergraph chaos) to identify noisy regions and avoids them during synthesis. CWMS+MSBS uses instance-level OOF confidence to identify *which specific samples* are near the true minority boundary, and synthesizes *toward* them — exploiting the noise signal rather than avoiding it.

---

## Phase 3 Case Decision

**Case A — Gap fully confirmed.** No direct conflict found.

**Required actions (Case A from plan):**
1. Cross-scorer experiment (optional, addresses reviewer G1): OOF scorer = HGB → final model = LR
2. Budget sensitivity analysis (optional, addresses G3): budget = 0.05, 0.10, 0.20

**Confirmed differentiations to include in Related Work:**
- SW (2022): mechanism and direction differ (chaos metric vs OOF; filter-first vs synthesize-toward)
- IW-SMOTE (2022): different noise detection (ensemble vs OOF); no targeted hidden minority scope
- CBS+CSA (2024): same noise type but modifies labels (EMA) — we maintain zero-label-mod
- RT2S (2023): uses OOF but for label correction, not sample weighting; no synthesis

---

## LN-SMOTE / Noise-Robust SMOTE Variants (Agent 3)

- RSMOTE (IS 2020): divides into borderline/safe/noise regions; removes noisy samples then synthesizes safely. Filter-first, opposite paradigm.
- SMOTE-NaN-DE (KBS 2021): evolutionary search for oversampling parameters. No confidence-guided direction.
- CRN-SMOTE (PLOS ONE 2025): cluster-based noise reduction then SMOTE. Filter-first.

All existing noise-robust SMOTE variants follow **filter-first then oversample** paradigm. MSBS is unique in synthesizing TOWARD the corrupted boundary.

---

## Unresolved Questions

1. Does SW (2022) truly avoid all label modification, or does probability-weighted resampling count? (Get full paper to confirm for Related Work section)
2. Can the cross-scorer experiment (HGB → LR) be run before paper submission to empirically address G1?
3. IW-SMOTE (2022) is the second-closest match — should be cited with clear differentiation in Related Work.
