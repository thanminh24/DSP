---
phase: 4
title: Competitor Baselines + Two-Table Paper Result Design
status: completed
priority: P1
effort: 6h
dependencies:
  - 2
---

# Phase 4: Competitor Baselines + Two-Table Paper Result Design

## Overview

Design and implement the two final paper result tables. Table 1 is our internal benchmark across model families. Table 2 is external comparison against published noise-robust SMOTE methods. Also adds plain SMOTE as a universal baseline (appears in every competitor paper) and G-mean as a metric (derivable for free from existing CSV columns).

## Competitor Landscape (updated from experiment-setup audit)

| Paper | Noise type | Models used | Metrics | No public code | Our strategy |
|-------|-----------|-------------|---------|----------------|--------------|
| **IW-SMOTE** (Pattern Rec. 2022) | None explicit / general | LR, SVM, k-NN, RF | F1, G-mean, AUC | No — **code downloaded** ✓ | Reproduce directly |
| **SW Framework** (KBS 2022) | General | SVM, k-NN, DT, LR | BA, G-mean, F1 | No | Approximate via k-NN chaos |
| **GK-SMOTE** (APWeb-WAIM 2025) | Symmetric random flips 10/20/30% | RF, LightGBM, LR, KNN, DT | MCC, BAc, AUPRC | **No code** | Reference only — cite, don't reproduce |
| **CRN-SMOTE** (PLOS ONE 2025) | Natural (DBSCAN removal) | SVM, RF, AdaBoost | Kappa, MCC, F1 | Unknown | Skip — no explicit noise injection, incomparable |
| **iHHO-SMOTe** (IJCA 2025) | Natural (DBSCAN) | Unspecified | AUC, G-mean | No | Skip — insufficient metadata |

**Consensus classifier for external comparison:** LR (used by IW-SMOTE, SW, GK-SMOTE, us)
**Consensus metric:** G-mean (IW-SMOTE, SW, GK-SMOTE all report it; derivable from existing CSV for free)

---

## Part A — Add G-mean to Metrics (free derivation, no re-run)

G-mean = sqrt(minority_recall × majority_recall). Both columns already in all CSVs.

```python
# Add to every analysis script before printing tables
df["g_mean"] = np.sqrt(
    df["minority_recall"].clip(lower=0) * df["majority_recall"].clip(lower=0)
)
```

Add to: `analyze_full_benchmark.py`, `analyze_scorer_agnosticism.py`, `analyze_competitor_headtohead.py`.

**Why G-mean matters:** IW-SMOTE, SW Framework, and GK-SMOTE all report G-mean. Including it makes Table 2 directly comparable to their published tables without re-running their code.

---

## Part B — Plain SMOTE Baseline (implementation in Phase 2)

SMOTE (Chawla 2002) appears in ALL competitor papers. Implementation dispatcher is in **Phase 2 Part D** (with full code). Runs in both Run A and Run C.

**Why it matters for framing:**
If CWMS+MSBS > IW-SMOTE > SMOTE > no_cleaning, that chain tells the full story: (1) any synthesis helps, (2) noise-aware synthesis helps more, (3) our targeted boundary synthesis helps most.

---

## Part C — IW-SMOTE (IMPLEMENTED ✓)

**Status:** `pipeline/baselines/iw_smote.py` done. Smoke test passed.

```
Input: n=441, minority=83, majority=358, budget=44
After IW-SMOTE: n=380, minority=127, n_synthetic=44
LR + IW-SMOTE:  BA=0.7094  RecMin=0.8507  (pima, medium noise, seed=42)
LR + no_cleaning: BA=0.7113  RecMin=0.6866
```

---

## Part D — SW Framework (approximate)

**Status:** `pipeline/baselines/sw_framework.py` — pending. Implementation spec in plan (see original phase-04 draft). Approximates hypergraph chaos with k-NN label inconsistency. Label clearly as "SW-approx" in paper.

---

## Part E — Head-to-Head Sweep Scope

**Script:** `scripts/run_competitor_headtohead.py`

```python
COMPETITOR_METHODS = [
    "no_cleaning",           # universal baseline
    "class_proportional",    # our primary internal baseline
    "smote",                 # noise-unaware oversampling (universal paper baseline)
    "iw_smote",              # real code, Pattern Rec. 2022
    "sw_framework",          # approximate, KBS 2022
    "cwms_msbs",             # ours
]
COMPETITOR_MODELS = ["lr"]   # LR only — consensus classifier across all papers
COMPETITOR_PROTOCOLS = ["hidden_minority_medium"]  # ε_mn=0.25, closest to GK-SMOTE 20-30%
# 1 model × 6 methods × 5 datasets × 10 seeds × 1 protocol = 300 rows
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead.csv"
```

**Why LR only for Table 2:** IW-SMOTE, SW, and GK-SMOTE all evaluate LR. Using LR for Table 2 gives the most direct apples-to-apples comparison. Table 1 covers all model families.

---

## Final Paper Table Designs

### Table 1 — Internal Benchmark (primary contribution)

**Title:** "CWMS+MSBS vs. baselines across model families (mean BA ± std, hidden-minority noise, 3 protocols × 5 datasets × 10 seeds)"

```
Model        | no_cleaning | class_prop | cwms_msbs | ΔBA  | p-value   | wins/150
-------------|-------------|------------|-----------|------|-----------|----------
LR           | 0.6XXX      | 0.7032     | 0.7454    |+4.22 | 6.2e-07 ↑ | 143/150
SVM          | ...         | ...        | ...       | ...  | ...       | ...
HGB          | ...         | ...        | ...       | ...  | ...       | ...
LightGBM     | ...         | ...        | ...       | ...  | ...       | ...
CatBoost     | ...         | ...        | ...       | ...  | ...       | ...
Random Forest| ...         | ...        | ...       | ...  | ...       | ...
Extra Trees  | ...         | ...        | ...       | ...  | ...       | ...
-------------|-------------|------------|-----------|------|-----------|----------
Avg (CWMS-   | ...         | ...        | ...       |      |           |
 compatible) |
```

Footnote: XGBoost and calibrated_lr run baselines only (structural incompatibilities documented in §3.2).
Shuffled ablation (Table 1b): cwms_msbs vs cwms_msbs_shuffled ΔBA + p-value per model, confirms scores load-bearing.

**Columns explained:**
- `ΔBA` = cwms_msbs − class_proportional (pp)
- `p-value` = Wilcoxon signed-rank, two-sided, n=150 pairs
- `wins/150` = fraction of (dataset, seed, protocol) triples where cwms_msbs > class_prop

---

### Table 2 — External Comparison (related work comparison)

**Title:** "Comparison with published noise-robust oversampling methods (LR classifier, hidden-minority noise ε_mn=0.25, 5 datasets × 10 seeds)"

```
Method              | Source              | BA     | G-mean | Min.Recall
--------------------|---------------------|--------|--------|------------
no_cleaning         | —                   | 0.XXX  | 0.XXX  | 0.XXX
SMOTE               | [Chawla 2002]       | 0.XXX  | 0.XXX  | 0.XXX
class_proportional  | [He & Garcia 2009]  | 0.XXX  | 0.XXX  | 0.XXX
IW-SMOTE            | [Zhang et al. 2022] | 0.XXX  | 0.XXX  | 0.XXX
SW (approx†)        | [X et al. 2022]     | 0.XXX  | 0.XXX  | 0.XXX
**CWMS+MSBS (ours)**| —                   | **0.XXX** | **0.XXX** | **0.XXX**
```

Caption note: †SW reproduced with k-NN label inconsistency approximation (original uses hypergraph chaos, no public code). GK-SMOTE (APWeb-WAIM 2025) not reproduced — no public code; evaluated under symmetric noise, incomparable noise protocol. All results under hidden-minority asymmetric noise (ε_mn=0.25, ε_mj=0.02), tabular benchmark (5 UCI/OpenML datasets), 10 seeds, budget=10% of |D_train|.

---

## Noise Protocol Alignment vs Competitors

| Paper | Their noise | Our noise | Bridge |
|-------|------------|-----------|--------|
| GK-SMOTE (2025) | Symmetric flips 20% | Asymmetric hidden minority 25% | Cite as reference; note "symmetric noise obscures hidden minority boundary corruption — we address a strictly harder case" |
| IW-SMOTE (2022) | General (no explicit injection in paper) | Hidden minority ε_mn=0.25 | Direct comparison valid — IW-SMOTE makes no noise-type assumption |
| SW (2022) | General (density-based detection) | Hidden minority | Direct comparison valid — SW targets general noise; we show it underperforms on asymmetric case |

**Key framing for Table 2 caption:** "We evaluate all methods under hidden-minority-class asymmetric noise — a strictly harder setting than the symmetric noise used by GK-SMOTE (2025) and the noise-agnostic setting of IW-SMOTE (2022). Our method is designed for this specific failure mode and outperforms noise-unaware baselines by X.Xpp BA."

---

## Analysis Script: `scripts/analyze_competitor_headtohead.py`

Tables to produce:

**Table A — Mean BA/G-mean/RecMin by method (n=50 pairs)**
```
Method | BA mean | G-mean | RecMin | ΔBA vs SMOTE | p-value
```

**Table B — Wilcoxon pairwise: CWMS+MSBS vs each competitor**
```
cwms_msbs vs IW-SMOTE:    ΔBA=+X.Xpp  p=...  wins=XX/50
cwms_msbs vs SW-approx:   ΔBA=+X.Xpp  p=...  wins=XX/50
cwms_msbs vs SMOTE:       ΔBA=+X.Xpp  p=...  wins=XX/50
cwms_msbs vs class_prop:  ΔBA=+X.Xpp  p=...  wins=XX/50
```

**Table C — Per-dataset breakdown (honest reporting)**
Shows any datasets where a competitor wins. If IW-SMOTE or SW wins on specific datasets, document and explain.

---

## Success Criteria

**Implementation (Phase 2 prerequisites must be done first):**
- [ ] `smote` dispatcher in `run_relabeling_viability_sweep.py` (Phase 2 Part D)
- [ ] `pipeline/baselines/iw_smote.py` — DONE ✓ (smoke tested)
- [ ] `pipeline/baselines/sw_framework.py` created (k-NN chaos approximation)

**Sweep:**
- [ ] `scripts/run_competitor_headtohead.py` created — Run C: 6 methods × LR × 5 datasets × 10 seeds = 300 rows
- [ ] `outputs/competitor-headtohead.csv` has 300 rows, zero NaN BA

**Analysis:**
- [ ] G-mean column added to all three analysis scripts (one-liner, no re-run needed)
- [ ] `scripts/analyze_competitor_headtohead.py` created with Tables A–C

**Paper output:**
- [ ] Table 1 (internal benchmark) filled with numbers from Run A — all 7 model rows
- [ ] Table 2 (external comparison) filled with numbers from Run C — all 6 method rows
- [ ] Caption language drafted: noise protocol alignment statement written
- [ ] GK-SMOTE non-reproduction justified in caption (no code, symmetric noise)

## Risk Assessment

- **SW approximation unfairness:** k-NN chaos is weaker than hypergraph chaos. If SW-approx underperforms badly, reviewers may say "your SW implementation is wrong." Mitigation: clearly label as approximate, explain the simplification, and note no public code exists.
- **IW-SMOTE lamda parameter:** Default `lamda=100` means `IR × 100` CART fits per seed. With IR≈6 for our datasets → ~600 CART fits per seed. For 50 seeds × 5 datasets, that's 150,000 tree fits. Estimate 30–60 min for IW-SMOTE alone. May need to reduce `lamda` to 20–30 for speed while maintaining quality.
- **G-mean vs BA alignment:** If our method maximizes BA but not G-mean (unlikely since BA = mean(minority_recall, majority_recall) and G-mean = sqrt of same), report honestly. Both metrics are dominated by minority recall — methods that boost minority recall will win on both.
- **GK-SMOTE citation:** We don't reproduce it, but reviewers may ask why. Pre-written response in caption: "no public implementation; uses symmetric noise incompatible with our asymmetric protocol."
