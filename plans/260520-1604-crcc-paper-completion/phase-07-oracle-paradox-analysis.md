---
phase: 7
title: "Oracle Paradox Analysis"
status: pending
priority: P1
effort: "2h"
dependencies: [3]
---

# Phase 7: Oracle Paradox Analysis

## Overview

CRCC-P consistently outperforms oracle deletion on balanced accuracy and minority recall, despite oracle having perfect noise-label knowledge. This is one of the most surprising findings in the dataset and needs formal analysis for the paper.

**The paradox**: Oracle knows exactly which labels are noisy. CRCC-P does not. Yet CRCC-P achieves higher balanced accuracy. Why?

**Hypothesis**: Oracle deletes noisy minority samples (correctly-labeled-minority samples that oracle misidentifies? No — oracle uses ground truth). Wait: oracle knows noisy labels. Noisy minority samples = clean minority samples whose labels were flipped to majority. Oracle deletes these correctly. But by deleting them, oracle removes samples from the feature space of the minority class — even though those samples have wrong labels, their *feature vectors* come from the minority distribution, helping the model learn the minority boundary. CRCC-P preserves those samples (cap prevents minority-class deletion) — the model keeps seeing minority-feature-space samples, even if mislabeled, which provides a form of feature distribution regularization.

## Related Code Files

- Create: `scripts/run_oracle_paradox_analysis.py` (≤ 180 lines)
- Read: `pipeline/core/experiment.py`, `pipeline/cleaning/selectors.py`
- Write: `outputs/oracle-paradox-analysis.csv`

## Implementation Steps

### Step 1 — Config

Use original 3 datasets (pima, credit-g, yeast) for controlled analysis:
```python
@dataclass(frozen=True)
class OracleParadoxConfig(BaseExperimentConfig):
    datasets: tuple = ("pima", "credit-g", "yeast")
    # all other defaults: 85/15, 30%/10% noise, 10% budget
```

### Step 2 — Per-seed analysis

For each dataset/model/seed, compute:
```python
oracle_set = set(select_oracle(noisy_mask, budget_count))
crcc_set   = set(select_crcc_p(suspiciousness, y_noisy, budget_count,
                                lambda_risk=0.5, minority_label=minority_label))

# Overlap analysis
overlap       = len(oracle_set & crcc_set)
oracle_only   = list(oracle_set - crcc_set)   # oracle deletes, CRCC preserves
crcc_only     = list(crcc_set - oracle_set)    # CRCC deletes, oracle wouldn't

# Of oracle_only: what fraction are CLEAN minority (the paradox samples)?
# oracle_only = noisy samples (by definition, since oracle only selects noisy)
# oracle_only minority class: samples with TRUE label = minority (flipped to majority in y_noisy)
# y_train_imb[oracle_only] gives true label
oracle_only_true_minority = (y_train_imb[oracle_only] == minority_label).sum()
oracle_only_minority_frac = oracle_only_true_minority / len(oracle_only) if oracle_only else float("nan")
```

The key: `oracle_only` are noisy minority samples (true label=minority, observed label=majority). Oracle correctly deletes them. But their feature vectors come from the minority distribution — removing them impoverishes the model's view of the minority feature space.

### Step 3 — Performance comparison

Train model on: (a) after oracle deletion, (b) after CRCC-P deletion.
Record `balanced_accuracy` and `minority_recall` for each. This is already done by `evaluate()` — just read from the existing `full-experiment-results.csv`.

```python
# From existing results CSV:
oracle_ba = df[(df["method"]=="oracle_deletion") & ...]["balanced_accuracy"].mean()
crcc_ba   = df[(df["method"]=="crcc_p_l05") & ...]["balanced_accuracy"].mean()
ba_gain   = crcc_ba - oracle_ba  # expected: positive
```

### Step 4 — Build analysis script

```python
def analyze_seed(dataset, model, seed, cfg):
    # Replicate the setup from run_single() but track set intersection
    # Returns dict with: overlap, oracle_only_count, oracle_only_minority_frac,
    #                    crcc_only_count, budget_count, minority_count
    ...

def main():
    cfg = OracleParadoxConfig()
    # Run analysis for all combos
    # Load full-experiment-results.csv for BA comparison
    # Merge and output oracle-paradox-analysis.csv
    # Print summary table
```

### Step 5 — Validate hypothesis

Expected findings:
1. `oracle_only_minority_frac` > 0.5 on most combos (oracle over-deletes minority feature space)
2. `ba_gain_over_oracle` > 0 on most combos (CRCC-P outperforms oracle on BA)
3. Correlation: higher `oracle_only_minority_frac` → higher `ba_gain_over_oracle`

If (1) is true: confirms hypothesis. Write the explanation.
If (1) is false: fall back explanation — "oracle is budget-limited; at 10% budget it deletes the noisiest samples which happen to include many majority-mislabeled-minority; CRCC-P's cap distributes the budget more favorably for minority class retention."

### Step 6 — Write analysis text

Produce 4-paragraph analysis in `docs/research-foundation.md` under § 8 Oracle Paradox:
- Para 1: State the paradox quantitatively ("CRCC-P outperforms oracle by X.XX BA on average")
- Para 2: Hypothesis (feature distribution argument)
- Para 3: Empirical evidence (oracle_only_minority_frac statistics)
- Para 4: Implication for method design (budget allocation matters; perfect detection isn't sufficient)

## Success Criteria

- [ ] `scripts/run_oracle_paradox_analysis.py` exists, ≤ 180 lines
- [ ] `outputs/oracle-paradox-analysis.csv` exists with `oracle_only_minority_frac` and `ba_gain_over_oracle` columns
- [ ] `oracle_only_minority_frac` > 0.5 confirmed or alternative explanation documented
- [ ] `ba_gain_over_oracle` > 0 on majority of dataset/model/seed combos
- [ ] 4-paragraph analysis appended to `docs/research-foundation.md`

## Risk Assessment

- The paradox may not hold for all datasets after switching to Yeast/Ecoli/Phoneme. Run on pima/credit-g/yeast only (controlled) and report dataset-specific results.
- If ba_gain_over_oracle is sometimes negative (oracle wins on some combos), report it honestly — "CRCC-P outperforms oracle in X/15 combos; on the remaining Y/15 combos oracle is superior."
- The analysis script partially duplicates `run_single()` logic to access internal sets. To avoid full duplication, add an optional `return_selection_sets=True` parameter to `run_single()` in `pipeline/core/experiment.py` that returns the selector outputs alongside metrics.
