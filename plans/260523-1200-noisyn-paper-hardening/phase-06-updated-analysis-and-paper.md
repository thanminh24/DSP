---
phase: 6
title: "Updated Analysis and Paper"
status: pending
priority: P1
effort: "3h"
dependencies: [1, 2, 3, 4, 5]
---

# Phase 6: Updated Analysis and Paper

## Overview

Rewrite all paper tables and sections with hardened numbers, honest framing, and new ablation results. Every number in the paper must be traceable to a specific CSV row or computed statistic.

## Related Code Files

- Modify: `docs/paper-draft.md`
- Modify: `scripts/analyze_full_benchmark.py` — add NB correction, IR breakdown, oracle row
- Modify: `scripts/analyze_competitor_headtohead.py` — Table 2a–2d, expanded scope
- Create: `scripts/analyze_rfet_ablation.py`
- Create: `scripts/analyze_failure_mode.py`
- Create: `scripts/analyze_clean_data.py`
- Modify: `docs/results-reference.md` — update all numbers

## Implementation Steps

### Step 1 — Re-run all analysis scripts and collect numbers

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/analyze_full_benchmark.py \
  --input outputs/full-benchmark-solution-v2.csv > plans/reports/table1-v2-numbers.txt

/home/than-minh/miniconda3/envs/dsp/bin/python scripts/analyze_competitor_headtohead.py \
  --input outputs/competitor-headtohead-v2.csv > plans/reports/table2-v2-numbers.txt

/home/than-minh/miniconda3/envs/dsp/bin/python scripts/analyze_rfet_ablation.py \
  > plans/reports/rfet-ablation-numbers.txt
```

### Step 2 — Update Table 1 in paper

Table 1 now covers 15 datasets. Update:
- Row count: "15 datasets × 10 seeds × 3 protocols = 450 pairs per row" (not 150)
- Replace aggregate Wilcoxon p-values with **per-dataset Wilcoxon + Stouffer combination** (from Phase 1 `per_dataset_wilcoxon_stouffer()`)
- Report: "X/15 datasets individually significant (p<0.05)" and "Stouffer Z=X.X, p=X.Xe-X"
- Re-evaluate which models are "significant": LR and SVM expected to remain significant; HGB/LGB may not be
- **Oracle relabel goes to Appendix only** — it is NOT a main Table 1 row. Reference as "oracle_relabel upper bound: BA=X.XX, see Appendix A." The oracle uses ground-truth labels at training time and is not a deployable method.
- Honest footnote for HGB/LGB: "HGB and LightGBM show positive mean ΔBA at IR=0.15 but are not individually significant across datasets (Stouffer Z=X.X, p>0.05). These models' built-in class balancing partially neutralises the OOF signal."

### Step 3 — Update Table 2 in paper

Table 2 now shows per-model breakdown (LR, SVM, HGB) and per-protocol breakdown. Key update required:
- Replace single aggregate row with Table 2a (aggregate) + Table 2b (per-model) structure
- Clearly state which models NoiSyn beats IW-SMOTE on and which it does not
- Add IW-SMOTE lamda choice justification as a footnote
- Add SW-approx status (kept or removed based on Phase 2)

### Step 4 — Add new ablation tables to paper

**Table 3 — RF/ET Ablation** (from Phase 2 Sweep B)
```
Model        | class_prop | cwms_only | msbs_only | cwms_msbs | ΔBA(cwms vs cp)
Random Forest|    0.701   |    ?      |    ?      |    0.655  |  -4.64pp
Extra Trees  |    0.687   |    ?      |    ?      |    0.649  |  -3.80pp
```
Caption: "Decomposing the RF/ET regression: CWMS-only (suppression without synthesis) vs MSBS-only (synthesis without suppression) isolates the source of harm."

**Table 4 — Failure-Mode Protocols** (from Phase 2 Sweep A)
```
Protocol        | no_clean | class_prop | NoiSyn | Δ(vs cp)
hidden_minority | 0.576    | 0.705      | 0.739  | +3.47pp ← designed for this
symmetric       | 0.xxx    | 0.xxx      | 0.xxx  | ±x.xxpp ← expected ~0
reverse_asymm.  | 0.xxx    | 0.xxx      | 0.xxx  | -x.xxpp ← expected negative
```
Caption: "NoiSyn operating conditions: designed for hidden minority-class asymmetric noise. Performance under symmetric or reverse-asymmetric noise confirms the method's assumptions."

**Table 5 — IR Sensitivity** (from Phase 4)
```
IR       | ΔBA (LR, NoiSyn vs class_prop) | p-value
0.15     | +3.47pp                         | 6.1e-15
0.30     | ???pp                            | ???
```

### Step 5 — Revise paper text for honest framing

Key rewrites required in `docs/paper-draft.md`:

1. **Abstract**: Change "across seven model families" → "for linear (LR) and kernel (SVM) classifiers, with marginal gains for gradient boosting models. Bootstrap ensemble models (RF, ET) show regression consistent with their noise-robust bootstrapping mechanism."

2. **Contributions section**: Remove "seven model families" claim. Replace with: "Empirical characterisation of which model families benefit from NoiSyn and why, including the first documented case where confidence-weighted synthesis actively harms bootstrap ensemble models."

3. **Discussion — RF/ET**: Replace "bagging robustness rationalisation" with Phase 2 ablation findings. Identify whether CWMS or MSBS is the harmful component.

4. **Discussion — Precision-Recall tradeoff**: Add explicit framing: "NoiSyn is a recall-first method. Users whose application penalises false positives equally to false negatives should use class_proportional or IW-SMOTE instead."

5. **Discussion — Scorer self-consistency**: Soften "same-family is better" to "in our evaluation, same-family OOF outperforms cross-family (HGB→LR: 0.687 vs LR→LR: 0.745). The confound of model capacity prevents a clean attribution — HGB scores are richer but may not be properly calibrated for LR exploitation."

6. **Limitations**: Add paragraphs for: (a) PR-AUC not computed in current benchmark, (b) natural noise not tested, (c) multi-class not addressed, (d) IR scope based on Phase 4 result.

### Step 6 — Update results-reference.md

Rebuild `docs/results-reference.md` with all new numbers. Add section headers per table. Include CSV row counts and checksums so future collaborators can reproduce from the exact CSVs.

### Step 7 — Final consistency check

Run grep on paper-draft.md for:
- Any remaining "1,350" → should be 0 after Phase 1 fix (correct: 1,050)
- Any "five datasets" → update to "fifteen" or "five (original) / fifteen (expanded)"
- Any "150 pairs" → update to "450 pairs" for 15-dataset results; keep "150 pairs" only where explicitly discussing the original 5-dataset run
- Any claim that HGB/LGB is significant without per-dataset Wilcoxon + Stouffer support
- Any mention of oracle_relabel in main results → move to Appendix reference

```bash
grep -n "1,350\|1350\|five datasets\|150 pairs\|oracle_relabel" docs/paper-draft.md
```

Each hit must be manually resolved before submission.

## Success Criteria

- [ ] `docs/paper-draft.md` updated with hardened numbers from v2 CSVs
- [ ] Table 1 uses per-dataset Wilcoxon + Stouffer p-values; HGB/LGB claim honestly scoped with correct statistic
- [ ] Oracle relabel is **Appendix only** — not a Table 1 row; referenced as "upper bound, see Appendix A"
- [ ] Table 2 has per-model breakdown (2a aggregate + 2b per-model)
- [ ] Tables 3 (RF/ET ablation), 4 (failure modes), 5 (IR sensitivity) added
- [ ] Abstract and Contributions accurately reflect the narrow but real LR/SVM story
- [ ] `docs/results-reference.md` rebuilt with all new numbers
- [ ] Zero grep hits for "1,350", "oracle_relabel" in main sections (must be 0 after fix)
- [ ] Limitations section covers: PR-AUC, natural noise, multi-class, IR scope

## Risk Assessment

- **Numbers moving**: After expanding to 15 datasets and 3 models in Table 2, some headline numbers will change. If LR+NoiSyn vs IW-SMOTE margin narrows below p=0.05 on the expanded comparison, the Table 2 story weakens. Mitigation: accept the result; the LR+SVM Table 1 story at 450 pairs is the primary contribution.
- **Writing coherence**: With 6 tables and 8 paper sections, maintaining internal consistency is the main editing risk. Use `docs/results-reference.md` as the single source of truth and reference it in all paper sections.
