# CRCC Full Experiment Pipeline -- Implementation Complete

**Date**: 2026-05-19 17:00
**Severity**: N/A (completed)
**Component**: Full experiment pipeline
**Status**: Resolved

## What Happened

Implemented the complete CRCC experiment pipeline: 7 scripts, 2 docs, 1 requirements.txt. The pipeline runs 300 experiment rows (3 datasets x 2 models x 5 seeds x 10 method variants) and produces 4 publication-quality plots. All modules kept under 200 lines.

Results confirm the hypothesis: per-class proportional caps reduce clean-minority deletion rate by 86-99% versus global top-loss cleaning, while simultaneously improving balanced accuracy and minority recall.

## The Brutal Truth

The lambda parameter -- the "risk-adjusted scoring" that distinguishes CRCC from plain class-proportional deletion -- is dead weight under 85/15 imbalance. Lambda grid {0, 0.25, 0.5, 1.0} produces identical results. The cap alone is the binding constraint; once the minority class hits its allocation (6-12 samples), no further minority deletions occur regardless of adjusted score. Two days of implementation for a finding of "lambda doesn't matter." This is a finding, not a failure, but it stings to discover the novel component of the method contributes nothing under the experimental conditions. The cap was the entire intervention all along.

## Technical Details

**Module collision:** `selectors.py` shadows Python's stdlib `selectors` module. Renamed to `cleaning_selectors.py`. First detected during import tests -- `import selectors` resolved to our file instead of the stdlib, breaking unrelated tooling.

**ColumnTransformer + numpy arrays:** `ColumnTransformer` requires integer column indices when receiving numpy arrays (no column names). Error: `"object of type 'int' has no len()"` when passing string names to `remainder='passthrough'` transformers. Fixed by unifying the preprocessing path: both LR and HGB receive pre-encoded numpy arrays (OrdinalEncoder for LR via SimpleImputer+StandardScaler, native cat codes for HGB). This eliminates the ColumnTransformer entirely for the LR path and avoids the string-index problem.

**File sizes:** cleaning_selectors.py (167 lines), scoring.py (42), evaluator.py (46), run_full_experiment.py (197). All under the 200-line threshold.

## Root Cause Analysis

The lambda insensitivity is structural, not a bug. Under extreme imbalance (85/15), the minority cap is reached almost immediately by samples that rank high on suspiciousness alone. The adjusted-score ranking only matters when caps are loose enough that cross-class competition actually occurs. Our experimental design (extreme imbalance, small datasets) precludes the regime where lambda matters.

## Lessons Learned

1. **Prototype the novel mechanism before building the full pipeline.** A 10-minute script testing lambda sensitivity on one dataset would have flagged this immediately.
2. **Check module names against stdlib.** `selectors` is deceptively generic. Grep `python -c "import sys; print(sys.stdlib_module_names)"` before committing a module name.
3. **ColumnTransformer's numpy-vs-dataframe behavior is a footgun.** Document this in the codebase: always pass DataFrames to ColumnTransformer, or always use integer indices with numpy arrays -- mixing is undefined behavior territory.

## Next Steps

- Consider running a second experiment with milder imbalance (e.g., 70/30) and larger budget (20%) to find the regime where lambda matters.
- The class-proportional baseline is the practical takeaway for extreme imbalance -- update the report to foreground this.
- No immediate code changes needed. The pipeline is clean and reproducible.
