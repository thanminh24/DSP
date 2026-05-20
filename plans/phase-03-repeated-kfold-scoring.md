---
phase: 3
title: "Repeated K-Fold Scoring (Parallel)"
status: pending
priority: P2
effort: "1h"
dependencies: []
---

# Phase 2: Repeated K-Fold Suspiciousness

## Overview

Replace the paper-killing LOO idea with **repeated stratified k-fold** (3 repetitions of 5-fold).
Average OOF scores across 3 independent CV rounds. Better signal-to-noise ratio. Literature-backed
(ReCoV, 2306.13990). No discriminability issue at any n_minority. ~3x slower but still fast.

**This phase is INDEPENDENT of Phase 3** — Phase 3 can run with standard OOF first.
Repeated k-fold is a secondary comparison: does it change which samples get selected?

## Why Not LOO

LOO at n_minority=12 trains on 11 minority samples. The discriminability between noisy and clean
minority samples at n=11 is not better than k-fold at n=16 — the signal degrades at very low n.
LOO is expensive and its benefit is unproven at the sample counts we care about.

## Why Repeated K-Fold Works

3×5-fold averaging:
- Each sample scored 3 independent times on differently-partitioned folds
- Average score has lower variance than single-round OOF
- Clean hard samples score consistently high across 3 rounds
- Noisy samples may score high in some rounds and low in others (the instability is the signal)
- No n_minority minimum requirement — works at any scale

## Architecture

`pipeline/scoring/oof_loss.py` — add `repeated_oof_loss()` alongside `out_of_fold_loss()`.
File must stay ≤ 200 lines. Check current size before adding.

## Related Code Files

- Modify: `pipeline/scoring/oof_loss.py` (add `repeated_oof_loss`)
- Modify: `pipeline/core/config.py` (add `n_cv_repeats: int = 3`)

## Implementation Steps

1. Read `pipeline/scoring/oof_loss.py`
2. Add `repeated_oof_loss()`:

```python
def repeated_oof_loss(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Average OOF cross-entropy over n_repeats independent k-fold rounds.

    Reduces score variance vs single-round OOF. Noisy samples show
    high variance across rounds; clean hard samples score consistently.
    """
    all_scores = np.zeros((n_repeats, len(y_noisy)), dtype=float)
    for r in range(n_repeats):
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r * 100)
        scores_r = np.zeros(len(y_noisy), dtype=float)
        for train_idx, valid_idx in folds.split(X, y_noisy):
            model = model_factory()
            model.fit(X[train_idx], y_noisy[train_idx])
            probs = model.predict_proba(X[valid_idx])
            for i, idx in enumerate(valid_idx):
                prob = probs[i, int(y_noisy[idx])]
                scores_r[idx] = -np.log(np.clip(prob, 1e-12, 1.0))
        all_scores[r] = scores_r
    return all_scores.mean(axis=0)
```

3. Add `n_cv_repeats: int = 1` to `BaseExperimentConfig` (default=1 = no change to existing runs)
4. In Phase 3 sweep script, expose a `--repeated` flag to toggle between standard and repeated OOF

## Validation

After running Phase 3 once with standard OOF and once with repeated OOF:
- Compute Jaccard overlap of top-k selected indices per (dataset, model, seed)
- If overlap > 90% across all combos: repeated OOF adds no practical difference — report as
  "robust to scoring variant" and drop from paper
- If overlap < 80% in extreme-IR combos: selection genuinely differs — report effect on metrics

## Success Criteria

- [ ] `repeated_oof_loss()` added; `oof_loss.py` ≤ 200 lines
- [ ] Jaccard overlap computed between standard and repeated OOF selections at ratio=0.05
- [ ] Runtime overhead verified: ≤ 3× single-round OOF per combo

## Risk Assessment

- If n_splits=5 and n_repeats=3: 15 model fits per sample group vs 5. At phoneme (n=4000), each
  fit is ~0.5s → extra 5×0.5=2.5s per combo. Total sweep adds ~1h. Acceptable.
- If repeated OOF = standard OOF in practice: drop Phase 2 from paper; it's an honest null result.
