"""Pilot: simulate CRCC-Adaptive (zero minority cap) vs class_proportional.

Tests phoneme + yeast at ratio=0.05. Go/no-go for full implementation.
Run: python3 scripts/run_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.scoring.oof_loss import out_of_fold_loss
from pipeline.cleaning.selectors import select_class_proportional

SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
RATIO = 0.05
BUDGET = 0.10
DATASETS = ["phoneme", "yeast"]
MINORITY_LABEL = 1
MIN_MINORITY = 20


def _encode(X_df):
    X = X_df.copy()
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.to_numpy(dtype=float)


def _make_lr(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))


def sim_adaptive(suspiciousness, y_noisy, budget_count):
    maj = np.where(y_noisy != MINORITY_LABEL)[0]
    if len(maj) == 0:
        return np.array([], dtype=int)
    return maj[np.argsort(suspiciousness[maj])[::-1]][:budget_count]


rows = []
for ds in DATASETS:
    X_raw, y_raw, _, _ = load_dataset(ds)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X_tr_df, X_te_df, y_tr, y_te = train_test_split(
            X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=seed,
        )
        X_tr_df = X_tr_df.reset_index(drop=True)
        X_tr_df, y_tr = induce_imbalance(
            X_tr_df, y_tr, minority_label=MINORITY_LABEL,
            target_ratio=RATIO, rng=rng,
        )
        if int(np.sum(y_tr == MINORITY_LABEL)) < MIN_MINORITY:
            continue
        y_noisy, noisy_mask = inject_noise(y_tr, minority_label=MINORITY_LABEL, rng=rng)
        X_tr = _encode(X_tr_df)
        X_te = _encode(X_te_df)
        n = len(y_noisy)
        budget_count = max(1, int(round(BUDGET * n)))
        factory = lambda s=seed: _make_lr(s)
        susp = out_of_fold_loss(X_tr, y_noisy, factory, n_splits=5, seed=seed)

        for method, sel in [
            ("class_proportional", select_class_proportional(susp, y_noisy, budget_count)),
            ("sim_adaptive", sim_adaptive(susp, y_noisy, budget_count)),
        ]:
            keep = np.ones(n, dtype=bool)
            keep[sel] = False
            if len(np.unique(y_noisy[keep])) < 2:
                continue
            model = _make_lr(seed)
            model.fit(X_tr[keep], y_noisy[keep])
            pred = model.predict(X_te)
            rows.append({
                "dataset": ds, "seed": seed, "method": method,
                "ba": balanced_accuracy_score(y_te, pred),
                "recall": recall_score(y_te, pred, pos_label=MINORITY_LABEL, zero_division=0),
            })

df = pd.DataFrame(rows)
print("\n=== PILOT RESULTS: class_proportional vs sim_adaptive at ratio=5% ===\n")
all_go = True
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        print(f"  {ds}: NO DATA (n_minority < {MIN_MINORITY} for all seeds)\n")
        continue
    cp = sub[sub["method"] == "class_proportional"]["recall"].mean()
    sa = sub[sub["method"] == "sim_adaptive"]["recall"].mean()
    gain = sa - cp
    go = "GO" if gain >= 0.02 else "NO-GO"
    print(f"  {ds:10s}  class_prop recall={cp:.4f}  sim_adaptive={sa:.4f}  gain={gain:+.4f}")
    print(f"           --> {go} (threshold: +0.02)\n")
    if gain < 0.02:
        all_go = False

print(f"OVERALL: {'GO — proceed to Phase 1' if all_go else 'NO-GO — stop and pivot'}")
