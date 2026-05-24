"""Shared utilities for sweep scripts — GPU check, resume-safe CSV loading."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def check_gpu():
    """Verify CUDA is available for boosting models. Abort if not."""
    results = {}
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(device="cuda", n_estimators=2, verbosity=0)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["xgboost"] = "OK"
    except Exception as e:
        results["xgboost"] = f"FAIL: {e}"
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(device="gpu", n_estimators=2, verbose=-1)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["lightgbm"] = "OK"
    except Exception as e:
        results["lightgbm"] = f"FAIL: {e}"
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(task_type="GPU", devices="0", n_estimators=2, verbose=0)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["catboost"] = "OK"
    except Exception as e:
        results["catboost"] = f"FAIL: {e}"
    failed = [k for k, v in results.items() if not v.startswith("OK")]
    for k, v in results.items():
        print(f"  GPU check [{k}]: {v}", flush=True)
    if failed:
        raise RuntimeError(
            f"--gpu requested but CUDA unavailable for: {failed}. "
            "Remove --gpu or fix CUDA installation."
        )
    print("GPU check: all OK", flush=True)


def load_completed(path: Path) -> set:
    """Load set of completed (dataset, model, seed, protocol, budget, ratio, method) tuples."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        if 'error' in df.columns:
            df = df[df['error'].isna()]
        df = df[pd.to_numeric(df["balanced_accuracy"], errors="coerce").notna()]
    except Exception as exc:
        print(f"WARNING: could not read {path}, restarting from scratch ({exc})", flush=True)
        return set()
    return {
        (r["dataset"], r["model"], int(r["seed"]), r["noise_protocol"],
         float(r["budget"]), float(r["target_ratio"]), r["method"])
        for _, r in df.iterrows()
    }
