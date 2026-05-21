"""Reusable sklearn-compatible model factories for tabular experiments."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ModelFactory = Callable[[], object]


def list_publication_models(include_optional: bool = True) -> list[str]:
    """Return the planned model families in increasing publication rigor."""
    models = ["lr", "calibrated_lr", "random_forest", "extra_trees", "hgb"]
    if include_optional:
        for name, package in [
            ("xgboost", "xgboost"),
            ("lightgbm", "lightgbm"),
            ("catboost", "catboost"),
        ]:
            if find_spec(package) is not None:
                models.append(name)
    return models


def model_supports_sample_weight(model_name: str) -> bool:
    """Whether balanced OOF should pass explicit sample weights."""
    return model_name == "hgb"


def make_model_factory(
    model_name: str,
    seed: int,
    cat_indices: list[int] | None = None,
    balanced: bool = False,
) -> ModelFactory:
    """Create a fresh sklearn-compatible estimator on every call."""
    cat_indices = cat_indices or []
    if model_name == "lr":
        return lambda: _make_lr(seed, balanced)
    if model_name == "calibrated_lr":
        return lambda: CalibratedClassifierCV(_make_lr(seed, balanced), method="sigmoid", cv=3)
    if model_name == "random_forest":
        return lambda: _make_rf(seed, balanced)
    if model_name == "extra_trees":
        return lambda: _make_extra_trees(seed, balanced)
    if model_name == "hgb":
        return lambda: make_pipeline(
            HistGradientBoostingClassifier(
                categorical_features=cat_indices if cat_indices else None,
                random_state=seed,
            )
        )
    if model_name == "xgboost":
        return lambda: _make_xgboost(seed, balanced)
    if model_name == "lightgbm":
        return lambda: _make_lightgbm(seed, balanced)
    if model_name == "catboost":
        return lambda: _make_catboost(seed, balanced)
    raise ValueError(f"Unknown model_name: {model_name}")


def _make_lr(seed: int, balanced: bool) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced" if balanced else None,
            max_iter=1000,
            random_state=seed,
        )),
    ])


def _make_rf(seed: int, balanced: bool) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced" if balanced else None,
            n_jobs=-1,
            random_state=seed,
        ),
    )


def _make_extra_trees(seed: int, balanced: bool) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced" if balanced else None,
            n_jobs=-1,
            random_state=seed,
        ),
    )


def _make_xgboost(seed: int, balanced: bool):
    if find_spec("xgboost") is None:
        raise ImportError("Install optional dependency: pip install xgboost")
    from xgboost import XGBClassifier

    scale = 5.67 if balanced else 1.0
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            eval_metric="logloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def _make_lightgbm(seed: int, balanced: bool):
    if find_spec("lightgbm") is None:
        raise ImportError("Install optional dependency: pip install lightgbm")
    from lightgbm import LGBMClassifier

    return make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced" if balanced else None,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    )


def _make_catboost(seed: int, balanced: bool):
    if find_spec("catboost") is None:
        raise ImportError("Install optional dependency: pip install catboost")
    from catboost import CatBoostClassifier

    return make_pipeline(
        SimpleImputer(strategy="median"),
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            auto_class_weights="Balanced" if balanced else None,
            random_seed=seed,
            verbose=False,
        ),
    )
