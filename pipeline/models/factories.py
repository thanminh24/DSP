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
from sklearn.svm import SVC


ModelFactory = Callable[[], object]


def list_publication_models(include_optional: bool = True) -> list[str]:
    """Return the planned model families in increasing publication rigor."""
    models = ["lr", "svm", "calibrated_lr", "random_forest", "extra_trees", "hgb"]
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
    """Whether balanced OOF should pass explicit sample weights.

    HGB: no class_weight param — must use sample_weight in fit().
    XGBoost: class_weight param doesn't exist; scale_pos_weight shifts the decision
    boundary globally but doesn't produce calibrated P(minority|x) scores suitable
    for ranking. Passing sample_weight gives proper per-sample loss reweighting and
    makes the balanced scoring model meaningfully different from the standard one.
    """
    return model_name in ("hgb", "xgboost", "svm")


def make_model_factory(
    model_name: str,
    seed: int,
    cat_indices: list[int] | None = None,
    balanced: bool = False,
    scale_pos_weight: float = 1.0,
    use_gpu: bool = False,
) -> ModelFactory:
    """Create a fresh sklearn-compatible estimator on every call."""
    cat_indices = cat_indices or []
    if model_name == "lr":
        return lambda: _make_lr(seed, balanced)
    if model_name == "svm":
        return lambda: _make_svm(seed, balanced)
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
        return lambda: _make_xgboost(seed, balanced, scale_pos_weight, use_gpu)
    if model_name == "lightgbm":
        return lambda: _make_lightgbm(seed, balanced, use_gpu)
    if model_name == "catboost":
        return lambda: _make_catboost(seed, balanced, use_gpu)
    raise ValueError(f"Unknown model_name: {model_name}")


def make_cwms_factory(
    model_name: str,
    seed: int,
    cat_indices: list[int] | None = None,
    use_gpu: bool = False,
) -> ModelFactory:
    """Factory with scale_pos_weight=1.0, for CWMS weights carrying the class balance signal.

    For LR: identical to std_factory (LR has no spw).
    For boosting (xgb, lgbm, catboost, hgb): disables built-in class correction
    so CWMS balanced weights are the sole correction mechanism.
    """
    return make_model_factory(
        model_name, seed, cat_indices,
        balanced=False, scale_pos_weight=1.0, use_gpu=use_gpu,
    )


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


def _make_svm(seed: int, balanced: bool) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        ),
    )


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


def _make_xgboost(seed: int, balanced: bool, scale_pos_weight: float = 1.0, use_gpu: bool = False):
    if find_spec("xgboost") is None:
        raise ImportError("Install optional dependency: pip install xgboost")
    from xgboost import XGBClassifier

    device = "cuda" if use_gpu else "cpu"
    # When balanced=True (OOF scoring phase), rely on sample_weight passed via fit()
    # so scale_pos_weight must be 1.0 to avoid double-counting imbalance correction.
    # When balanced=False (final training), use the runtime-computed scale_pos_weight.
    effective_spw = 1.0 if balanced else scale_pos_weight
    return make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            device=device,
            eval_metric="logloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=effective_spw,
            random_state=seed,
            n_jobs=1 if use_gpu else -1,
        ),
    )


def _make_lightgbm(seed: int, balanced: bool, use_gpu: bool = False):
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
            n_jobs=1,
            verbose=-1,
            device="gpu" if use_gpu else "cpu",
        ),
    )


def _make_catboost(seed: int, balanced: bool, use_gpu: bool = False):
    if find_spec("catboost") is None:
        raise ImportError("Install optional dependency: pip install catboost")
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        auto_class_weights="Balanced" if balanced else None,
        random_seed=seed,
        verbose=False,
        task_type="GPU" if use_gpu else "CPU",
        devices="0" if use_gpu else None,
    )
    return make_pipeline(SimpleImputer(strategy="median"), cb)
