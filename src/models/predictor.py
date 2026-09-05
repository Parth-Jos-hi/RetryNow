"""The ML predictor: P(success | retry this failed payment).

Learns, from engineered features, the probability that a *failed* transaction
would succeed if we retried it.  A light manual random search (no extra
dependencies) picks hyperparameters; probabilities can be calibrated so the
decision engine's thresholds are meaningful; the model saves/loads with joblib.

Model choice follows ``config.model.type``:
  * ``xgboost``      -> XGBoost (if installed)
  * ``histogram_gb`` -> sklearn.HistGradientBoostingClassifier
  * ``random_forest``-> sklearn.RandomForestClassifier
If the requested type is missing, it falls back in that order (XGBoost -> HistGB
-> RandomForest) so the pipeline always runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# Priority-ordered backends.  XGBoost is optional in requirements; the fallbacks
# keep the whole pipeline runnable on a plain sklearn install.
_BACKENDS = {
    "xgboost": "xgboost",
    "histogram_gb": "sklearn.ensemble.HistGradientBoostingClassifier",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
}

# Per-backend hyperparameter search spaces, as {name: [min, max]} for numeric
# draws (uniform, log-scaled where noted) and {name: [choices]} for categoricals.
_PARAM_SPACES: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": [100, 400],            # integer
        "max_depth": [3, 8],                   # integer
        "learning_rate": [0.02, 0.25],         # log-ish
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "min_child_weight": [1, 7],            # integer
    },
    "histogram_gb": {
        "max_iter": [100, 300],                # integer
        "learning_rate": [0.02, 0.25],
        "max_leaf_nodes": [15, 64],            # integer
        "min_samples_leaf": [10, 60],          # integer
    },
    "random_forest": {
        "n_estimators": [200, 600],            # integer
        "max_depth": [5, 20],                  # integer
        "min_samples_split": [2, 10],          # integer
        "min_samples_leaf": [1, 8],            # integer
    },
}

# Backends where a range [lo, hi] should be drawn on a log scale.
_LOG_DRAWS: dict[str, set[str]] = {
    "xgboost": {"learning_rate"},
    "histogram_gb": {"learning_rate"},
    "random_forest": set(),
}


def resolve_backend(requested: str) -> str:
    """Return the concrete backend to use, honoring availability fallbacks."""
    order = _BACKENDS.keys() if requested in _BACKENDS else _BACKENDS.keys()
    # try requested first, then fall back through the priority list
    candidates = [requested] + [b for b in _BACKENDS if b != requested]
    for name in candidates:
        if name not in _BACKENDS:
            continue
        if _backend_available(name):
            if name != requested:
                log.warning("backend '%s' unavailable; falling back to '%s'",
                            requested, name)
            return name
    raise RuntimeError("no usable model backend (xgboost / sklearn both missing)")


def _backend_available(name: str) -> bool:
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
            return True
        except Exception:
            return False
    return True  # histogram_gb / random_forest are always in sklearn


def make_model(backend: str, **params) -> Any:
    """Instantiate a fresh, un-fit estimator for ``backend`` with ``params``."""
    if backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, verbosity=0, n_jobs=-1, **params)
    if backend == "histogram_gb":
        return HistGradientBoostingClassifier(
            random_state=0, early_stopping=True, **params)
    if backend == "random_forest":
        return RandomForestClassifier(random_state=0, n_jobs=-1, **params)
    raise ValueError(f"unknown backend: {backend}")


def sample_params(backend: str, rng: np.random.Generator) -> dict[str, Any]:
    """Draw one hyperparameter candidate from the backend's search space."""
    space = _PARAM_SPACES[backend]
    log_spans = _LOG_DRAWS[backend]
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, list) and len(spec) == 2 and _is_numeric(spec):
            lo, hi = spec
            if name in log_spans:
                val = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                val = rng.uniform(lo, hi)
            params[name] = int(round(val)) if _is_integer_param(backend, name) else val
        else:
            params[name] = rng.choice(spec)
    return params


def _is_numeric(spec: list) -> bool:
    return all(isinstance(x, (int, float)) for x in spec)


def _is_integer_param(backend: str, name: str) -> bool:
    return name in {
        "n_estimators", "max_depth", "min_child_weight", "max_iter",
        "max_leaf_nodes", "min_samples_leaf", "min_samples_split",
    }


def hyperparameter_search(X_train: pd.DataFrame, y_train: pd.Series,
                          backend: str, n_trials: int = 30,
                          valid_size: float = 0.20,
                          seed: int = 42) -> tuple[Any, dict[str, Any]]:
    """Manual random search: pick the param draw with best validation ROC-AUC.

    Splits a small validation slice off the training set, tries ``n_trials``
    candidates, and returns ``(best_model, best_params)`` where ``best_model``
    is refit on the *full* training set with the winning hyperparameters.
    """
    rng = np.random.default_rng(seed)
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=seed, stratify=y_train)

    best_score, best_params = -1.0, None
    for trial in range(n_trials):
        params = sample_params(backend, rng)
        try:
            model = make_model(backend, **params)
            model.fit(X_sub, y_sub)
            score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        except Exception as exc:  # a bad param draw should not kill the search
            log.debug("trial %d failed: %s", trial, exc)
            continue
        if score > best_score:
            best_score, best_params = score, params
        log.info("trial %2d/%d  AUC=%.4f  params=%s", trial + 1, n_trials, score, params)

    if best_params is None:
        raise RuntimeError("hyperparameter search produced no valid model")
    log.info("best validation AUC=%.4f with %s", best_score, best_params)

    # refit the winner on the whole training set
    model = make_model(backend, **best_params)
    model.fit(X_train, y_train)
    return model, best_params


def calibrate(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
              method: str = "isotonic", cv: int = 3) -> Any:
    """Wrap ``model`` in a probability calibrator (Platt/isotonic, CV).

    Calibration turns raw scores into trustworthy probabilities so the decision
    engine's thresholds (e.g. ``retry_soon_threshold: 0.45``) behave like real
    probabilities, not relative rankings.
    """
    calibrator = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrator.fit(X_train, y_train)
    return calibrator


def train(X_train: pd.DataFrame, y_train: pd.Series,
          backend: str = "xgboost",
          n_trials: int = 30,
          probability_calibration: bool = True,
          seed: int = 42) -> tuple[Any, dict[str, Any]]:
    """Full training entrypoint: search -> fit -> (optionally) calibrate.

    Returns ``(fitted_model, meta)`` where ``meta`` records the backend and best
    params so evaluation/simulation can attribute results.
    """
    backend = resolve_backend(backend)
    model, best_params = hyperparameter_search(
        X_train, y_train, backend=backend, n_trials=n_trials, seed=seed)

    meta = {"backend": backend, "best_params": best_params,
            "probability_calibration": probability_calibration}

    if probability_calibration:
        model = calibrate(model, X_train, y_train)
        meta["calibrated"] = True

    return model, meta


def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted model and return a metrics dict."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "n_test": float(len(y_test)),
        "positive_rate": float(y_test.mean()),
    }
    return metrics


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(success) for each row, as a float array in [0, 1]."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def save_model(model: Any, path: str | Path, meta: dict | None = None) -> Path:
    """Persist the fitted model (and optional metadata) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "meta": meta or {}}
    joblib.dump(payload, path)
    log.info("saved model -> %s", path)
    return path


def load_model(path: str | Path) -> tuple[Any, dict]:
    """Load a model saved by :func:`save_model`; returns ``(model, meta)``."""
    payload = joblib.load(path)
    return payload["model"], payload.get("meta", {})
