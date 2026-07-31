"""Honest, data-scarce model evaluation.

A single hold-out R2 is unreliable at small n and, in this project, was reported to four
decimals on ~2 test points. This module cross-validates every model with imputation done
INSIDE each fold (so no test information leaks into training), reports the spread, and
refuses to emit a skill number when there are too few samples — an explicit
"insufficient data" verdict instead of an inflated score.
"""

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

MIN_ROWS_FOR_METRIC = 8  # below this, refuse to report any skill metric
SMALL_N_THRESHOLD = 30  # below this, use leave-one-out instead of repeated k-fold
RANDOM_STATE = 42


def choose_cv(n_samples: int):
    """Pick an evaluation scheme appropriate for the sample size."""
    if n_samples < MIN_ROWS_FOR_METRIC:
        return None, "insufficient"
    if n_samples < SMALL_N_THRESHOLD:
        return LeaveOneOut(), "leave-one-out"
    return RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE), "repeated-5-fold"


def _point_scores(y_true, y_pred) -> dict:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}


def evaluate_model(model, X, y) -> dict:
    """Return cross-validated metrics for ``model`` on ``(X, y)``, with an honesty verdict.

    Imputation is fitted inside each fold via a Pipeline. Below MIN_ROWS_FOR_METRIC no
    score is reported; below SMALL_N_THRESHOLD leave-one-out is used and the result is
    flagged non-robust with a caveat.
    """
    n_samples = len(y)
    cv, scheme = choose_cv(n_samples)
    result = {"n": int(n_samples), "cv_scheme": scheme, "robust": False}

    if cv is None:
        result["note"] = (
            f"insufficient data (n={n_samples} < {MIN_ROWS_FOR_METRIC}); no metric reported"
        )
        return result

    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", clone(model))])

    if scheme == "leave-one-out":
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        result.update(_point_scores(y, y_pred))
        result["r2_std"] = None
    else:
        r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
        rmse_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        result["r2"] = round(float(np.mean(r2_scores)), 4)
        result["r2_std"] = round(float(np.std(r2_scores)), 4)
        result["rmse"] = round(float(np.mean(rmse_scores)), 4)
        result["rmse_std"] = round(float(np.std(rmse_scores)), 4)

    result["robust"] = n_samples >= SMALL_N_THRESHOLD
    result["caveat"] = (
        ""
        if result["robust"]
        else f"small-n (n={n_samples}); cross-validated but interpret with care"
    )
    return result


if __name__ == "__main__":
    import numpy as _np

    rng = _np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=40)
    from sklearn.linear_model import LinearRegression

    good = evaluate_model(LinearRegression(), X, y)
    assert good["robust"] and good["r2"] > 0.9, good
    tiny = evaluate_model(LinearRegression(), X[:5], y[:5])
    assert not tiny["robust"] and "note" in tiny, tiny
    print(
        "evaluation smoke OK ->",
        {k: good[k] for k in ("n", "cv_scheme", "r2")},
        "| tiny:",
        tiny["note"],
    )
