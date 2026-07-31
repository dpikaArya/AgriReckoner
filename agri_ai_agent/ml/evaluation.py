"""Honest, data-scarce model evaluation.

A single hold-out R2 is unreliable at small n and, in this project, was reported to four
decimals on ~2 test points. This module cross-validates every model with imputation done
INSIDE each fold (so no test information leaks into training), reports the spread, and
refuses to emit a skill number when there are too few samples — an explicit
"insufficient data" verdict instead of an inflated score.

Observations here are extracted from papers, and one paper contributes many treatment rows
that share a site, season, soil, and cultivar. Splitting such rows at random puts siblings
of a test row in the training set, so the model is scored on interpolation within a trial it
has already seen. Passing ``groups`` (paper identifiers) switches evaluation to leave-one-
paper-out, which measures what the project actually claims: transfer to an unseen trial.
"""

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    RepeatedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

MIN_ROWS_FOR_METRIC = 8  # below this, refuse to report any skill metric
SMALL_N_THRESHOLD = 30  # below this, use leave-one-out instead of repeated k-fold
MIN_GROUPS_FOR_METRIC = 3  # below this, generalization across papers is unmeasurable
GROUP_KFOLD_THRESHOLD = 10  # at or above this many papers, use grouped k-fold
RANDOM_STATE = 42


def choose_cv(n_samples: int):
    """Pick an evaluation scheme appropriate for the sample size."""
    if n_samples < MIN_ROWS_FOR_METRIC:
        return None, "insufficient"
    if n_samples < SMALL_N_THRESHOLD:
        return LeaveOneOut(), "leave-one-out"
    return RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE), "repeated-5-fold"


def choose_grouped_cv(n_groups: int):
    """Pick a paper-aware scheme; None when too few papers to measure generalization."""
    if n_groups < MIN_GROUPS_FOR_METRIC:
        return None, "insufficient-groups"
    if n_groups < GROUP_KFOLD_THRESHOLD:
        return LeaveOneGroupOut(), "leave-one-paper-out"
    return GroupKFold(n_splits=5), "grouped-5-fold"


def _point_scores(y_true, y_pred) -> dict:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}


def evaluate_model(model, X, y, groups=None) -> dict:
    """Return cross-validated metrics for ``model`` on ``(X, y)``, with an honesty verdict.

    Imputation is fitted inside each fold via a Pipeline. Below MIN_ROWS_FOR_METRIC no
    score is reported; below SMALL_N_THRESHOLD leave-one-out is used and the result is
    flagged non-robust with a caveat. When ``groups`` (one paper id per row) is supplied,
    folds are split by paper so no trial appears in both train and test.
    """
    n_samples = len(y)
    grouped = groups is not None
    if grouped:
        result = _grouped_setup(groups, n_samples)
        if "note" in result:
            return result
        cv, scheme = choose_grouped_cv(result["n_groups"])
    else:
        result = {"n": int(n_samples), "robust": False}
        cv, scheme = choose_cv(n_samples)
    result["cv_scheme"] = scheme

    if cv is None:
        result["note"] = (
            f"insufficient data (n={n_samples} < {MIN_ROWS_FOR_METRIC}); no metric reported"
        )
        return result

    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", clone(model))])
    fit_params = {"groups": groups} if grouped else {}

    if scheme in ("leave-one-out", "leave-one-paper-out"):
        y_pred = cross_val_predict(pipeline, X, y, cv=cv, **fit_params)
        result.update(_point_scores(y, y_pred))
        result["r2_std"] = None
    else:
        r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", **fit_params)
        rmse_scores = -cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", **fit_params
        )
        result["r2"] = round(float(np.mean(r2_scores)), 4)
        result["r2_std"] = round(float(np.std(r2_scores)), 4)
        result["rmse"] = round(float(np.mean(rmse_scores)), 4)
        result["rmse_std"] = round(float(np.std(rmse_scores)), 4)

    result["robust"] = _is_robust(result, n_samples, grouped)
    result["caveat"] = "" if result["robust"] else _caveat(result, n_samples, grouped)
    return result


def _grouped_setup(groups, n_samples: int) -> dict:
    """Build the result skeleton for a grouped evaluation, or an insufficiency verdict."""
    n_groups = int(len(set(np.asarray(groups).tolist())))
    result = {"n": int(n_samples), "n_groups": n_groups, "robust": False}
    if n_groups < MIN_GROUPS_FOR_METRIC:
        result["cv_scheme"] = "insufficient-groups"
        result["note"] = (
            f"insufficient independent papers (n_papers={n_groups} < {MIN_GROUPS_FOR_METRIC}); "
            f"the {n_samples} rows are not independent, so no generalization metric is reported"
        )
    return result


def _is_robust(result: dict, n_samples: int, grouped: bool) -> bool:
    """Robust means enough independent units, not merely enough rows."""
    if grouped:
        return result["n_groups"] >= GROUP_KFOLD_THRESHOLD
    return n_samples >= SMALL_N_THRESHOLD


def _caveat(result: dict, n_samples: int, grouped: bool) -> str:
    if grouped:
        return (
            f"few independent papers (n_papers={result['n_groups']}); "
            "cross-validated across papers but interpret with care"
        )
    return f"small-n (n={n_samples}); cross-validated but interpret with care"


if __name__ == "__main__":
    import numpy as _np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    rng = _np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=40)

    good = evaluate_model(LinearRegression(), X, y)
    assert good["robust"] and good["r2"] > 0.9, good
    tiny = evaluate_model(LinearRegression(), X[:5], y[:5])
    assert not tiny["robust"] and "note" in tiny, tiny

    # Rows cluster by paper: the target is driven by a per-paper offset, not by the
    # features. Random splitting sees siblings and looks skilful; splitting by paper
    # must not, because nothing in X transfers to an unseen paper.
    papers = _np.repeat(_np.arange(12), 5)
    offsets = rng.normal(scale=10.0, size=12)
    Xg = rng.normal(size=(60, 3))
    yg = offsets[papers] + rng.normal(scale=0.1, size=60)
    naive = evaluate_model(RandomForestRegressor(random_state=0), Xg, yg)
    honest = evaluate_model(RandomForestRegressor(random_state=0), Xg, yg, groups=papers)
    assert naive["r2"] > honest["r2"], (naive, honest)
    assert honest["cv_scheme"] == "grouped-5-fold", honest

    one_paper = evaluate_model(LinearRegression(), Xg[:10], yg[:10], groups=_np.zeros(10))
    assert "note" in one_paper, one_paper

    print(
        f"evaluation smoke OK -> ungrouped r2={naive['r2']:.3f} "
        f"vs paper-grouped r2={honest['r2']:.3f} ({honest['cv_scheme']})"
    )
    print("  single-paper verdict:", one_paper["note"])
