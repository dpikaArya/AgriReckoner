"""Tests for honest, data-scarce evaluation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from agri_ai_agent.ml.evaluation import (
    MIN_GROUPS_FOR_METRIC,
    MIN_ROWS_FOR_METRIC,
    SMALL_N_THRESHOLD,
    choose_cv,
    choose_grouped_cv,
    evaluate_model,
)


def _signal(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=n)
    return X, y


def _clustered_by_paper(n_papers=12, per_paper=5, seed=0):
    """Rows whose target is set by a per-paper offset, unrelated to the features.

    Nothing in X transfers to an unseen paper, so an honest paper-aware evaluation
    must not credit the model with skill.
    """
    rng = np.random.default_rng(seed)
    papers = np.repeat(np.arange(n_papers), per_paper)
    offsets = rng.normal(scale=10.0, size=n_papers)
    X = rng.normal(size=(n_papers * per_paper, 3))
    y = offsets[papers] + rng.normal(scale=0.1, size=n_papers * per_paper)
    return X, y, papers


def test_choose_cv_by_sample_size():
    assert choose_cv(3)[1] == "insufficient"
    assert choose_cv(15)[1] == "leave-one-out"
    assert choose_cv(60)[1] == "repeated-5-fold"


def test_choose_grouped_cv_by_paper_count():
    assert choose_grouped_cv(2)[1] == "insufficient-groups"
    assert choose_grouped_cv(5)[1] == "leave-one-paper-out"
    assert choose_grouped_cv(20)[1] == "grouped-5-fold"


def test_single_paper_reports_no_generalization_metric():
    """Many rows from one trial are not many observations."""
    X, y, _ = _clustered_by_paper()
    result = evaluate_model(LinearRegression(), X[:10], y[:10], groups=np.zeros(10))
    assert result["cv_scheme"] == "insufficient-groups"
    assert "r2" not in result
    assert "note" in result
    assert result["n_groups"] == 1


def test_too_few_papers_is_refused_even_with_enough_rows():
    X, y, papers = _clustered_by_paper(n_papers=MIN_GROUPS_FOR_METRIC - 1, per_paper=20)
    result = evaluate_model(LinearRegression(), X, y, groups=papers)
    assert "r2" not in result
    assert result["robust"] is False


def test_paper_grouping_removes_the_sibling_row_advantage():
    """Random folds let a paper's siblings train the model that is scored on it."""
    X, y, papers = _clustered_by_paper()
    model = RandomForestRegressor(random_state=0, n_estimators=50)
    naive = evaluate_model(model, X, y)
    honest = evaluate_model(model, X, y, groups=papers)
    assert honest["cv_scheme"] == "grouped-5-fold"
    assert honest["r2"] < naive["r2"]
    assert honest["n_groups"] == 12


def test_grouped_robustness_counts_papers_not_rows():
    """600 rows from 4 papers is still 4 independent units."""
    X, y, papers = _clustered_by_paper(n_papers=4, per_paper=150)
    result = evaluate_model(LinearRegression(), X, y, groups=papers)
    assert result["n"] == 600
    assert result["n_groups"] == 4
    assert result["robust"] is False
    assert "few independent papers" in result["caveat"]


def test_insufficient_data_reports_no_metric():
    X, y = _signal(MIN_ROWS_FOR_METRIC - 2)
    result = evaluate_model(LinearRegression(), X, y)
    assert result["robust"] is False
    assert "r2" not in result
    assert "note" in result


def test_small_n_is_flagged_not_robust():
    X, y = _signal(15)
    result = evaluate_model(LinearRegression(), X, y)
    assert result["cv_scheme"] == "leave-one-out"
    assert result["robust"] is False
    assert result["caveat"]
    assert "r2" in result


def test_large_n_is_robust_with_spread():
    X, y = _signal(SMALL_N_THRESHOLD + 30)
    result = evaluate_model(LinearRegression(), X, y)
    assert result["robust"] is True
    assert result["r2"] > 0.9
    assert result["r2_std"] is not None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
