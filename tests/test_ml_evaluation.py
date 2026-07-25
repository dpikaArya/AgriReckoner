"""Tests for honest, data-scarce evaluation."""

import numpy as np
from sklearn.linear_model import LinearRegression

from agri_ai_agent.ml.evaluation import (
    MIN_ROWS_FOR_METRIC,
    SMALL_N_THRESHOLD,
    choose_cv,
    evaluate_model,
)


def _signal(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=n)
    return X, y


def test_choose_cv_by_sample_size():
    assert choose_cv(3)[1] == "insufficient"
    assert choose_cv(15)[1] == "leave-one-out"
    assert choose_cv(60)[1] == "repeated-5-fold"


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
