from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ml.hyperparameter_optimizer import HyperparameterOptimizer


@pytest.fixture
def small_dataset():
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(50),
        "f2": np.random.randn(50),
        "f3": np.random.randn(50),
    })
    y = pd.Series(X["f1"] * 2 + X["f2"] + np.random.randn(50) * 0.1)
    return X, y


class TestHyperparameterOptimizer:
    def test_optimize_xgboost_returns_dict(self, small_dataset):
        X, y = small_dataset
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=3)
        result = optimizer.optimize_xgboost(X, y)
        assert "best_params" in result
        assert "n_trials" in result

    def test_optimize_random_forest_returns_dict(self, small_dataset):
        X, y = small_dataset
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=3)
        result = optimizer.optimize_random_forest(X, y)
        assert "best_params" in result
        assert "n_trials" in result

    def test_output_dirs_created_with_optuna(self, small_dataset):
        X, y = small_dataset
        try:
            import optuna
            optimizer = HyperparameterOptimizer(output_dir="models/optimization", n_trials=1, cv_folds=2)
            optimizer.optimize_xgboost(X, y)
            assert Path("models/optimization/xgboost_best_params.json").exists()
        except ImportError:
            pytest.skip("optuna not installed")

    def test_default_params_returned_without_optuna(self, small_dataset):
        X, y = small_dataset
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=3)
        result = optimizer.optimize_xgboost(X, y)
        assert "best_params" in result
        assert "n_estimators" in result["best_params"]

    def test_too_small_dataset(self):
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([1, 2])
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=5)
        result = optimizer.optimize_xgboost(X, y)
        assert result["best_value"] is None or result["best_value"] < 0
