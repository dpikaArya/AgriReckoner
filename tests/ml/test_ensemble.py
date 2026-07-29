import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.ml.ensemble.weighted_average import WeightedAverageEnsemble
from src.ml.ensemble.stacking import StackingEnsemble


@pytest.fixture
def ensemble_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "f3": np.random.randn(100),
    })
    y = pd.Series(X["f1"] * 2 + X["f2"] * 0.5 + np.random.randn(100) * 0.2)
    models = {
        "rf": RandomForestRegressor(n_estimators=20, random_state=42),
        "lr": LinearRegression(),
    }
    return models, X, y


class TestWeightedAverage:
    def test_fit_stores_weights(self, ensemble_data):
        models, X, y = ensemble_data
        ensemble = WeightedAverageEnsemble()
        ensemble.fit(models, X, y, auto_weight=True)
        assert len(ensemble.weights_) == len(models)
        assert abs(sum(ensemble.weights_.values()) - 1.0) < 0.01

    def test_predict_returns_array(self, ensemble_data):
        models, X, y = ensemble_data
        ensemble = WeightedAverageEnsemble()
        ensemble.fit(models, X, y, auto_weight=True)
        preds = ensemble.predict(X)
        assert len(preds) == len(X)
        assert isinstance(preds, np.ndarray)

    def test_custom_weights(self, ensemble_data):
        models, X, y = ensemble_data
        ensemble = WeightedAverageEnsemble()
        ensemble.fit(models, X, y, auto_weight=False, weights={"rf": 0.7, "lr": 0.3})
        assert abs(ensemble.weights_["rf"] - 0.7) < 0.01
        assert abs(ensemble.weights_["lr"] - 0.3) < 0.01

    def test_weights_sum_to_one(self, ensemble_data):
        models, X, y = ensemble_data
        ensemble = WeightedAverageEnsemble()
        ensemble.fit(models, X, y, auto_weight=True)
        assert abs(sum(ensemble.weights_.values()) - 1.0) < 0.01

    def test_not_fitted_raises(self):
        ensemble = WeightedAverageEnsemble()
        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(pd.DataFrame({"a": [1, 2, 3]}))


class TestStackingEnsemble:
    def test_fit_sets_fitted_flag(self, ensemble_data):
        models, X, y = ensemble_data
        stack = StackingEnsemble(cv_folds=2)
        stack.fit(models, X, y)
        assert stack.is_fitted_

    def test_predict_returns_array(self, ensemble_data):
        models, X, y = ensemble_data
        stack = StackingEnsemble(cv_folds=2)
        stack.fit(models, X, y)
        preds = stack.predict(X)
        assert len(preds) == len(X)

    def test_not_fitted_raises(self):
        stack = StackingEnsemble()
        with pytest.raises(ValueError, match="not fitted"):
            stack.predict(pd.DataFrame({"a": [1, 2, 3]}))
