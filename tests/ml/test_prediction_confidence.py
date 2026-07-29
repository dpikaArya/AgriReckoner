import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.ml.prediction_confidence import PredictionConfidence, RiskLevel


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
    y = pd.Series(X["f1"] * 2 + np.random.randn(50) * 0.5)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestPredictionConfidence:
    def test_estimate_returns_dataframe(self, trained_model):
        model, X, y = trained_model
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred, y_actual=y.values)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert "confidence" in result.columns
        assert "risk" in result.columns

    def test_confidence_between_0_and_1(self, trained_model):
        model, X, y = trained_model
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred, y_actual=y.values)
        assert result["confidence"].between(0, 1).all()

    def test_risk_levels_valid(self, trained_model):
        model, X, y = trained_model
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred)
        assert all(r in ("low", "medium", "high") for r in result["risk"])

    def test_prediction_interval_present(self, trained_model):
        model, X, y = trained_model
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred)
        assert "prediction_lower" in result.columns
        assert "prediction_upper" in result.columns
        assert (result["prediction_lower"] <= result["prediction_upper"]).all()

    def test_format_prediction(self, trained_model):
        model, X, y = trained_model
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred, y_actual=y.values)
        formatted = conf.format_prediction(result, idx=0)
        assert "Prediction:" in formatted
        assert "Confidence:" in formatted
        assert "Risk:" in formatted

    def test_high_confidence_on_good_fit(self):
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(50)})
        y = pd.Series(X["f1"] * 3 + np.random.randn(50) * 0.01)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        conf = PredictionConfidence()
        result = conf.estimate(model, X, y_pred, y_actual=y.values)
        assert result["confidence"].mean() > 0.5
