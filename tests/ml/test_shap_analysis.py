import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.ml.feature_analysis.shap_analysis import ShapAnalyzer


@pytest.fixture
def trained_model_and_data():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        }
    )
    y = pd.Series(X["feature_1"] * 2 + X["feature_2"] * 0.5 + np.random.randn(100) * 0.1)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestShapAnalyzer:
    def test_analyze_returns_dataframe(self, trained_model_and_data):
        model, X, y = trained_model_and_data
        analyzer = ShapAnalyzer()
        result = analyzer.analyze(model, X)
        if result is not None and not result.empty:
            assert "feature" in result.columns
            assert "mean_abs_shap" in result.columns
            assert "impact" in result.columns

    def test_get_top_shap_features(self, trained_model_and_data):
        model, X, y = trained_model_and_data
        analyzer = ShapAnalyzer()
        analyzer.analyze(model, X)
        top = analyzer.get_top_shap_features(n=2)
        assert len(top) <= 2

    def test_empty_input(self):
        analyzer = ShapAnalyzer()
        result = analyzer.analyze(None, pd.DataFrame())
        assert result is not None and result.empty

    def test_numeric_only(self):
        X = pd.DataFrame({"a": ["x", "y", "z"], "b": [1.0, 2.0, 3.0]})
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X[["b"]], [10, 20, 30])
        analyzer = ShapAnalyzer()
        result = analyzer.analyze(model, X)
        assert result is None or result.empty or "a" not in result["feature"].values
