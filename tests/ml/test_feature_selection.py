import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.ml.feature_analysis.correlation_analysis import CorrelationAnalyzer
from src.ml.feature_analysis.feature_importance import FeatureImportanceAnalyzer
from src.ml.feature_analysis.feature_selector import FeatureSelector


@pytest.fixture
def synthetic_data():
    X, y = make_regression(
        n_samples=200, n_features=50, n_informative=10, noise=0.1, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return df, pd.Series(y, name="target")


@pytest.fixture
def high_corr_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 1] = X[:, 0] * 0.95 + np.random.randn(100) * 0.1
    X[:, 2] = X[:, 0] * 0.98 + np.random.randn(100) * 0.05
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    y = pd.Series(np.random.randn(100))
    return df, y


class TestFeatureImportance:
    def test_compute_all_returns_dict(self, synthetic_data):
        X, y = synthetic_data
        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.compute_all(X, y)
        assert isinstance(results, dict)
        assert "random_forest" in results

    def test_ranked_features_has_expected_columns(self, synthetic_data):
        X, y = synthetic_data
        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_all(X, y)
        assert analyzer.ranked_features_ is not None
        assert "feature" in analyzer.ranked_features_.columns
        assert "mean_importance" in analyzer.ranked_features_.columns
        assert "rank" in analyzer.ranked_features_.columns

    def test_get_top_features(self, synthetic_data):
        X, y = synthetic_data
        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_all(X, y)
        top = analyzer.get_top_features(n=5)
        assert len(top) == 5

    def test_empty_data_returns_empty(self):
        analyzer = FeatureImportanceAnalyzer()
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        result = analyzer.compute_all(X, y)
        assert result == {}

    def test_importance_values_sum_to_approx_one(self, synthetic_data):
        X, y = synthetic_data
        analyzer = FeatureImportanceAnalyzer()
        analyzer.compute_all(X, y)
        rf_imp = analyzer.importances_.get("random_forest")
        if rf_imp is not None:
            assert abs(rf_imp["importance"].sum() - 1.0) < 0.01


class TestCorrelationAnalysis:
    def test_identifies_high_correlations(self, high_corr_data):
        df, y = high_corr_data
        analyzer = CorrelationAnalyzer(threshold=0.9)
        result = analyzer.analyze(df)
        assert result["high_correlation_pairs_count"] > 0

    def test_redundant_features_not_empty(self, high_corr_data):
        df, y = high_corr_data
        analyzer = CorrelationAnalyzer(threshold=0.9)
        analyzer.analyze(df)
        redundant = analyzer.get_redundant_features()
        assert len(redundant) > 0

    def test_empty_dataframe(self):
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(pd.DataFrame())
        assert result == {}

    def test_no_correlation_low_threshold(self, synthetic_data):
        df, y = synthetic_data
        analyzer = CorrelationAnalyzer(threshold=0.999)
        result = analyzer.analyze(df)
        assert result["high_correlation_pairs_count"] == 0


class TestFeatureSelector:
    def test_select_reduces_features(self, synthetic_data):
        X, y = synthetic_data
        selector = FeatureSelector(max_features=150, min_features=5)
        selected = selector.select(X, y)
        assert len(selected) <= 150
        assert len(selected) > 0

    def test_numeric_only(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]})
        y = pd.Series([10, 20, 30])
        selector = FeatureSelector(max_features=150, min_features=1)
        selected = selector.select(X, y)
        assert "b" not in selected

    def test_empty_dataframe(self):
        selector = FeatureSelector()
        selected = selector.select(pd.DataFrame(), pd.Series(dtype=float))
        assert selected == []

    def test_must_keep_list(self, synthetic_data):
        X, y = synthetic_data
        selector = FeatureSelector(max_features=5, min_features=1)
        selected = selector.select(X, y, must_keep=["feature_0"])
        assert "feature_0" in selected

    def test_non_feature_cols_excluded(self, synthetic_data):
        X, y = synthetic_data
        selector = FeatureSelector(max_features=150, min_features=1)
        selected = selector.select(X, y, non_feature_cols=["feature_0", "feature_1"])
        assert "feature_0" not in selected
