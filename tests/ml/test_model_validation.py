import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.ml.model_validation.cross_validation import PipelineCrossValidator
from src.ml.model_validation.leakage_detector import LeakageDetector


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "location": np.random.choice(["A", "B", "C", "D"], 100),
        "year": np.random.randint(2010, 2020, 100),
    })
    y = pd.Series(X["feature_1"] * 2 + np.random.randn(100) * 0.5)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    return model, X, y


class TestCrossValidation:
    def test_kfold_returns_expected_keys(self, simple_data):
        model, X, y = simple_data
        validator = PipelineCrossValidator()
        result = validator.validate_kfold(model, X, y, n_splits=3)
        assert "method" in result
        assert "r2_mean" in result
        assert "rmse_mean" in result

    def test_kfold_r2_reasonable(self, simple_data):
        model, X, y = simple_data
        validator = PipelineCrossValidator()
        result = validator.validate_kfold(model, X, y, n_splits=3)
        assert result.get("r2_mean") is None or result["r2_mean"] > -1

    def test_spatial_validation_fallback(self, simple_data):
        model, X, y = simple_data
        validator = PipelineCrossValidator()
        result = validator.validate_spatial(model, X, y, location_col="location", n_splits=2)
        assert "method" in result

    def test_missing_location_falls_back(self, simple_data):
        model, X, y = simple_data
        validator = PipelineCrossValidator()
        result = validator.validate_spatial(model, X, y, location_col="nonexistent", n_splits=2)
        assert result["method"] == "kfold"


class TestLeakageDetector:
    def test_no_leakage_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        })
        X_unique = pd.concat([
            X,
            pd.DataFrame({"f1": np.random.randn(50) + 10, "f2": np.random.randn(50) + 10}),
        ], ignore_index=True)
        y = pd.Series(np.random.randn(100))
        detector = LeakageDetector()
        result = detector.check_all(X_unique, y)
        assert not detector.leakage_found_

    def test_duplicate_records_detected(self):
        X = pd.DataFrame({"a": [1, 2, 1, 2], "b": [3, 4, 3, 4]})
        y = pd.Series([10, 20, 10, 20])
        detector = LeakageDetector()
        result = detector.check_all(X, y)
        assert "duplicate_records" in result

    def test_target_leakage_detected(self):
        np.random.seed(42)
        y = pd.Series(np.random.randn(50))
        X = pd.DataFrame({
            "f1": np.random.randn(50),
            "leaky_feature": y * 0.999 + np.random.randn(50) * 0.001,
        })
        detector = LeakageDetector(corr_threshold=0.9)
        result = detector.check_all(X, y)
        assert "target_correlation" in result

    def test_empty_dataframe(self):
        detector = LeakageDetector()
        result = detector.check_all(pd.DataFrame(), pd.Series(dtype=float))
        assert isinstance(result, dict)
        assert not detector.leakage_found_
