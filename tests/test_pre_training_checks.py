import numpy as np
import pandas as pd
import pytest

from src.ml.pre_training_checks import PreTrainingChecks


@pytest.fixture
def checker():
    return PreTrainingChecks()


class TestPreTrainingChecks:
    def test_check_target_exists_pass(self, checker):
        y = pd.Series([1, 2, 3])
        result = checker._check_target_exists(y)
        assert result["status"] == "PASS"

    def test_check_target_exists_fail_empty(self, checker):
        y = pd.Series([], dtype=float)
        result = checker._check_target_exists(y)
        assert result["status"] == "FAIL"

    def test_check_target_exists_fail_none(self, checker):
        result = checker._check_target_exists(None)
        assert result["status"] == "FAIL"

    def test_check_dataset_not_empty_pass(self, checker):
        X = pd.DataFrame({"a": list(range(15))})
        result = checker._check_dataset_not_empty(X)
        assert result["status"] == "PASS"

    def test_check_dataset_not_empty_fail_empty_df(self, checker):
        X = pd.DataFrame()
        result = checker._check_dataset_not_empty(X)
        assert result["status"] == "FAIL"

    def test_check_dataset_not_empty_fail_none(self, checker):
        result = checker._check_dataset_not_empty(None)
        assert result["status"] == "FAIL"

    def test_check_missing_values_pass(self, checker):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = checker._check_missing_values(X)
        assert result["status"] == "PASS"

    def test_check_missing_values_fail(self, checker):
        X = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        result = checker._check_missing_values(X)
        assert result["status"] == "FAIL"

    def test_check_duplicate_samples_flagged(self, checker):
        X = pd.DataFrame({"a": [1, 1, 1, 1, 1, 1]})
        result = checker._check_duplicate_samples(X)
        assert result["status"] == "WARN"

    def test_check_duplicate_samples_pass(self, checker):
        X = pd.DataFrame({"a": [1, 2, 3]})
        result = checker._check_duplicate_samples(X)
        assert result["status"] == "PASS"

    def test_check_feature_variance_zero_var(self, checker):
        X = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
        result = checker._check_feature_variance(X)
        assert result["status"] == "FAIL"
        assert "a" in result["zero_var_cols"]

    def test_check_feature_variance_pass(self, checker):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = checker._check_feature_variance(X)
        assert result["status"] == "PASS"

    def test_check_all_returns_dict(self, checker):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([10, 20, 30])
        result = checker.check_all(X, y)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "checks" in result
        assert "warnings" in result
        assert "errors" in result
        assert "target_exists" in result["checks"]
        assert "dataset_not_empty" in result["checks"]
        assert "missing_values" in result["checks"]
        assert "duplicate_samples" in result["checks"]

    def test_raise_if_failed_raises(self, checker):
        result = {
            "passed": False,
            "checks": {
                "target_exists": {"status": "FAIL", "detail": "y is None"},
            },
        }
        with pytest.raises(ValueError, match="Pre-training checks failed"):
            checker.raise_if_failed(result)

    def test_raise_if_failed_pass_no_raise(self, checker):
        result = {"passed": True, "checks": {}}
        checker.raise_if_failed(result)

    def test_classification_class_balance_fail(self, checker):
        y = pd.Series(["A", "A", "A", "A", "B"])
        result = checker._check_class_balance(y)
        assert result["status"] == "FAIL"

    def test_classification_class_balance_pass(self, checker):
        y = pd.Series(["A", "A", "B", "B", "C", "C"])
        result = checker._check_class_balance(y)
        assert result["status"] == "PASS"
