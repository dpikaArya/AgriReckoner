import numpy as np
import pandas as pd
import pytest

from src.validation.data_quality import DataQualityValidator


@pytest.fixture
def validator():
    return DataQualityValidator()


class TestDataQualityValidator:
    def test_validate_returns_dict(self, validator):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validator.validate(df)
        assert isinstance(result, dict)
        assert "total_rows" in result
        assert "total_columns" in result
        assert "missing_values" in result
        assert "duplicate_rows" in result
        assert "outliers" in result

    def test_missing_values_detected(self, validator):
        df = pd.DataFrame(
            {
                "x": [1.0, np.nan, 3.0, np.nan],
                "y": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        result = validator.validate(df)
        assert result["missing_values"]["x"]["count"] == 2
        assert result["missing_values"]["y"]["count"] == 4
        assert result["missing_values"]["x"]["percentage"] == 50.0

    def test_duplicate_rows_detected(self, validator):
        df = pd.DataFrame({"a": [1, 2, 2, 3, 3, 3]})
        result = validator.validate(df)
        assert result["duplicate_rows"]["count"] == 3

    def test_no_duplicates_returns_zero(self, validator):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = validator.validate(df)
        assert result["duplicate_rows"]["count"] == 0

    def test_outliers_iqr_detected(self, validator):
        values = list(range(100)) + [1000, 2000]
        df = pd.DataFrame({"x": values})
        result = validator.validate(df)
        assert result["outliers"]["x"]["count"] > 0

    def test_outliers_zscore_detected(self, validator):
        values = list(range(100)) + [500, 600]
        series = pd.Series(values)
        mask = validator.detect_outliers_zscore(series, threshold=2.0)
        assert mask.sum() > 0

    def test_outliers_zscore_no_std(self, validator):
        series = pd.Series([5.0, 5.0, 5.0])
        mask = validator.detect_outliers_zscore(series)
        assert mask.sum() == 0

    def test_duplicate_by_subset(self, validator):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        result = validator.validate(df, key_columns=["a"])
        assert result["duplicate_by_subset"]["count"] == 1
