import logging

import numpy as np
import pandas as pd


class DataQualityValidator:
    def __init__(
        self,
        missing_threshold: float = 0.5,
        duplicate_threshold: float = 0.0,
        logger: logging.Logger | None = None,
    ):
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold
        self.logger = logger or logging.getLogger(__name__)

    def validate(self, df: pd.DataFrame, key_columns: list[str] | None = None) -> dict:
        total_rows = len(df)
        total_columns = len(df.columns)

        missing_values = {}
        for col in df.columns:
            count = int(df[col].isnull().sum())
            percentage = round(count / total_rows * 100, 2) if total_rows > 0 else 0.0
            missing_values[col] = {"count": count, "percentage": percentage}

        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_rows_pct = (
            round(duplicate_rows_count / total_rows * 100, 2) if total_rows > 0 else 0.0
        )

        duplicate_by_subset = {}
        if key_columns:
            valid_keys = [k for k in key_columns if k in df.columns]
            if valid_keys:
                dup_subset = df.duplicated(subset=valid_keys).sum()
                duplicate_by_subset = {
                    "columns": valid_keys,
                    "count": int(dup_subset),
                    "percentage": (
                        round(dup_subset / total_rows * 100, 2) if total_rows > 0 else 0.0
                    ),
                }

        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            mask = self.detect_outliers_iqr(df[col].dropna())
            outlier_count = int(mask.sum())
            outlier_pct = round(outlier_count / total_rows * 100, 2) if total_rows > 0 else 0.0
            outliers[col] = {
                "count": outlier_count,
                "percentage": outlier_pct,
            }

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "missing_values": missing_values,
            "duplicate_rows": {
                "count": duplicate_rows_count,
                "percentage": duplicate_rows_pct,
            },
            "duplicate_by_subset": duplicate_by_subset,
            "outliers": outliers,
        }

    def detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)

    def detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        z = (series - mean) / std
        return z.abs() > threshold
