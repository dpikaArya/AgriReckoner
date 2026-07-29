import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("CorrelationAnalyzer")


class CorrelationAnalyzer:
    def __init__(self, output_dir: Optional[Path] = None, threshold: float = 0.9):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.corr_matrix_: Optional[pd.DataFrame] = None
        self.high_corr_pairs_: list[tuple[str, str, float]] = []

    def analyze(self, df: pd.DataFrame) -> dict:
        if df.empty:
            logger.warning("Empty DataFrame for correlation analysis")
            return {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns for correlation analysis")
            return {}

        df_num = df[numeric_cols].fillna(df[numeric_cols].median())

        self.corr_matrix_ = df_num.corr(method="pearson")
        self._find_high_correlations()
        result = self._build_summary(df_num)
        self._save_results(df_num)
        return result

    def _find_high_correlations(self):
        if self.corr_matrix_ is None:
            return
        corr = self.corr_matrix_.abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        pairs = [
            (col, row, self.corr_matrix_.loc[row, col])
            for row in upper.index
            for col in upper.columns
            if not pd.isna(upper.loc[row, col]) and upper.loc[row, col] >= self.threshold
        ]
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        self.high_corr_pairs_ = pairs

    def _build_summary(self, df: pd.DataFrame) -> dict:
        if self.corr_matrix_ is None:
            return {}
        corr_abs = self.corr_matrix_.abs()
        mean_corr = corr_abs.mean(axis=1).sort_values(ascending=False)

        return {
            "total_numeric_columns": len(df.columns),
            "high_correlation_pairs_count": len(self.high_corr_pairs_),
            "high_correlation_threshold": self.threshold,
            "mean_correlation_by_feature": mean_corr.head(20).to_dict(),
            "top_correlated_pairs": [
                {"feature_1": p[0], "feature_2": p[1], "correlation": round(p[2], 4)}
                for p in self.high_corr_pairs_[:30]
            ],
            "redundant_features": self._identify_redundant_features(),
        }

    def _identify_redundant_features(self) -> list[str]:
        if self.corr_matrix_ is None:
            return []
        corr = self.corr_matrix_.abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] >= self.threshold):
                if col not in to_drop:
                    correlated_with = upper.index[upper[col] >= self.threshold].tolist()
                    to_drop.add(col)
        return sorted(to_drop)

    def _save_results(self, df: pd.DataFrame):
        if self.corr_matrix_ is not None:
            corr_path = self.output_dir / "correlation_matrix.csv"
            self.corr_matrix_.to_csv(corr_path)
            logger.info("Saved correlation matrix to %s", corr_path)

        redundant = self._identify_redundant_features()
        summary = {
            "total_features": len(df.columns) if df is not None else 0,
            "correlation_threshold": self.threshold,
            "redundant_features_count": len(redundant),
            "redundant_features": redundant,
            "high_correlation_pairs_count": len(self.high_corr_pairs_),
        }
        path = self.output_dir / "correlation_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Saved correlation summary to %s", path)

    def get_redundant_features(self) -> list[str]:
        return self._identify_redundant_features()
