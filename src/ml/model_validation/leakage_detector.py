import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("LeakageDetector")


class LeakageDetector:
    def __init__(self, output_dir: Optional[Path] = None, corr_threshold: float = 0.999):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corr_threshold = corr_threshold
        self.leakage_found_ = False
        self.leakage_details_: dict[str, list] = {}

    def check_all(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, list]:
        self.leakage_found_ = False
        self.leakage_details_ = {}

        duplicates = self.check_duplicate_records(X)
        if duplicates:
            self.leakage_found_ = True
            self.leakage_details_["duplicate_records"] = duplicates

        train_test = self.check_train_test_leakage(X)
        if train_test:
            self.leakage_found_ = True
            self.leakage_details_["train_test_leakage"] = train_test

        location = self.check_location_leakage(X)
        if location:
            self.leakage_found_ = True
            self.leakage_details_["location_leakage"] = location

        temporal = self.check_temporal_leakage(X)
        if temporal:
            self.leakage_found_ = True
            self.leakage_details_["temporal_leakage"] = temporal

        target_corr = self.check_target_correlation(X, y)
        if target_corr:
            self.leakage_found_ = True
            self.leakage_details_["target_correlation"] = target_corr

        self._save_results()
        return self.leakage_details_

    def check_duplicate_records(self, X: pd.DataFrame) -> list:
        if X.empty:
            return []
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []
        dup_mask = X[numeric_cols].duplicated(keep=False)
        dup_indices = X.index[dup_mask].tolist()
        if dup_indices:
            dup_count = dup_mask.sum()
            logger.warning("Found %d duplicate records (%.1f%%)", dup_count, dup_count / len(X) * 100)
            return [{"count": int(dup_count), "indices": [int(i) for i in dup_indices[:20]]}]
        return []

    def check_train_test_leakage(self, X: pd.DataFrame) -> list:
        if X.empty:
            return []
        id_cols = [c for c in X.columns if any(id_kw in c.lower()
                   for id_kw in ["paper_id", "experiment_id", "dataset_id", "observation_id"])]
        if not id_cols:
            return []
        issues = []
        for col in id_cols:
            if col in X.columns:
                dup_count = X[col].duplicated().sum()
                if dup_count > 0:
                    issues.append({
                        "column": col,
                        "duplicate_count": int(dup_count),
                        "duplicate_pct": round(dup_count / len(X) * 100, 2),
                    })
                    logger.warning("Train-test leakage: column %s has %d duplicates", col, dup_count)
        return issues

    def check_location_leakage(self, X: pd.DataFrame) -> list:
        loc_cols = [c for c in X.columns if c.lower() in
                    ["location", "region", "country", "state", "district", "site"]]
        if not loc_cols:
            return []
        issues = []
        for col in loc_cols:
            if col in X.columns:
                n_unique = X[col].nunique()
                if n_unique <= 1:
                    issues.append({
                        "column": col,
                        "unique_values": int(n_unique),
                        "issue": "Single location — model may not generalize",
                    })
        return issues

    def check_temporal_leakage(self, X: pd.DataFrame) -> list:
        time_cols = [c for c in X.columns if any(t in c.lower()
                     for t in ["year", "season", "date", "planting", "harvest"])]
        if not time_cols:
            return []
        issues = []
        for col in time_cols[:3]:
            if col in X.columns:
                try:
                    sorted_vals = X[col].dropna().sort_values()
                    if len(sorted_vals) > 1:
                        diff_years = (sorted_vals.iloc[-1] - sorted_vals.iloc[0])
                        issues.append({
                            "column": col,
                            "min": float(sorted_vals.iloc[0]),
                            "max": float(sorted_vals.iloc[-1]),
                            "range": float(diff_years),
                            "note": "Verify temporal split separates train/test by time"
                        })
                except (ValueError, TypeError):
                    pass
        return issues

    def check_target_correlation(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []
        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        corr_with_target = X_num.corrwith(y).abs().sort_values(ascending=False)
        high_corr = corr_with_target[corr_with_target >= self.corr_threshold]
        if not high_corr.empty:
            return [
                {
                    "feature": col,
                    "correlation": round(float(corr), 4),
                    "threshold": self.corr_threshold,
                    "action": "Remove — likely target leakage",
                }
                for col, corr in high_corr.items()
            ]
        return []

    def _save_results(self):
        path = self.output_dir / "leakage_detection_report.json"
        path.write_text(
            json.dumps({
                "leakage_found": self.leakage_found_,
                "details": self.leakage_details_,
                "correlation_threshold": self.corr_threshold,
            }, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Leakage detection complete — found: %s", self.leakage_found_)
        if self.leakage_found_:
            logger.warning("LEAKAGE DETECTED: %s", json.dumps(self.leakage_details_, indent=2))
