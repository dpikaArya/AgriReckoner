import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score,
)

logger = logging.getLogger("PipelineCrossValidator")


class PipelineCrossValidator:
    def __init__(self, output_dir: Optional[Path] = None, random_state: int = 42):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.results_: dict[str, dict] = {}

    def validate_kfold(
        self, model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> dict:
        if len(X) < n_splits:
            n_splits = max(2, len(X))

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return self._run_cv("kfold", model, X, y, cv)

    def validate_spatial(
        self, model, X: pd.DataFrame, y: pd.Series, location_col: str, n_splits: int = 3
    ) -> dict:
        if location_col not in X.columns:
            logger.warning("Location column '%s' not found; using KFold", location_col)
            return self.validate_kfold(model, X, y, n_splits)

        locations = X[location_col].fillna("unknown")
        cv = GroupKFold(n_splits=min(n_splits, locations.nunique()))
        try:
            return self._run_cv("spatial", model, X, y, cv, groups=locations)
        except Exception as e:
            logger.warning("Spatial validation failed: %s", e)
            return self.validate_kfold(model, X, y, n_splits)

    def validate_temporal(
        self, model, X: pd.DataFrame, y: pd.Series, time_col: str, n_splits: int = 3
    ) -> dict:
        if time_col not in X.columns:
            logger.warning("Time column '%s' not found; using KFold", time_col)
            return self.validate_kfold(model, X, y, n_splits)

        sorted_idx = X[time_col].argsort()
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]
        cv = TimeSeriesSplit(n_splits=n_splits)
        return self._run_cv("temporal", model, X_sorted, y_sorted, cv)

    def _run_cv(
        self, method: str, model, X: pd.DataFrame, y: pd.Series, cv, groups=None
    ) -> dict:
        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))

        try:
            kwargs = {"X": X_filled, "y": y, "cv": cv, "scoring": "neg_root_mean_squared_error", "n_jobs": -1}
            if groups is not None:
                kwargs["groups"] = groups

            rmse_scores = cross_val_score(**kwargs)
            kwargs["scoring"] = "r2"
            r2_scores = cross_val_score(**kwargs)

            result = {
                "method": method,
                "n_splits": cv.get_n_splits(),
                "rmse_mean": round(float(-rmse_scores.mean()), 4),
                "rmse_std": round(float(rmse_scores.std()), 4),
                "r2_mean": round(float(r2_scores.mean()), 4),
                "r2_std": round(float(r2_scores.std()), 4),
                "rmse_per_fold": [round(float(s), 4) for s in -rmse_scores],
                "r2_per_fold": [round(float(s), 4) for s in r2_scores],
            }
        except Exception as e:
            logger.warning("CV failed for method %s: %s", method, e)
            result = {
                "method": method,
                "error": str(e),
                "rmse_mean": None,
                "r2_mean": None,
            }

        self.results_[method] = result
        logger.info(
            "%s CV — RMSE: %.4f ± %.4f, R2: %.4f ± %.4f",
            method,
            result.get("rmse_mean", 0),
            result.get("rmse_std", 0),
            result.get("r2_mean", 0),
            result.get("r2_std", 0),
        )
        return result

    def validate_all(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        location_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> dict:
        results = {}
        results["kfold"] = self.validate_kfold(model, X, y)

        if location_col:
            results["spatial"] = self.validate_spatial(model, X, y, location_col)
        if time_col:
            results["temporal"] = self.validate_temporal(model, X, y, time_col)

        self._save_report(results)
        return results

    def _save_report(self, results: dict):
        path = self.output_dir / "validation_report.json"
        path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        logger.info("Saved validation report to %s", path)
