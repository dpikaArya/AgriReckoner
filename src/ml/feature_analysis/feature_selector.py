import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("FeatureSelector")


class FeatureSelector:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.90,
        variance_threshold: float = 0.001,
        max_features: int = 150,
        min_features: int = 80,
        max_missing_pct: float = 1.0,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.max_missing_pct = max_missing_pct
        self.selected_features_: list[str] = []
        self.removed_features_: dict[str, list[str]] = {}

    def select(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        importance_df: Optional[pd.DataFrame] = None,
        shap_df: Optional[pd.DataFrame] = None,
        non_feature_cols: Optional[list[str]] = None,
        must_keep: Optional[list[str]] = None,
    ) -> list[str]:
        if X.empty:
            logger.warning("Empty DataFrame for feature selection")
            return []

        non_feature_cols = non_feature_cols or []
        must_keep = must_keep or []
        removed: dict[str, list[str]] = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[numeric_cols].fillna(X[numeric_cols].median())

        candidates = [c for c in X_num.columns if c not in non_feature_cols]
        for col in must_keep:
            if col not in candidates:
                candidates.append(col)

        initial_count = len(candidates)
        logger.info("Starting feature selection with %d candidates", initial_count)

        high_missing = self._remove_high_missingness(X_num, candidates)
        removed["high_missingness"] = high_missing
        candidates = [c for c in candidates if c not in high_missing]
        logger.info("After high-missingness removal: %d features", len(candidates))

        low_var = self._remove_low_variance(X_num, candidates)
        removed["low_variance"] = low_var
        candidates = [c for c in candidates if c not in low_var]
        logger.info("After low-variance removal: %d features", len(candidates))

        high_corr = self._remove_highly_correlated(X_num, candidates)
        removed["high_correlation"] = high_corr
        candidates = [c for c in candidates if c not in high_corr]
        logger.info("After correlation removal: %d features", len(candidates))

        if importance_df is not None and not importance_df.empty:
            low_imp = self._remove_low_importance(importance_df, candidates)
            removed["low_importance"] = low_imp
            candidates = [c for c in candidates if c not in low_imp]
            logger.info("After importance filter: %d features", len(candidates))

        if shap_df is not None and not shap_df.empty:
            self._apply_shap_boost(shap_df, candidates)

        if len(candidates) > self.max_features:
            trimmed = self._trim_to_max(candidates, importance_df, shap_df)
            removed["trimmed_excess"] = [c for c in candidates if c not in trimmed]
            candidates = trimmed
            logger.info("After max trim: %d features", len(candidates))

        if len(candidates) < self.min_features and must_keep:
            added = self._restore_from_must_keep(
                candidates, must_keep, initial_count
            )
            candidates = added
            logger.info("After restoring must-keep: %d features", len(candidates))

        self.selected_features_ = candidates
        self.removed_features_ = removed
        self._save_selection(X_num, y)
        self._generate_report(initial_count)
        return candidates

    def _remove_high_missingness(
        self, X: pd.DataFrame, candidates: list[str]
    ) -> list[str]:
        if self.max_missing_pct >= 1.0:
            return []
        removed = []
        n = len(X)
        for col in candidates:
            if col not in X.columns:
                continue
            missing_pct = X[col].isna().sum() / n
            if missing_pct > self.max_missing_pct:
                removed.append(col)
        return removed

    def _remove_low_variance(
        self, X: pd.DataFrame, candidates: list[str]
    ) -> list[str]:
        removed = []
        for col in candidates:
            if col not in X.columns:
                continue
            var = X[col].var()
            if pd.isna(var) or var <= self.variance_threshold:
                removed.append(col)
        return removed

    def _remove_highly_correlated(
        self, X: pd.DataFrame, candidates: list[str]
    ) -> list[str]:
        if len(candidates) <= 1:
            return []
        corr = X[candidates].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            high = upper.index[upper[col] >= self.correlation_threshold].tolist()
            for h in high:
                to_drop.add(h)
        return sorted(to_drop)

    def _remove_low_importance(
        self, importance_df: pd.DataFrame, candidates: list[str]
    ) -> list[str]:
        if "feature" not in importance_df.columns:
            return []
        imp_map = {}
        if "mean_importance" in importance_df.columns:
            imp_map = dict(
                zip(importance_df["feature"], importance_df["mean_importance"])
            )
        elif "importance" in importance_df.columns:
            imp_map = dict(
                zip(importance_df["feature"], importance_df["importance"])
            )
        removed = []
        for col in candidates:
            imp = imp_map.get(col, 0)
            if imp < self.importance_threshold:
                removed.append(col)
        return removed

    def _apply_shap_boost(
        self, shap_df: pd.DataFrame, candidates: list[str]
    ):
        if "feature" not in shap_df.columns or "impact" not in shap_df.columns:
            return
        high_impact = shap_df[shap_df["impact"] == "high"]["feature"].tolist()
        for col in high_impact:
            if col not in candidates:
                candidates.append(col)

    def _trim_to_max(
        self,
        candidates: list[str],
        importance_df: Optional[pd.DataFrame],
        shap_df: Optional[pd.DataFrame],
    ) -> list[str]:
        scores = {}
        for col in candidates:
            scores[col] = 0
        if importance_df is not None and "feature" in importance_df.columns:
            col = "mean_importance" if "mean_importance" in importance_df.columns else "importance"
            for _, row in importance_df.iterrows():
                if row["feature"] in scores:
                    scores[row["feature"]] += row[col]
        if shap_df is not None and "feature" in shap_df.columns:
            for _, row in shap_df.iterrows():
                if row["feature"] in scores:
                    scores[row["feature"]] += row.get("mean_abs_shap", 0)
        sorted_cols = sorted(scores, key=scores.get, reverse=True)
        return sorted_cols[: self.max_features]

    def _restore_from_must_keep(
        self, candidates: list[str], must_keep: list[str], max_possible: int
    ) -> list[str]:
        result = list(candidates)
        for col in must_keep:
            if col not in result and len(result) < max_possible:
                result.append(col)
        return result

    def _save_selection(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        features_dir = Path("features")
        features_dir.mkdir(parents=True, exist_ok=True)

        full_path = features_dir / "full_features.csv"
        X.to_csv(full_path, index=False)
        logger.info("Saved full features to %s", full_path)

        if self.selected_features_:
            selected = [c for c in self.selected_features_ if c in X.columns]
            opt_path = features_dir / "optimized_features.csv"
            X[selected].to_csv(opt_path, index=False)
            logger.info("Saved optimized features to %s", opt_path)

            selected_list = {
                "selected_features": selected,
                "count": len(selected),
                "removed_features": self.removed_features_,
                "total_removed": sum(len(v) for v in self.removed_features_.values()),
            }
            json_path = features_dir / "selected_feature_list.json"
            json_path.write_text(
                json.dumps(selected_list, indent=2, default=str), encoding="utf-8"
            )

        if y is not None:
            target_path = features_dir / "target_variable.csv"
            y.to_csv(target_path, index=False)

    def _generate_report(self, initial_count: int):
        selected_count = len(self.selected_features_)
        removed_count = sum(len(v) for v in self.removed_features_.values())
        report = {
            "initial_features": initial_count,
            "selected_features": selected_count,
            "total_removed": removed_count,
            "reduction_pct": round(
                (1 - selected_count / max(initial_count, 1)) * 100, 2
            ),
            "removed_by_reason": {
                k: {"count": len(v), "features": v}
                for k, v in self.removed_features_.items()
            },
            "thresholds": {
                "importance": self.importance_threshold,
                "correlation": self.correlation_threshold,
                "variance": self.variance_threshold,
                "max_features": self.max_features,
                "min_features": self.min_features,
                "max_missing_pct": self.max_missing_pct,
            },
            "selected_features": self.selected_features_,
        }
        path = self.output_dir / "feature_selection_report.json"
        path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Feature selection: %d → %d features (%.1f%% reduction)",
            initial_count,
            selected_count,
            report["reduction_pct"],
        )
