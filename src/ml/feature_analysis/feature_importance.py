import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("FeatureImportanceAnalyzer")


class FeatureImportanceAnalyzer:
    def __init__(
        self,
        output_dir: Path | None = None,
        random_state: int = 42,
        n_estimators: int = 100,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.importances_: dict[str, pd.DataFrame] = {}
        self.ranked_features_: pd.DataFrame | None = None

    def compute_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_catboost: bool = False,
    ) -> dict[str, pd.DataFrame]:
        if X.empty or y.empty:
            logger.warning("Empty input data")
            return {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns found")
            return {}

        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        results = {}

        rf_imp = self._random_forest_importance(X_num, y)
        results["random_forest"] = rf_imp
        self.importances_["random_forest"] = rf_imp

        if use_xgboost:
            try:
                xgb_imp = self._xgboost_importance(X_num, y)
                results["xgboost"] = xgb_imp
                self.importances_["xgboost"] = xgb_imp
            except Exception as e:
                logger.warning("XGBoost importance failed: %s", e)

        if use_lightgbm:
            try:
                lgb_imp = self._lightgbm_importance(X_num, y)
                results["lightgbm"] = lgb_imp
                self.importances_["lightgbm"] = lgb_imp
            except Exception as e:
                logger.warning("LightGBM importance failed: %s", e)

        if use_catboost:
            try:
                cb_imp = self._catboost_importance(X_num, y)
                results["catboost"] = cb_imp
                self.importances_["catboost"] = cb_imp
            except Exception as e:
                logger.warning("CatBoost importance failed: %s", e)

        self.ranked_features_ = self._aggregate_rankings(results)
        self._save_results()
        return results

    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                    "method": "random_forest",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _xgboost_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X, y)
        gain = model.feature_importances_
        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": gain,
                    "method": "xgboost",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _lightgbm_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(X, y)
        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                    "method": "lightgbm",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _catboost_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=self.n_estimators,
            random_seed=self.random_state,
            verbose=False,
        )
        model.fit(X, y)
        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                    "method": "catboost",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _aggregate_rankings(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()

        combined = pd.concat(results.values(), ignore_index=True)
        agg = (
            combined.groupby("feature")["importance"]
            .agg(["mean", "std", "max", "count"])
            .reset_index()
        )
        agg.columns = [
            "feature",
            "mean_importance",
            "std_importance",
            "max_importance",
            "model_count",
        ]
        agg["rank"] = agg["mean_importance"].rank(ascending=False).astype(int)
        agg = agg.sort_values("rank").reset_index(drop=True)
        n = len(agg)
        agg["importance_tier"] = pd.cut(
            agg["rank"],
            bins=[0, max(1, n // 4), max(1, n // 2), n],
            labels=["top_25", "mid_50", "bottom_25"],
        )
        return agg

    def _save_results(self):

        path = self.output_dir / "feature_importance.json"
        if self.ranked_features_ is not None and not self.ranked_features_.empty:
            data = self.ranked_features_.to_dict(orient="records")
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Saved feature importance to %s", path)

        for method, df in self.importances_.items():
            csv_path = self.output_dir / f"importance_{method}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Saved %s importance to %s", method, csv_path)

    def get_top_features(self, n: int = 50) -> list[str]:
        if self.ranked_features_ is None or self.ranked_features_.empty:
            return []
        return self.ranked_features_.head(n)["feature"].tolist()

    def get_feature_summary(self) -> dict:
        if self.ranked_features_ is None or self.ranked_features_.empty:
            return {}
        return {
            "total_features": len(self.ranked_features_),
            "top_10": self.ranked_features_.head(10)["feature"].tolist(),
            "bottom_10": self.ranked_features_.tail(10)["feature"].tolist(),
            "tier_counts": self.ranked_features_["importance_tier"].value_counts().to_dict(),
        }
