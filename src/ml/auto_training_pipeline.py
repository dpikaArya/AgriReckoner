import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml.data_quality.scoring import DatasetScorer
from src.ml.feature_analysis.correlation_analysis import CorrelationAnalyzer
from src.ml.feature_analysis.feature_importance import FeatureImportanceAnalyzer
from src.ml.feature_analysis.feature_selector import FeatureSelector
from src.ml.feature_analysis.shap_analysis import ShapAnalyzer
from src.ml.hyperparameter_optimizer import HyperparameterOptimizer
from src.ml.model_benchmark import ModelBenchmark
from src.ml.model_validation.cross_validation import PipelineCrossValidator
from src.ml.model_validation.leakage_detector import LeakageDetector
from src.ml.prediction_confidence import PredictionConfidence

logger = logging.getLogger("AutoTrainingPipeline")


class AutoTrainingPipeline:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        random_state: int = 42,
        config: Optional[dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.config = config or self._default_config()
        self.pipeline_state: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "status": "initialized",
            "errors": [],
        }

    def _default_config(self) -> dict:
        return {
            "importance_threshold": 0.001,
            "correlation_threshold": 0.95,
            "variance_threshold": 1e-12,
            "max_features": 200,
            "min_features": 80,
            "max_missing_pct": 0.7,
            "n_trials": 200,
            "cv_folds": 10,
            "test_size": 0.2,
            "use_shap": True,
            "use_ensemble": True,
            "use_hyperparameter_opt": True,
            "use_data_scoring": True,
            "min_acceptable_r2": 0.89,
            "use_loocv": True,
            "loocv_models": ["ridge", "lasso", "random_forest", "gradient_boosting"],
        }

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        non_feature_cols: Optional[list[str]] = None,
        must_keep_features: Optional[list[str]] = None,
        location_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> dict[str, Any]:
        self.pipeline_state["status"] = "running"
        logger.info("=" * 60)
        logger.info("Auto Training Pipeline Started")
        logger.info("=" * 60)

        if X.empty or y.empty:
            raise ValueError("Empty input data")

        non_feature_cols = non_feature_cols or []
        must_keep_features = must_keep_features or []

        try:
            step1_data = self._step_dataset_validation(X, y)
            if step1_data.get("leakage_found"):
                logger.error("Pipeline stopped: leakage detected")
                self.pipeline_state["status"] = "failed"
                return self.pipeline_state

            step2_score = self._step_data_scoring(X, non_feature_cols)
            step3_features = self._step_feature_analysis(X, y, non_feature_cols)
            step4_selected = self._step_feature_selection(
                X, y, step3_features, non_feature_cols, must_keep_features
            )

            X_selected = X[[c for c in step4_selected if c in X.columns]]
            X_filled = X_selected.select_dtypes(include=[np.number]).fillna(
                X_selected.median(numeric_only=True)
            )

            step5_hyper = self._step_hyperparameter_optimization(X_filled, y)
            step6_bench = self._step_benchmarking(X_filled, y)
            self._check_r2_threshold(step6_bench)
            step6_bench = self._step_loocv_evaluation(X_filled, y, step6_bench)
            step7_ensemble = self._step_ensemble_creation(X_filled, y, step6_bench)
            step8_cv = self._step_cross_validation(step6_bench, X_filled, y, location_col, time_col)
            step9_confidence = self._step_prediction_confidence(step6_bench, step7_ensemble, X_filled, y)

            self.pipeline_state["status"] = "completed"
            self.pipeline_state["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            self.pipeline_state["status"] = "failed"
            self.pipeline_state["errors"].append(str(e))

        self._save_pipeline_state()
        logger.info("=" * 60)
        logger.info("Pipeline completed with status: %s", self.pipeline_state["status"])
        logger.info("=" * 60)
        return self.pipeline_state

    def _step_dataset_validation(self, X: pd.DataFrame, y: pd.Series) -> dict:
        logger.info("--- Step 1: Dataset Validation ---")
        detector = LeakageDetector(output_dir=self.output_dir)
        leakage = detector.check_all(X, y)
        result = {"leakage_found": detector.leakage_found_, "leakage_details": leakage}
        self.pipeline_state["steps"]["dataset_validation"] = result
        return result

    def _step_data_scoring(
        self, X: pd.DataFrame, non_feature_cols: list[str]
    ) -> Optional[pd.DataFrame]:
        logger.info("--- Step 2: Data Scoring ---")
        if not self.config.get("use_data_scoring"):
            logger.info("Data scoring disabled")
            return None
        scorer = DatasetScorer(output_dir=self.output_dir)
        scores = scorer.score_dataset(X)
        self.pipeline_state["steps"]["data_scoring"] = {
            "tier_counts": scores["tier"].value_counts().to_dict() if not scores.empty else {},
            "average_score": float(scores["total_score"].mean()) if not scores.empty else 0,
        }
        return scores

    def _step_feature_analysis(
        self, X: pd.DataFrame, y: pd.Series, non_feature_cols: list[str]
    ) -> dict:
        logger.info("--- Step 3: Feature Analysis ---")
        X_feat = X.drop(columns=[c for c in non_feature_cols if c in X.columns], errors="ignore")
        importance = FeatureImportanceAnalyzer(output_dir=self.output_dir)
        imp_results = importance.compute_all(X_feat, y)

        shap_result = None
        if self.config.get("use_shap"):
            try:
                if "random_forest" in imp_results:
                    from sklearn.ensemble import RandomForestRegressor
                    rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                    rf.fit(X_feat.select_dtypes(include=[np.number]).fillna(0), y)
                    shap_analyzer = ShapAnalyzer(output_dir=self.output_dir)
                    shap_result = shap_analyzer.analyze(rf, X_feat)
            except Exception as e:
                logger.warning("SHAP analysis skipped: %s", e)

        corr = CorrelationAnalyzer(output_dir=self.output_dir)
        corr_result = corr.analyze(X_feat)

        summary = {
            "importance": importance.get_feature_summary(),
            "correlation": {
                "high_corr_pairs": len(corr.high_corr_pairs_),
                "redundant_features_count": len(corr.get_redundant_features()),
            },
            "shap_available": shap_result is not None,
        }
        self.pipeline_state["steps"]["feature_analysis"] = summary
        return {
            "importance_analyzer": importance,
            "shap_analyzer": shap_analyzer if shap_result is not None else None,
            "correlation_analyzer": corr,
            "shap_df": shap_result,
        }

    def _step_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        analysis_results: dict,
        non_feature_cols: list[str],
        must_keep_features: list[str],
    ) -> list[str]:
        logger.info("--- Step 4: Feature Selection ---")
        importance_df = (
            analysis_results["importance_analyzer"].ranked_features_
            if analysis_results["importance_analyzer"] is not None
            else None
        )
        shap_df = analysis_results.get("shap_df")

        selector = FeatureSelector(
            output_dir=self.output_dir,
            importance_threshold=self.config["importance_threshold"],
            correlation_threshold=self.config["correlation_threshold"],
            variance_threshold=self.config["variance_threshold"],
            max_features=self.config["max_features"],
            min_features=self.config["min_features"],
            max_missing_pct=self.config.get("max_missing_pct", 1.0),
        )

        selected = selector.select(
            X, y, importance_df=importance_df, shap_df=shap_df,
            non_feature_cols=non_feature_cols, must_keep=must_keep_features,
        )

        summary = {
            "selected_count": len(selected),
            "total_before": len(X.columns),
            "reduction_pct": round((1 - len(selected) / len(X.columns)) * 100, 2),
        }
        self.pipeline_state["steps"]["feature_selection"] = summary
        logger.info("Selected %d features from %d", len(selected), len(X.columns))
        return selected

    def _step_hyperparameter_optimization(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict:
        logger.info("--- Step 5: Hyperparameter Optimization ---")
        if not self.config.get("use_hyperparameter_opt"):
            logger.info("Hyperparameter optimization disabled")
            return {}
        if len(X) < 20:
            logger.info("Too few samples (%d) for meaningful optimization", len(X))
            return {}

        optimizer = HyperparameterOptimizer(
            output_dir=Path("models/optimization"),
            n_trials=self.config["n_trials"],
            cv_folds=min(self.config["cv_folds"], len(X)),
            random_state=self.random_state,
        )
        results = optimizer.optimize_all(X, y)
        self.pipeline_state["steps"]["hyperparameter_optimization"] = {
            k: {"best_value": v.get("best_value"), "n_trials": v.get("n_trials")}
            for k, v in results.items()
        }
        return results

    def _step_benchmarking(self, X: pd.DataFrame, y: pd.Series) -> dict:
        logger.info("--- Step 6: Model Benchmarking ---")
        benchmark = ModelBenchmark(
            output_dir=self.output_dir, cv_folds=self.config["cv_folds"],
            random_state=self.random_state,
        )
        results = benchmark.benchmark_regression(X, y)
        best = benchmark.get_best_model_name()
        summary = {
            "best_model": best,
            "models_evaluated": len(results) if not results.empty else 0,
            "results": results.to_dict(orient="records") if not results.empty else [],
        }
        self.pipeline_state["steps"]["benchmarking"] = summary
        logger.info("Best model: %s", best)
        return {"results": results, "best_name": best, "benchmark": benchmark}

    def _check_r2_threshold(self, benchmark_results: dict):
        min_r2 = self.config.get("min_acceptable_r2", 0.0)
        if min_r2 <= 0:
            return
        df = benchmark_results.get("results")
        if df is None or df.empty:
            return
        valid = df[df["r2"].notna()]
        if valid.empty:
            return
        best_r2 = valid["r2"].max()
        best_model = valid.loc[valid["r2"].idxmax(), "model"]
        self.pipeline_state["steps"]["r2_threshold_check"] = {
            "best_r2": round(best_r2, 4),
            "best_model": best_model,
            "min_acceptable_r2": min_r2,
            "threshold_met": bool(best_r2 >= min_r2),
        }
        if best_r2 >= min_r2:
            logger.info("R² threshold check PASSED: %.4f >= %.2f (model: %s)", best_r2, min_r2, best_model)
        else:
            logger.warning("R² threshold check FAILED: %.4f < %.2f (model: %s)", best_r2, min_r2, best_model)
            logger.warning("Consider: (1) more data, (2) relaxed thresholds, (3) different target")

    def _step_loocv_evaluation(
        self, X: pd.DataFrame, y: pd.Series, benchmark_results: dict
    ) -> dict:
        if not self.config.get("use_loocv"):
            return benchmark_results
        logger.info("--- LOOCV Evaluation ---")
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        from sklearn.metrics import make_scorer, r2_score
        import numpy as np

        loocv_models = self.config.get("loocv_models", ["ridge", "random_forest"])
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        model_map = {
            "ridge": ("Ridge (LOOCV)", Ridge(alpha=1.0, random_state=self.random_state)),
            "lasso": ("Lasso (LOOCV)", Lasso(alpha=0.01, random_state=self.random_state)),
            "random_forest": ("Random Forest (LOOCV)", RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)),
            "gradient_boosting": ("Gradient Boosting (LOOCV)", GradientBoostingRegressor(n_estimators=200, random_state=self.random_state)),
        }

        loo = LeaveOneOut()
        loocv_results = []
        for key in loocv_models:
            if key not in model_map:
                continue
            name, model = model_map[key]
            try:
                r2_scores = cross_val_score(model, X, y, cv=loo, scoring="r2", n_jobs=-1)
                rmse_scores = cross_val_score(model, X, y, cv=loo, scoring="neg_root_mean_squared_error", n_jobs=-1)
                mean_r2 = float(np.mean(r2_scores))
                mean_rmse = float(np.mean(-rmse_scores))
                loocv_results.append({
                    "model": name,
                    "loocv_r2": round(mean_r2, 4),
                    "loocv_rmse": round(mean_rmse, 4),
                    "loocv_r2_std": round(float(np.std(r2_scores)), 4),
                })
                logger.info("LOOCV %s: R²=%.4f, RMSE=%.4f", name, mean_r2, mean_rmse)
            except Exception as e:
                logger.warning("LOOCV %s failed: %s", name, e)

        if loocv_results:
            best_loocv = max(loocv_results, key=lambda r: r["loocv_r2"])
            logger.info("Best LOOCV: %s with R²=%.4f", best_loocv["model"], best_loocv["loocv_r2"])
            self.pipeline_state["steps"]["loocv_evaluation"] = {
                "results": loocv_results,
                "best_loocv_model": best_loocv["model"],
                "best_loocv_r2": best_loocv["loocv_r2"],
            }

        return benchmark_results

    def _step_ensemble_creation(
        self, X: pd.DataFrame, y: pd.Series, benchmark_results: dict
    ) -> Optional[object]:
        logger.info("--- Step 7: Ensemble Creation ---")
        if not self.config.get("use_ensemble"):
            logger.info("Ensemble disabled")
            return None

        df_results = benchmark_results.get("results")
        if df_results is None or df_results.empty:
            logger.info("No benchmark results for ensemble")
            return None

        from src.ml.ensemble.weighted_average import WeightedAverageEnsemble

        models = {}
        for _, row in df_results.iterrows():
            name = row["model"]
            if name == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                models[name] = LinearRegression()
            elif name == "Ridge":
                from sklearn.linear_model import Ridge
                models[name] = Ridge(random_state=self.random_state)
            elif name == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                models[name] = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            elif name == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                models[name] = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            elif name == "XGBoost":
                import xgboost
                models[name] = xgboost.XGBRegressor(n_estimators=100, random_state=self.random_state)
            elif name == "LightGBM":
                import lightgbm
                models[name] = lightgbm.LGBMRegressor(n_estimators=100, random_state=self.random_state)

        if len(models) < 2:
            logger.info("Need at least 2 models for ensemble")
            return None

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        ensemble = WeightedAverageEnsemble(output_dir=Path("models"), random_state=self.random_state)
        ensemble.fit(models, X_filled, y)
        self.pipeline_state["steps"]["ensemble"] = {
            "models": list(models.keys()),
            "weights": ensemble.weights_,
            "training_r2": ensemble.training_score_,
        }
        return ensemble

    def _step_cross_validation(
        self,
        benchmark_results: dict,
        X: pd.DataFrame,
        y: pd.Series,
        location_col: Optional[str],
        time_col: Optional[str],
    ) -> dict:
        logger.info("--- Step 8: Cross Validation ---")
        best_name = benchmark_results.get("best_name")
        if not best_name:
            return {}

        model = self._get_best_model(benchmark_results)
        if model is None:
            logger.warning("Could not reconstruct best model for CV")
            return {}

        validator = PipelineCrossValidator(output_dir=self.output_dir, random_state=self.random_state)
        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        results = validator.validate_all(model, X_filled, y, location_col, time_col)
        self.pipeline_state["steps"]["cross_validation"] = results
        return results

    def _get_best_model(self, benchmark_results: dict) -> Optional[object]:
        best_name = benchmark_results.get("best_name")
        if best_name == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        elif best_name == "XGBoost":
            import xgboost
            return xgboost.XGBRegressor(n_estimators=100, random_state=self.random_state)
        elif best_name == "LightGBM":
            import lightgbm
            return lightgbm.LGBMRegressor(n_estimators=100, random_state=self.random_state)
        elif best_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        elif best_name in ("Ridge", "Linear Regression"):
            from sklearn.linear_model import Ridge
            return Ridge(random_state=self.random_state)
        return None

    def _step_prediction_confidence(
        self,
        benchmark_results: dict,
        ensemble: Optional[object],
        X: pd.DataFrame,
        y: pd.Series,
    ):
        logger.info("--- Step 9: Prediction Confidence ---")
        model = ensemble or self._get_best_model(benchmark_results)
        if model is None:
            return

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        try:
            model.fit(X_filled, y)
            y_pred = model.predict(X_filled)
            conf = PredictionConfidence()
            result = conf.estimate(model, X_filled, y_pred, y_actual=y.values)
            self.pipeline_state["steps"]["prediction_confidence"] = {
                "avg_confidence": float(result["confidence"].mean()),
                "risk_distribution": result["risk"].value_counts().to_dict(),
            }
            logger.info("Average prediction confidence: %.2f", result["confidence"].mean())
        except Exception as e:
            logger.warning("Prediction confidence estimation failed: %s", e)

    def _save_pipeline_state(self):
        state_path = self.output_dir / "auto_training_pipeline_state.json"
        state_path.write_text(
            json.dumps(self.pipeline_state, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved pipeline state to %s", state_path)
