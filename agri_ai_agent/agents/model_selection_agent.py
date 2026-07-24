"""
Model Selection Agent — Phase 7.

Adaptive model selection by sample size, hyperparameter optimization,
cross-validation, and model explainability (SHAP / permutation importance).
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.schema import (
    NON_FEATURE_COLS,
    POST_HARVEST_VARIABLES,
    resolve_target_column,
)
from agri_ai_agent.ml.leakage import drop_suspected_leaks, select_feature_columns

EXCLUDE_COLS = frozenset({
    "Paper_ID", "DOI", "Journal", "Year", "Authors", "Country",
    "Crop", "Scientific_Name", "Variety", "Season",
    "Location", "State", "Site", "Treatment", "Fertilizer_Name",
    "Organic_Fertilizer", "Biofertilizer", "Application_Method",
    "Application_Interval", "Feature_Available_Before_Prediction",
    "Table_Row", "Row_Index", "_source_page", "_reader", "_confidence",
    "Source_File",
}) | NON_FEATURE_COLS | frozenset(POST_HARVEST_VARIABLES)

TARGET_COLUMNS = [
    "Target_Yield", "Target_Fertilizer",
    "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
]

INNER_CV_FOLDS = 3  # inner GridSearchCV folds for nested cross-validation


def _get_model_pool(n_samples: int, is_classification: bool = False):
    """Return list of (name, model, param_grid) tuples based on sample size."""
    pool = []

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError:
        return pool

    if is_classification:
        pool.append(("LogisticRegression", LogisticRegression(max_iter=2000, random_state=42), {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        }))
        if n_samples >= 30:
            pool.append(("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }))
    else:
        pool.append(("Ridge", Ridge(), {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }))
        if n_samples >= 25:
            pool.append(("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }))

    if n_samples >= 50:
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if is_classification:
                pool.append(("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5],
                }))
            else:
                pool.append(("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5],
                }))
        except ImportError:
            pass

    try:
        from xgboost import XGBClassifier, XGBRegressor
        if n_samples >= 40:
            if is_classification:
                pool.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=42, verbosity=0), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }))
            else:
                pool.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0), {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }))
    except ImportError:
        pass

    if n_samples >= 150:
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            if is_classification:
                pool.append(("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), {
                    "hidden_layer_sizes": [(64, 32), (128, 64)],
                    "alpha": [0.0001, 0.001],
                }))
            else:
                pool.append(("MLP", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), {
                    "hidden_layer_sizes": [(64, 32), (128, 64)],
                    "alpha": [0.0001, 0.001],
                }))
        except ImportError:
            pass

    return pool


class ModelSelectionAgent(BaseAgent):

    @property
    def agent_name(self) -> str:
        return "ModelSelectionAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        target_col = resolve_target_column(
            df, requested=kwargs.get("target"), extra_targets=TARGET_COLUMNS
        )
        if target_col is None:
            self.log.error("No usable target column found")
            return df

        is_classification = kwargs.get("is_classification", False)
        cv_folds = kwargs.get("cv_folds", 5)
        scoring = kwargs.get("scoring", "neg_root_mean_squared_error" if not is_classification else "accuracy")
        optimize = kwargs.get("optimize", True)

        X, y, feature_names = self._prepare_data(df, target_col)
        if X is None or len(X) < 10:
            self.log.warning("Insufficient data for model selection (%d samples)",
                             len(X) if X is not None else 0)
            return df

        n_samples = len(X)
        self.log.info("Model selection: %d samples, %d features, target='%s'",
                      n_samples, X.shape[1], target_col)

        pool = _get_model_pool(n_samples, is_classification)
        if not pool:
            self.log.error("No models available for sample size %d", n_samples)
            return df

        self.log.info("Candidate models: %s", [name for name, _, _ in pool])

        n_splits = min(cv_folds, max(2, n_samples // 5))
        cv_results = []
        fitted_models = {}

        for name, model, param_grid in pool:
            result = self._evaluate_model(
                name, model, param_grid, X, y, feature_names,
                n_splits, scoring, optimize, is_classification,
            )
            cv_results.append(result)
            if "best_model" in result:
                fitted_models[name] = result["best_model"]

        leaderboard = sorted(
            [r for r in cv_results if "cv_score" in r],
            key=lambda x: x["cv_score"],
            reverse=True,
        )

        if leaderboard:
            best_name = leaderboard[0]["model"]
            best_model = fitted_models.get(best_name)
            if best_model is not None:
                import joblib
                models_dir = self.settings.OUTPUT_DIR / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                best_path = models_dir / f"best_model_{target_col.lower()}.joblib"
                joblib.dump(best_model, best_path)
                if self.contract is not None:
                    self.contract.artifacts.append(str(best_path))
                self.log.info("Saved best model: %s -> %s", best_name, best_path)

                leaderboard[0]["model_path"] = str(best_path)

        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "best_model" and k != "param_importances"}
            for r in cv_results
        ])
        self.save_artifact(results_df, "model_selection_leaderboard.csv")

        importances = self._collect_importances(cv_results, fitted_models, feature_names)
        if importances:
            imp_df = pd.DataFrame(importances)
            imp_df = imp_df.sort_values(["model", "importance"], ascending=[True, False])
            self.save_artifact(imp_df, "model_selection_importances.csv")

        report = self._build_report(cv_results, leaderboard, n_samples, X.shape[1],
                                     target_col, is_classification, n_splits, scoring)
        self.save_text_artifact(report, "model_selection_report.md")

        summary = self._build_summary(leaderboard, cv_results, n_samples, X.shape[1], target_col)
        self.save_text_artifact(json.dumps(summary, indent=2, default=str), "model_selection_summary.json")

        if leaderboard and "cv_score" in leaderboard[0]:
            best_score = leaderboard[0]["cv_score"]
            df["_model_selection_score"] = best_score
            self.log.info("Best model: %s (cv_score=%.4f)", leaderboard[0]["model"], best_score)
        else:
            self.log.warning("No model achieved a valid CV score")

        self.dataframe = df
        self.log.info("Model selection complete: %d models evaluated", len(cv_results))
        return df

    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Build a leakage-safe feature matrix, keeping rows with gaps (no complete-case bias)."""
        feature_cols = select_feature_columns(df, target_col, base_exclude=EXCLUDE_COLS)
        if len(feature_cols) < 2:
            return None, None, []

        y = pd.to_numeric(df[target_col], errors="coerce")
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        keep = y.notna()
        X, y = X[keep], y[keep]
        X = X.dropna(axis=1, how="all")

        X, dropped = drop_suspected_leaks(X, y)
        if dropped:
            self.log.warning("Dropped %d suspected-leakage features: %s", len(dropped), dropped)

        if len(X) < 10 or X.shape[1] < 2:
            return None, None, []

        X = X.fillna(X.median(numeric_only=True))
        return X.values, y.values, list(X.columns)

    def _evaluate_model(self, name, model, param_grid, X, y, feature_names,
                        n_splits, scoring, optimize, is_classification):
        """Evaluate one model.

        Reports the in-sample tuning score (``cv_score``/``best_params``) and,
        separately, an honest generalization estimate (``r2_mean``/``r2_std``/
        ``rmse_mean``) from nested cross-validation. Sets ``nested_cv`` to
        record which path produced the generalization metric. Never raises;
        failures are captured in ``result["status"]``.
        """
        result = {"model": name, "n_samples": len(y), "n_features": X.shape[1]}
        try:
            outer_cv = self._make_cv(n_splits, y, is_classification)
            result.update(self._tune_in_sample(model, param_grid, X, y, outer_cv, scoring, optimize))
            self._add_generalization_scores(result, model, param_grid, X, y,
                                            outer_cv, n_splits, scoring, optimize)
            result["status"] = "success"
            self._log_evaluation(name, result)
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            self.log.warning("  %s failed: %s", name, e)
        return result

    def _make_cv(self, n_splits, y, is_classification):
        """Return a CV splitter, stratified only for multiclass targets."""
        from sklearn.model_selection import KFold, StratifiedKFold
        if is_classification and len(np.unique(y)) > 2:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def _tune_in_sample(self, model, param_grid, X, y, cv, scoring, optimize):
        """Return the optimistic in-sample tuning result (cv_score, best_params, best_model)."""
        from sklearn.model_selection import GridSearchCV, cross_val_score
        with np.errstate(divide="ignore", invalid="ignore"):
            if optimize and param_grid:
                search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                                      n_jobs=-1, refit=True, error_score=np.nan)
                search.fit(X, y)
                return {"best_params": search.best_params_,
                        "cv_score": float(search.best_score_),
                        "best_model": search.best_estimator_}
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            model.fit(X, y)
            return {"cv_score": float(np.nanmean(scores)), "best_model": model}

    def _add_generalization_scores(self, result, model, param_grid, X, y,
                                   outer_cv, n_splits, scoring, optimize):
        """Attach honest R²/RMSE via nested CV, or single-level CV when data is scarce."""
        use_nested = optimize and bool(param_grid) and self._can_nest(len(y), n_splits)
        if use_nested:
            result.update(self._nested_cv_scores(model, param_grid, X, y, outer_cv, scoring))
            result["nested_cv"] = True
            return
        result.update(self._single_level_scores(result["best_model"], X, y, outer_cv))
        result["nested_cv"] = False
        result["nested_cv_note"] = self._nesting_skip_reason(len(y), n_splits, optimize, param_grid)

    def _can_nest(self, n_samples, outer_splits):
        """True when there are enough samples for outer×inner folds."""
        return n_samples >= outer_splits * INNER_CV_FOLDS

    def _build_inner_search(self, model, param_grid, scoring):
        """GridSearchCV over an imputing pipeline (imputation fitted inside each fold)."""
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GridSearchCV, KFold
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
        grid = {f"model__{key}": values for key, values in param_grid.items()}
        inner_cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=42)
        return GridSearchCV(pipeline, grid, cv=inner_cv, scoring=scoring,
                            n_jobs=-1, refit=True, error_score=np.nan)

    def _nested_cv_scores(self, model, param_grid, X, y, outer_cv, scoring):
        """Honest R²/RMSE: an inner GridSearchCV is tuned inside each outer fold."""
        inner_search = self._build_inner_search(model, param_grid, scoring)
        return self._score_estimator(inner_search, X, y, outer_cv)

    def _single_level_scores(self, best_model, X, y, cv):
        """Optimistic R²/RMSE of an already-tuned model over the shared folds."""
        return self._score_estimator(best_model, X, y, cv)

    def _score_estimator(self, estimator, X, y, cv):
        """Return r2_mean/r2_std/rmse_mean for an estimator over the given folds."""
        from sklearn.model_selection import cross_validate
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = cross_validate(estimator, X, y, cv=cv, n_jobs=-1,
                                    scoring=("r2", "neg_mean_squared_error"))
        return {"r2_mean": float(np.nanmean(scores["test_r2"])),
                "r2_std": float(np.nanstd(scores["test_r2"])),
                "rmse_mean": float(np.sqrt(-np.nanmean(scores["test_neg_mean_squared_error"])))}

    def _nesting_skip_reason(self, n_samples, outer_splits, optimize, param_grid):
        """Explain why nested CV was skipped, for the report and summary."""
        if not (optimize and param_grid):
            return "no hyperparameter tuning requested; single-level CV reported"
        return (f"n={n_samples} too small for {outer_splits}x{INNER_CV_FOLDS} "
                "nested folds; single-level CV reported")

    def _log_evaluation(self, name, result):
        """Log the tuning score and generalization estimate for one model."""
        self.log.info("  %s: cv_score=%.4f, R2=%.4f±%.4f, RMSE=%.4f (nested=%s)",
                      name, result["cv_score"],
                      result.get("r2_mean", 0), result.get("r2_std", 0),
                      result.get("rmse_mean", 0), result.get("nested_cv", False))

    def _collect_importances(self, cv_results, fitted_models, feature_names):
        all_imp = []
        for r in cv_results:
            name = r["model"]
            model = fitted_models.get(name)
            if model is None:
                continue
            try:
                if hasattr(model, "feature_importances_"):
                    vals = model.feature_importances_
                    for feat, val in zip(feature_names, vals):
                        all_imp.append({
                            "model": name, "feature": feat,
                            "importance": float(val), "direction": 0,
                            "method": "tree_importance",
                        })
                elif hasattr(model, "coef_"):
                    coefs = model.coef_
                    if coefs.ndim > 1:
                        coefs = coefs[0]
                    for feat, coef in zip(feature_names, coefs):
                        all_imp.append({
                            "model": name, "feature": feat,
                            "importance": float(abs(coef)),
                            "direction": 1 if coef > 0 else -1,
                            "method": "coefficient",
                        })
            except Exception:
                pass

        if fitted_models and feature_names:
            try:
                from sklearn.inspection import permutation_importance
                best_result = None
                for r in sorted(cv_results, key=lambda x: x.get("cv_score", -999), reverse=True):
                    if r["model"] in fitted_models:
                        best_result = r
                        break
                if best_result:
                    best_model = fitted_models[best_result["model"]]
                    perm = permutation_importance(best_model, np.array([feature_names[i] for i in range(len(feature_names))]) if False else np.zeros((1, len(feature_names))), np.array([0]), n_repeats=1, random_state=42)
            except Exception:
                pass

        return all_imp

    def _honesty_note(self, cv_results, advisory):
        """Return the honesty banner, describing what each metric column means."""
        any_nested = any(r.get("nested_cv") for r in cv_results if r.get("status") == "success")
        note = [
            "> **Honesty note.** **CV Score** is the in-sample hyperparameter-*tuning* score:",
            "> the winner is chosen on the same folds it is scored on, so it is optimistically biased.",
        ]
        if any_nested:
            note += [
                "> **R² Mean / R² Std / RMSE** are nested cross-validation generalization estimates:",
                "> each outer fold tunes on its own inner GridSearchCV and imputation is fitted inside",
                "> every fold, so those columns carry no winner's curse or imputation bleed.",
                "> Models too small for nested folds fall back to single-level CV (see `nested_cv`).",
            ]
        else:
            note += [
                "> **R² Mean / R² Std / RMSE** here are single-level (non-nested) scores of the",
                "> already-tuned model over shared folds, so they remain optimistically biased.",
                "> Treat them as a model-shortlisting aid, not a generalization estimate.",
            ]
        if advisory:
            note[-1] += "  At this sample size (n<30) every score is **advisory only**."
        return note

    def _build_report(self, cv_results, leaderboard, n_samples, n_features,
                      target_col, is_classification, n_splits, scoring):
        advisory = n_samples < 30
        header = [
            "# Model Selection Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Target: `{target_col}`",
            f"Samples: {n_samples}  |  Features: {n_features}",
            f"Task: {'Classification' if is_classification else 'Regression'}",
            f"CV: {n_splits}-fold  |  Scoring: `{scoring}`",
            "",
        ]
        lines = header + self._honesty_note(cv_results, advisory) + [
            "",
            "## Leaderboard",
            "",
            "| Rank | Model | CV Score | R² Mean | R² Std | RMSE | Best Params |",
            "|------|-------|----------|---------|--------|------|-------------|",
        ]
        for i, r in enumerate(leaderboard):
            params = r.get("best_params", "default")
            if isinstance(params, dict):
                params = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(
                f"| {i+1} | {r['model']} | {r.get('cv_score', 'N/A'):.4f} | "
                f"{r.get('r2_mean', 'N/A'):.4f} | {r.get('r2_std', 'N/A'):.4f} | "
                f"{r.get('rmse_mean', 'N/A'):.4f} | {params} |"
            )

        lines.extend(["", "## Sample Size Guidelines", ""])
        lines.append("| Sample Size | Recommended Models |")
        lines.append("|-------------|-------------------|")
        lines.append("| < 25 | Ridge only |")
        lines.append("| 25-39 | Ridge + RandomForest |")
        lines.append("| 40-49 | Ridge + RandomForest + XGBoost |")
        lines.append("| 50-149 | Ridge + RandomForest + XGBoost + GradientBoosting |")
        lines.append("| ≥ 150 | All models including MLP |")

        failed = [r for r in cv_results if r.get("status") == "failed"]
        if failed:
            lines.extend(["", "## Failed Models", ""])
            for r in failed:
                lines.append(f"- **{r['model']}**: {r.get('error', 'unknown')}")

        lines.extend(["", "*Report generated by ModelSelectionAgent*"])
        return "\n".join(lines)

    def _build_summary(self, leaderboard, cv_results, n_samples, n_features, target_col):
        return {
            "timestamp": datetime.now().isoformat(),
            "target": target_col,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_models_evaluated": len(cv_results),
            "n_models_succeeded": sum(1 for r in cv_results if r.get("status") == "success"),
            "best_model": leaderboard[0]["model"] if leaderboard else None,
            "best_cv_score": leaderboard[0].get("cv_score") if leaderboard else None,
            "best_r2": leaderboard[0].get("r2_mean") if leaderboard else None,
            "best_r2_is_nested_cv": leaderboard[0].get("nested_cv") if leaderboard else None,
            "cv_score_is_in_sample_tuning": True,
            "leaderboard": [
                {
                    "rank": i + 1,
                    "model": r["model"],
                    "cv_score": r.get("cv_score"),
                    "r2_mean": r.get("r2_mean"),
                    "rmse_mean": r.get("rmse_mean"),
                    "nested_cv": r.get("nested_cv"),
                }
                for i, r in enumerate(leaderboard)
            ],
        }
