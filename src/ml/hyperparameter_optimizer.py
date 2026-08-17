import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("HyperparameterOptimizer")

REGRESSION_METRICS = ["rmse", "mae", "r2", "mape"]


class HyperparameterOptimizer:
    def __init__(
        self,
        output_dir: Path | None = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout_seconds: int | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("models/optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout_seconds = timeout_seconds
        self.best_params_: dict[str, Any] = {}
        self.optimization_history_: list[dict[str, Any]] = []

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        try:
            import optuna  # noqa: F401
        except ImportError:
            logger.warning("optuna not installed; using default XGBoost params")
            return self._default_xgboost_params()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "random_state": self.random_state,
                "verbosity": 0,
            }
            return self._cv_score("xgb", X, y, params)

        return self._run_study("xgboost", objective)

    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        try:
            import optuna  # noqa: F401
        except ImportError:
            logger.warning("optuna not installed; using default LightGBM params")
            return self._default_lightgbm_params()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                "random_state": self.random_state,
                "verbosity": -1,
            }
            return self._cv_score("lgb", X, y, params)

        return self._run_study("lightgbm", objective)

    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        try:
            import optuna  # noqa: F401
        except ImportError:
            logger.warning("optuna not installed; using default RF params")
            return self._default_rf_params()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            return self._cv_score("rf", X, y, params)

        return self._run_study("random_forest", objective)

    def optimize_catboost(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        try:
            import optuna  # noqa: F401
        except ImportError:
            logger.warning("optuna not installed; using default CatBoost params")
            return self._default_catboost_params()

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 50, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
                "random_seed": self.random_state,
                "verbose": False,
            }
            return self._cv_score("catboost", X, y, params)

        return self._run_study("catboost", objective)

    def optimize_all(self, X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, Any]]:
        return {
            "xgboost": self.optimize_xgboost(X, y),
            "lightgbm": self.optimize_lightgbm(X, y),
            "random_forest": self.optimize_random_forest(X, y),
        }

    def _cv_score(self, model_type: str, X: pd.DataFrame, y: pd.Series, params: dict) -> float:
        if len(X) < self.cv_folds:
            return -1e6

        try:
            if model_type == "xgb":
                import xgboost as xgb

                model = xgb.XGBRegressor(**params)
            elif model_type == "lgb":
                import lightgbm as lgb

                model = lgb.LGBMRegressor(**params)
            elif model_type == "rf":
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(**params)
            elif model_type == "catboost":
                from catboost import CatBoostRegressor

                model = CatBoostRegressor(**params)
            else:
                return -1e6

            scores = cross_val_score(
                model,
                X,
                y,
                cv=min(self.cv_folds, len(X)),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            return float(scores.mean())
        except Exception as e:
            logger.warning("CV score failed for %s: %s", model_type, e)
            return -1e6

    def _run_study(self, name: str, objective) -> dict[str, Any]:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            study_name=name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False,
        )

        result = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
        self.best_params_[name] = study.best_params
        self._save_study(name, study)
        return result

    def _save_study(self, name: str, study):
        params_path = self.output_dir / f"{name}_best_params.json"
        params_path.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")

        history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ]
        hist_path = self.output_dir / f"{name}_optimization_history.json"
        hist_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

        self.optimization_history_.extend(history)

        all_params = {name: study.best_params}
        combined_path = self.output_dir / "best_parameters.json"
        if combined_path.exists():
            existing = json.loads(combined_path.read_text(encoding="utf-8"))
            existing.update(all_params)
            all_params = existing
        combined_path.write_text(json.dumps(all_params, indent=2), encoding="utf-8")

        logger.info("Saved best params for %s (score: %.4f)", name, study.best_value)

    def _default_xgboost_params(self) -> dict:
        return {
            "best_params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
            },
            "best_value": None,
            "n_trials": 0,
        }

    def _default_lightgbm_params(self) -> dict:
        return {
            "best_params": {
                "n_estimators": 100,
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
            },
            "best_value": None,
            "n_trials": 0,
        }

    def _default_rf_params(self) -> dict:
        return {
            "best_params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "best_value": None,
            "n_trials": 0,
        }

    def _default_catboost_params(self) -> dict:
        return {
            "best_params": {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "l2_leaf_reg": 3.0,
            },
            "best_value": None,
            "n_trials": 0,
        }
