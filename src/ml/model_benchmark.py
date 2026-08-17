import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("ModelBenchmark")


class ModelBenchmark:
    def __init__(self, output_dir: Path | None = None, cv_folds: int = 5, random_state: int = 42):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results_: list[dict] = []

    def benchmark_regression(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if X.empty or y.empty:
            logger.warning("Empty data for regression benchmark")
            return pd.DataFrame()

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        models = self._get_regression_models()
        results = []

        for name, model in models:
            try:
                result = self._evaluate_regression(name, model, X_filled, y)
                results.append(result)
            except Exception as e:
                logger.warning("Benchmark for %s failed: %s", name, e)
                results.append(
                    {
                        "model": name,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "mape": None,
                        "cv_rmse_mean": None,
                        "cv_rmse_std": None,
                        "error": str(e),
                    }
                )

        self.results_ = results
        df = pd.DataFrame(results)
        self._save_results(df, "regression")
        return df

    def benchmark_classification(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if X.empty or y.empty:
            logger.warning("Empty data for classification benchmark")
            return pd.DataFrame()

        X_filled = X.select_dtypes(include=[np.number]).fillna(X.median(numeric_only=True))
        models = self._get_classification_models()
        results = []

        for name, model in models:
            try:
                result = self._evaluate_classification(name, model, X_filled, y)
                results.append(result)
            except Exception as e:
                logger.warning("Benchmark for %s failed: %s", name, e)
                results.append(
                    {
                        "model": name,
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "roc_auc": None,
                        "error": str(e),
                    }
                )

        self.results_ = results
        df = pd.DataFrame(results)
        self._save_results(df, "classification")
        return df

    def _get_regression_models(self) -> list[tuple[str, object]]:
        models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0, random_state=self.random_state)),
            ("Lasso", Lasso(alpha=0.01, random_state=self.random_state)),
            (
                "Random Forest",
                RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            ),
            (
                "Gradient Boosting",
                GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            ),
        ]
        try:
            import xgboost

            models.append(
                (
                    "XGBoost",
                    xgboost.XGBRegressor(
                        n_estimators=100, random_state=self.random_state, verbosity=0, n_jobs=-1
                    ),
                )
            )
        except ImportError:
            pass
        try:
            import lightgbm

            models.append(
                (
                    "LightGBM",
                    lightgbm.LGBMRegressor(
                        n_estimators=100, random_state=self.random_state, verbosity=-1, n_jobs=-1
                    ),
                )
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor

            models.append(
                (
                    "CatBoost",
                    CatBoostRegressor(iterations=100, random_seed=self.random_state, verbose=False),
                )
            )
        except ImportError:
            pass
        return models

    def _get_classification_models(self) -> list[tuple[str, object]]:
        models = [
            (
                "Random Forest",
                RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
            ),
        ]
        try:
            import xgboost

            models.append(
                (
                    "XGBoost",
                    xgboost.XGBClassifier(
                        n_estimators=100, random_state=self.random_state, verbosity=0, n_jobs=-1
                    ),
                )
            )
        except ImportError:
            pass
        try:
            import lightgbm

            models.append(
                (
                    "LightGBM",
                    lightgbm.LGBMClassifier(
                        n_estimators=100, random_state=self.random_state, verbosity=-1, n_jobs=-1
                    ),
                )
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostClassifier

            models.append(
                (
                    "CatBoost",
                    CatBoostClassifier(
                        iterations=100, random_seed=self.random_state, verbose=False
                    ),
                )
            )
        except ImportError:
            pass
        return models

    def _evaluate_regression(self, name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        y_test_safe = y_test.replace(0, 1e-6)
        mape = float((np.abs((y_test - y_pred) / y_test_safe).mean()) * 100)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=min(self.cv_folds, len(X)),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        return {
            "model": name,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "cv_rmse_mean": round(float(-cv_scores.mean()), 4),
            "cv_rmse_std": round(float(cv_scores.std()), 4),
        }

    def _evaluate_classification(self, name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(
                precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
            ),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }

        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] >= 2:
                    result["roc_auc"] = round(
                        roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4
                    )
                else:
                    result["roc_auc"] = None
            else:
                result["roc_auc"] = None
        except Exception:
            result["roc_auc"] = None

        return result

    def _save_results(self, df: pd.DataFrame, benchmark_type: str):
        csv_path = self.output_dir / f"{benchmark_type}_benchmark.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %s benchmark to %s", benchmark_type, csv_path)

        best_idx = None
        if benchmark_type == "regression" and "r2" in df.columns:
            valid = df[df["r2"].notna()]
            if not valid.empty:
                best_idx = valid["r2"].idxmax()
        elif benchmark_type == "classification" and "f1" in df.columns:
            valid = df[df["f1"].notna()]
            if not valid.empty:
                best_idx = valid["f1"].idxmax()

        best_model = df.loc[best_idx].to_dict() if best_idx is not None else None

        self._generate_html(df, benchmark_type, best_model)

        all_results = {
            "type": benchmark_type,
            "best_model": best_model,
            "models": df.to_dict(orient="records"),
        }
        with open(self.output_dir / "model_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    def _generate_html(self, df: pd.DataFrame, bench_type: str, best: dict | None):
        metric_cols = [c for c in df.columns if c not in ["model", "error"]]
        html_rows = []
        for _, row in df.iterrows():
            cells = f"<td>{row['model']}</td>"
            for col in metric_cols:
                val = row.get(col, "")
                cells += f"<td>{val}</td>"
            is_best = best is not None and row["model"] == best.get("model")
            cls = "best" if is_best else ""
            html_rows.append(f"<tr class='{cls}'>{cells}</tr>")

        metric_headers = "".join(f"<th>{c}</th>" for c in metric_cols)
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Model Comparison - {bench_type}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #2196F3; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.best {{ background-color: #c8e6c9 !important; font-weight: bold; }}
</style></head><body>
<h1>Model Comparison - {bench_type.title()}</h1>
<table><thead><tr><th>Model</th>{metric_headers}</tr></thead>
<tbody>{"".join(html_rows)}</tbody></table>
</body></html>"""
        path = self.output_dir / f"model_comparison_{bench_type}.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved %s comparison HTML to %s", bench_type, path)

    def get_best_model_name(self) -> str | None:
        if not self.results_:
            return None
        valid = [r for r in self.results_ if r.get("r2") is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r["r2"])["model"]
