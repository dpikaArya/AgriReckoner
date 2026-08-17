"""
Model Benchmark.
Automatically trains and evaluates multiple ML models.
"""

import time
import warnings

import numpy as np
import pandas as pd

from agri_ai_agent.ml.leakage import select_feature_columns
from evaluation.utils import (
    get_master_df,
    write_csv,
    write_report,
)

warnings.filterwarnings("ignore")


def _prepare_data():
    df = get_master_df()
    if df is None or len(df) < 5:
        return None, None, None

    target_cols = [
        c for c in ["Target_Yield", "Target_Fertilizer", "Target_Nitrogen"] if c in df.columns
    ]
    if not target_cols:
        return None, None, None

    target = target_cols[0]
    exclude = set(target_cols) - {target} | {"Paper_ID", "DOI", "Authors"}

    # The shared guard drops post-harvest outcomes and target restatements, which a
    # plain "all numeric columns" selection would feed straight into the model.
    feature_cols = select_feature_columns(df, target, base_exclude=exclude)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    # Only the target must be present. Requiring every feature to be non-null discarded
    # every row of this sparse literature-derived table; gaps are imputed inside folds.
    keep = y.notna()
    X, y = X[keep], y[keep]
    X = X.dropna(axis=1, how="all")

    if len(X) < 5 or len(X.columns) < 2:
        return None, None, None

    return X, y, target


def _train_model(model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t0) / len(y_test)

    residuals = y_test - y_pred
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (y_test + 1e-10))) * 100
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
        "adj_r2": round(float(adj_r2), 4),
        "train_time_sec": round(train_time, 4),
        "infer_time_per_sample_ms": round(infer_time * 1000, 4),
    }


def benchmark_models():
    X, y, target = _prepare_data()
    if X is None:
        report = "# Model Benchmark\nInsufficient data to train models.\n"
        write_report("Model_Benchmark_Summary.md", report.replace("Model\n", "Model\n"))
        path = write_report("Model_Benchmark.md", report)
        return [], str(path)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    model_map = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Support Vector Regression": SVR(kernel="rbf"),
    }

    results = []
    feature_importance_data = []

    for model_name in [
        "Multiple Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net",
        "Random Forest",
        "Extra Trees",
        "Decision Tree",
        "Support Vector Regression",
    ]:
        model = model_map.get(model_name)
        if model is None:
            continue

        try:
            metrics = _train_model(model, X_train, y_train, X_test, y_test)
            metrics["model"] = model_name
            results.append(metrics)

            if hasattr(model, "feature_importances_"):
                for i, col in enumerate(X.columns):
                    feature_importance_data.append(
                        {
                            "model": model_name,
                            "feature": col,
                            "importance": round(float(model.feature_importances_[i]), 6),
                        }
                    )
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                for i, col in enumerate(X.columns):
                    feature_importance_data.append(
                        {
                            "model": model_name,
                            "feature": col,
                            "coefficient": round(float(coef[i]), 6),
                        }
                    )
        except Exception as e:
            results.append(
                {
                    "model": model_name,
                    "error": str(e),
                }
            )

    for model_name in ["XGBoost", "LightGBM", "CatBoost"]:
        try:
            if model_name == "XGBoost":
                from xgboost import XGBRegressor

                model = XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0)
            elif model_name == "LightGBM":
                from lightgbm import LGBMRegressor

                model = LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1)
            elif model_name == "CatBoost":
                from catboost import CatBoostRegressor

                model = CatBoostRegressor(
                    n_estimators=50, max_depth=6, random_state=42, verbose=False
                )

            metrics = _train_model(model, X_train, y_train, X_test, y_test)
            metrics["model"] = model_name
            results.append(metrics)

            if hasattr(model, "feature_importances_"):
                for i, col in enumerate(X.columns):
                    feature_importance_data.append(
                        {
                            "model": model_name,
                            "feature": col,
                            "importance": round(float(model.feature_importances_[i]), 6),
                        }
                    )
        except ImportError:
            results.append(
                {
                    "model": model_name,
                    "error": f"{model_name} not installed",
                }
            )
        except Exception as e:
            results.append(
                {
                    "model": model_name,
                    "error": str(e),
                }
            )

    write_csv(
        "Model_Benchmark.csv",
        [{k: v for k, v in r.items() if k != "error"} for r in results if "error" not in r],
    )

    if feature_importance_data:
        write_csv("Feature_Importance.csv", feature_importance_data)

    results_with_metrics = [r for r in results if "rmse" in r]
    best_r2 = max(results_with_metrics, key=lambda x: x["r2"]) if results_with_metrics else None

    report = f"""# Model Benchmark Results
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Target variable: {target}
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Features: {X.shape[1]}

## Leaderboard
| Rank | Model | RMSE | MAE | MAPE (%) | R² | Adj. R² | Train Time (s) | Infer (ms/sample) |
|------|-------|------|-----|----------|-----|---------|----------------|-------------------|
"""
    sorted_results = sorted(results_with_metrics, key=lambda x: x["r2"], reverse=True)
    for i, r in enumerate(sorted_results):
        report += f"| {i + 1} | {r['model']} | {r['rmse']} | {r['mae']} | {r['mape']} | {r['r2']} | {r['adj_r2']} | {r['train_time_sec']} | {r['infer_time_per_sample_ms']} |\n"

    report += f"""
## Best Model
**{best_r2["model"]}** with R² = {best_r2["r2"]:.4f} (RMSE = {best_r2["rmse"]:.4f})

## Failed Models
"""
    for r in results:
        if "error" in r:
            report += f"- {r['model']}: {r['error']}\n"

    if not any("error" in r for r in results):
        report += "None - all models trained successfully\n"

    report += """
## Recommendations
1. Best performing model(s) should be further tuned with hyperparameter optimization
2. Cross-validate to ensure generalization
3. Feature importance analysis can guide feature selection
4. Consider ensemble of top models for improved performance
5. Test on held-out datasets to verify robustness
"""
    path = write_report("Leaderboard.md", report)
    return results, str(path)
