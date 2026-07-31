"""
Consolidated Training Agent.
Flow: Model Readiness → Train (XGBoost, RF, LR) → Evaluate → Export Metrics → Documentation
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
from agri_ai_agent.ml.evaluation import MIN_ROWS_FOR_METRIC, evaluate_model
from agri_ai_agent.ml.leakage import drop_suspected_leaks, select_feature_columns

TARGET_COLUMNS = [
    "Target_Yield",
    "Target_Fertilizer",
    "Target_Nitrogen",
    "Target_Phosphorus",
    "Target_Potassium",
]

EXCLUDE_COLS = (
    {
        "Paper_ID",
        "DOI",
        "Journal",
        "Year",
        "Authors",
        "Country",
        "Crop",
        "Scientific_Name",
        "Variety",
        "Season",
        "Location",
        "State",
        "Site",
        "Treatment",
        "Fertilizer_Name",
        "Organic_Fertilizer",
        "Biofertilizer",
        "Application_Method",
        "Application_Interval",
        "Feature_Available_Before_Prediction",
        "Table_Row",
        "Row_Index",
        "_source_page",
        "_reader",
        "_confidence",
        "Source_File",
    }
    | NON_FEATURE_COLS
    | {v for v in POST_HARVEST_VARIABLES}
)

MIN_SAMPLES_FOR_TRAINING = 50

MODEL_REQUIREMENTS = {
    "XGBoost": {
        "needs_scaling": False,
        "needs_encoding": True,
        "no_missing": True,
        "min_samples": 50,
    },
    "Random Forest": {
        "needs_scaling": False,
        "needs_encoding": True,
        "no_missing": False,
        "min_samples": 30,
    },
    "Linear Regression": {
        "needs_scaling": False,
        "needs_encoding": True,
        "no_missing": False,
        "min_samples": 10,
    },
}


class TrainingAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "TrainingAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        target_col = resolve_target_column(
            df, requested=kwargs.get("target"), extra_targets=TARGET_COLUMNS
        )
        if target_col is None:
            self.log.error("No usable target column found in dataset")
            return df

        self._check_readiness(df)

        X, y = self._prepare_data(df, target_col)
        if X is None or len(X) < MIN_ROWS_FOR_METRIC:
            n = 0 if X is None else len(X)
            self.log.warning("Insufficient data for honest training (%d samples)", n)
            return df

        models_dir = self.settings.OUTPUT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        feature_list = list(X.columns)
        with open(models_dir / "feature_list.json", "w") as f:
            json.dump(feature_list, f, indent=2)

        results, artifacts, all_importances = [], [], []
        fitted = {}
        for name, model in self._get_models().items():
            try:
                metrics = evaluate_model(model, X, y)
                pipeline = self._fit_pipeline(model, X, y)
                fitted[name] = pipeline
                model_path = (
                    models_dir / f"{name.lower().replace(' ', '_')}_{target_col.lower()}.joblib"
                )
                import joblib

                joblib.dump(pipeline, model_path)
                artifacts.append(str(model_path))
                all_importances.extend(
                    self._extract_feature_importance(
                        pipeline.named_steps["model"], name, feature_list
                    )
                )
                results.append(
                    {"model": name, "target": target_col, **metrics, "model_path": str(model_path)}
                )
                self.log.info("%s -> %s", name, self._format_metrics(metrics))
            except Exception as e:
                self.log.warning("Failed to train %s: %s", name, e)
                results.append({"model": name, "target": target_col, "error": str(e)})

        if all_importances:
            fi_df = pd.DataFrame(all_importances).sort_values(
                ["model", "importance"], ascending=[True, False]
            )
            self.save_artifact(fi_df, "feature_importance.csv")

        self._save_results(results, target_col)
        self._generate_documentation(df, results)

        best = self._best_result(results)
        if best is not None:
            import joblib

            yield_model_path = models_dir / "yield_model.pkl"
            joblib.dump(fitted[best["model"]], yield_model_path)
            artifacts.append(str(yield_model_path))
            self.log.info(
                "Saved yield_model.pkl (%s, %s)", best["model"], self._format_metrics(best)
            )

        if artifacts:
            self.contract.artifacts.extend(artifacts)

        self.dataframe = df
        self.log.info(
            "Training complete: %d/%d models scored",
            sum(1 for r in results if "r2" in r),
            len(results),
        )
        return df

    @staticmethod
    def _fit_pipeline(model, X, y):
        """Fit a median-impute + model pipeline on all rows (for persistence/importances)."""
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", model.__class__(**model.get_params())),
            ]
        )
        return pipeline.fit(X, y)

    @staticmethod
    def _best_result(results: list[dict]):
        """Pick the best cross-validated model, preferring robust (non-small-n) scores."""
        scored = [r for r in results if r.get("r2") is not None and "r2" in r]
        if not scored:
            return None
        return sorted(scored, key=lambda r: (r.get("robust", False), r["r2"]), reverse=True)[0]

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        if "r2" not in metrics:
            return metrics.get("note", "no score")
        std = f"±{metrics['r2_std']}" if metrics.get("r2_std") is not None else ""
        flag = "" if metrics.get("robust") else " [small-n]"
        return f"R2={metrics['r2']}{std} RMSE={metrics.get('rmse')} (n={metrics.get('n')}, {metrics.get('cv_scheme')}){flag}"

    def _get_models(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = None
        models = {}
        if XGBRegressor is not None:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        models["Linear Regression"] = LinearRegression()
        return models

    def _extract_feature_importance(self, model, model_name, feature_names):
        importances = []
        try:
            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
                for feat, imp in zip(feature_names, vals, strict=True):
                    importances.append(
                        {
                            "feature": feat,
                            "importance": float(imp),
                            "model": model_name,
                            "direction": 0,
                        }
                    )
            elif hasattr(model, "coef_"):
                coefs = model.coef_ if hasattr(model.coef_, "__iter__") else [model.coef_]
                for feat, coef in zip(feature_names, coefs, strict=True):
                    importances.append(
                        {
                            "feature": feat,
                            "importance": float(abs(coef)),
                            "model": model_name,
                            "direction": 1 if coef > 0 else -1,
                        }
                    )
        except Exception as e:
            self.log.warning("Could not extract importances from %s: %s", model_name, e)
        return importances

    def _check_readiness(self, df: pd.DataFrame):
        n_rows = len(df)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        missing_pct_by_col = (
            df[num_cols].isna().mean() * 100 if num_cols else pd.Series(dtype=float)
        )
        total_missing_pct = (
            df.isna().sum().sum() / (n_rows * len(df.columns)) * 100
            if n_rows * len(df.columns) > 0
            else 0
        )

        has_high_missing = any(missing_pct_by_col > 10) if len(missing_pct_by_col) else False
        known_codes = {
            "Crop_Code",
            "Season_Code",
            "Variety_Code",
            "Fertilizer_Code",
            "Soil_Texture_Code",
            "Country_Code",
        }
        has_unencoded_cat = any(col not in known_codes for col in cat_cols)

        issues = []
        if total_missing_pct > 5:
            issues.append(f"Total missing values: {total_missing_pct:.1f}%")
        if has_high_missing:
            high = [
                c for c in num_cols if c in missing_pct_by_col.index and missing_pct_by_col[c] > 10
            ]
            issues.append(f"{len(high)} columns with >10% missing values")
        if has_unencoded_cat:
            issues.append(f"Unencoded categorical variables ({len(cat_cols)} cols)")
        if n_rows < MIN_SAMPLES_FOR_TRAINING:
            issues.append(f"Dataset too small ({n_rows} rows; min {MIN_SAMPLES_FOR_TRAINING})")

        compatible = []
        incompatible = []
        for model_name, reqs in MODEL_REQUIREMENTS.items():
            model_issues = []
            if n_rows < reqs["min_samples"]:
                model_issues.append(f"insufficient samples ({n_rows} < {reqs['min_samples']})")
            if reqs.get("no_missing") and has_high_missing:
                model_issues.append("contains missing values")
            if reqs.get("needs_encoding") and has_unencoded_cat:
                model_issues.append("unencoded categorical variables")
            if model_issues:
                incompatible.append({"model": model_name, "issues": model_issues})
            else:
                compatible.append(model_name)

        lines = [
            "# Model Readiness Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Dataset: {n_rows} rows x {len(df.columns)} columns",
            "",
            f"## Compatible Models ({len(compatible)})",
        ]
        lines.extend(f"- {m}" for m in compatible)
        lines.append("")
        lines.append(f"## Incompatible Models ({len(incompatible)})")
        for m in incompatible:
            lines.append(f"- {m['model']}: {'; '.join(m['issues'])}")
        lines.append("")
        lines.append("## Data Quality Issues")
        if issues:
            lines.extend(f"- {i}" for i in issues)
        else:
            lines.append("- No critical issues detected")

        self.save_text_artifact("\n".join(lines), "Model_Readiness_Report.md")
        self.log.info(
            "Readiness check complete: %d compatible, %d incompatible",
            len(compatible),
            len(incompatible),
        )

    def _prepare_data(self, df: pd.DataFrame, target_col: str):
        """Build a leakage-safe feature matrix; keep rows with gaps (imputed inside CV)."""
        feature_cols = select_feature_columns(df, target_col, base_exclude=EXCLUDE_COLS)
        if len(feature_cols) < 2:
            return None, None

        y = pd.to_numeric(df[target_col], errors="coerce")
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        keep = y.notna()
        X, y = X[keep], y[keep]
        X = X.dropna(axis=1, how="all")

        X, dropped = drop_suspected_leaks(X, y)
        if dropped:
            self.log.warning(
                "Dropped %d near-perfect-correlation features (suspected leakage): %s",
                len(dropped),
                dropped,
            )

        if len(X) < MIN_ROWS_FOR_METRIC or X.shape[1] < 2:
            return None, None
        return X, y

    def _save_results(self, results: list[dict], target_col: str):
        results_df = pd.DataFrame(results)
        self.save_artifact(results_df, "metrics.csv")

        leaderboard = sorted([r for r in results if "r2" in r], key=lambda x: x["r2"], reverse=True)
        rows = ""
        for i, r in enumerate(leaderboard):
            std = f" ± {r['r2_std']}" if r.get("r2_std") is not None else ""
            robust = "yes" if r.get("robust") else "no (small-n)"
            rows += (
                f"<tr>"
                f"<td>{i + 1}</td><td>{r['model']}</td>"
                f"<td>{r['r2']}{std}</td><td>{r.get('rmse', 'N/A')}</td>"
                f"<td>{r.get('n', 'N/A')}</td><td>{r.get('cv_scheme', 'N/A')}</td>"
                f"<td>{robust}</td>"
                f"</tr>\n"
            )

        failed_rows = ""
        failed = [r for r in results if "error" in r]
        if failed:
            for r in failed:
                failed_rows += f"<tr><td>{r['model']}</td><td>{r['error']}</td></tr>\n"

        now = datetime.now().isoformat()
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; }}
h1 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #3498db; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
</style>
</head>
<body>
<h1>Training Report</h1>
<p>Generated: {now}</p>
<p>Target variable: <strong>{target_col}</strong></p>
<p style="background:#fff3cd;border:1px solid #ffe69c;padding:8px">
Scores are <strong>cross-validated</strong> (imputation fitted inside each fold; leakage-derived
features excluded). Rows flagged <em>not robust</em> are small-n (n&lt;30) and are advisory only.</p>
<h2>Leaderboard</h2>
<table>
<thead>
<tr><th>Rank</th><th>Model</th><th>Cross-validated R<sup>2</sup></th><th>RMSE</th><th>n</th><th>CV scheme</th><th>Robust</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
"""

        fi_path = self.settings.OUTPUT_DIR / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi_df = pd.read_csv(fi_path)
                fi_top = fi_df.groupby("model").head(10)
                fi_rows = ""
                for _, r in fi_top.iterrows():
                    bar_width = int(r["importance"] * 200) if r["importance"] > 0 else 1
                    direction = (
                        "+"
                        if r.get("direction", 0) > 0
                        else "-"
                        if r.get("direction", 0) < 0
                        else ""
                    )
                    fi_rows += (
                        f"<tr><td>{r['model']}</td><td>{r['feature']}</td>"
                        f"<td>{r['importance']:.4f}</td><td>{direction}</td>"
                        f"<td><div style='background:#3498db;width:{bar_width}px;height:12px'></div></td></tr>\n"
                    )
                html += f"""<h2>Feature Importance</h2>
<table>
<thead><tr><th>Model</th><th>Feature</th><th>Importance</th><th>Direction</th><th>Bar</th></tr></thead>
<tbody>
{fi_rows}
</tbody>
</table>
"""
            except Exception as e:
                self.log.debug("Failed to render feature importance HTML: %s", e)

        if failed_rows:
            html += f"""<h2>Failed Models</h2>
<table>
<thead><tr><th>Model</th><th>Error</th></tr></thead>
<tbody>
{failed_rows}
</tbody>
</table>
"""
        html += "</body>\n</html>"
        self.save_text_artifact(html, "training_report.html")
        self.log.info("Saved metrics.csv and training_report.html")

    def _generate_documentation(self, df: pd.DataFrame, results: list[dict]):
        rows = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = int(df[col].nunique())
            n_missing = int(df[col].isna().sum())
            missing_pct = round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0
            sample_vals = df[col].dropna().unique()[:5].tolist()
            rows.append(
                {
                    "Column": col,
                    "Data_Type": dtype,
                    "Unique_Values": n_unique,
                    "Missing_Count": n_missing,
                    "Missing_Pct": missing_pct,
                    "Sample_Values": str(sample_vals),
                }
            )
        fdf = pd.DataFrame(rows)
        self.save_artifact(fdf, "Feature_Dictionary.csv")

        trained = [r for r in results if "r2" in r]
        doc_lines = [
            "# Training Documentation",
            f"Generated: {datetime.now().isoformat()}",
            f"Dataset: {len(df)} rows x {len(df.columns)} columns",
            "",
            "## Models Trained",
        ]
        for r in trained:
            doc_lines.append(f"- {r['model']}: {self._format_metrics(r)}")
        best = self._best_result(results)
        if best is not None:
            doc_lines.append(f"\nBest model: {best['model']} ({self._format_metrics(best)})")
        self.save_text_artifact("\n".join(doc_lines), "Training_Documentation.md")
        self.log.info("Generated Feature_Dictionary.csv and Training_Documentation.md")

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "training_complete": True,
        }
