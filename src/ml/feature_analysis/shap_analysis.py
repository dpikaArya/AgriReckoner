import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("ShapAnalyzer")


class ShapAnalyzer:
    def __init__(self, output_dir: Path | None = None, random_state: int = 42):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.shap_values_ = None
        self.base_value_ = None
        self.feature_names_: list[str] = []
        self.summary_: pd.DataFrame | None = None

    def analyze(self, model, X: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed; skipping SHAP analysis")
            return pd.DataFrame()

        if X.empty:
            logger.warning("Empty input for SHAP analysis")
            return pd.DataFrame()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric columns for SHAP analysis")
            return pd.DataFrame()

        X_num = X[numeric_cols].fillna(X[numeric_cols].median())
        self.feature_names_ = X_num.columns.tolist()

        if len(X_num) > sample_size:
            X_sample = X_num.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X_num

        try:
            explainer = shap.Explainer(model, X_sample, check_additivity=False)
            shap_values = explainer(X_sample)
            self.shap_values_ = shap_values
            self.base_value_ = shap_values.base_values
        except Exception as e:
            logger.warning("SHAP explainer failed: %s", e)
            return pd.DataFrame()

        self.summary_ = self._build_summary()
        self._generate_report()
        return self.summary_

    def _build_summary(self) -> pd.DataFrame:
        if self.shap_values_ is None:
            return pd.DataFrame()

        values = self.shap_values_.values
        if values.ndim == 3:
            values = values[:, :, 0]

        mean_abs = np.abs(values).mean(axis=0)
        mean_shap = values.mean(axis=0)
        std_shap = values.std(axis=0)
        pos_effect = (values > 0).mean(axis=0)
        neg_effect = (values < 0).mean(axis=0)

        df = (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "mean_abs_shap": mean_abs,
                    "mean_shap": mean_shap,
                    "std_shap": std_shap,
                    "positive_effect_pct": pos_effect,
                    "negative_effect_pct": neg_effect,
                    "rank": np.argsort(-mean_abs) + 1,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        df["impact"] = pd.cut(
            df["mean_abs_shap"],
            bins=[
                -np.inf,
                df["mean_abs_shap"].quantile(0.25),
                df["mean_abs_shap"].quantile(0.75),
                np.inf,
            ],
            labels=["low", "medium", "high"],
        )
        return df

    def _generate_report(self):
        if self.summary_ is None or self.summary_.empty:
            return

        path = self.output_dir / "shap_summary.html"
        html_rows = []
        for _, row in self.summary_.head(50).iterrows():
            html_rows.append(f"""<tr>
            <td>{int(row["rank"])}</td>
            <td>{row["feature"]}</td>
            <td>{row["mean_abs_shap"]:.6f}</td>
            <td>{row["mean_shap"]:.6f}</td>
            <td>{row["positive_effect_pct"]:.1%}</td>
            <td>{row["negative_effect_pct"]:.1%}</td>
            <td>{row["impact"]}</td>
        </tr>""")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SHAP Feature Importance</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #4CAF50; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.high {{ background-color: #c8e6c9; }}
.medium {{ background-color: #fff9c4; }}
.low {{ background-color: #ffcdd2; }}
</style></head><body>
<h1>SHAP Feature Importance Analysis</h1>
<p>Top 50 features by mean absolute SHAP value</p>
<table><thead><tr>
<th>Rank</th><th>Feature</th><th>Mean |SHAP|</th>
<th>Mean SHAP</th><th>Positive Effect</th><th>Negative Effect</th><th>Impact</th>
</tr></thead><tbody>
{"".join(html_rows)}
</tbody></table></body></html>"""
        path.write_text(html, encoding="utf-8")
        logger.info("Saved SHAP report to %s", path)

        json_path = self.output_dir / "shap_summary.json"
        data = self.summary_.to_dict(orient="records") if self.summary_ is not None else []
        json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def get_top_shap_features(self, n: int = 50) -> list[str]:
        if self.summary_ is None or self.summary_.empty:
            return []
        return self.summary_.head(n)["feature"].tolist()
