"""
EvaluationDashboard — Phase 15.

Computes and displays 8 key pipeline metrics:
1. Extraction Accuracy
2. Schema Mapping Accuracy
3. Missing Data Rate
4. Validation Pass Rate
5. Feature Engineering Coverage
6. Recommendation Confidence
7. Ready Reckoner Completeness
8. Pipeline Success Rate
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


METRIC_NAMES = [
    "extraction_accuracy",
    "schema_mapping_accuracy",
    "missing_data_rate",
    "validation_pass_rate",
    "feature_coverage",
    "recommendation_confidence",
    "reckoner_completeness",
    "pipeline_success_rate",
]


class EvaluationDashboard:
    def __init__(self, output_dir: str | Path = "outputs/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, float] = {}
        self._details: dict[str, dict] = {}

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    @property
    def details(self) -> dict[str, dict]:
        return dict(self._details)

    def compute_extraction_accuracy(self, extracted_df: pd.DataFrame,
                                     reference_df: Optional[pd.DataFrame] = None) -> float:
        if reference_df is not None and len(reference_df) > 0:
            common_cols = [c for c in extracted_df.columns if c in reference_df.columns]
            if common_cols:
                filled = 0
                total = 0
                for col in common_cols:
                    ext_vals = extracted_df[col].dropna()
                    ref_vals = reference_df[col].dropna()
                    total += len(ref_vals)
                    filled += len(ext_vals[ext_vals.isin(ref_vals)])
                accuracy = filled / total if total > 0 else 0.0
            else:
                accuracy = self._compute_filling_rate(extracted_df)
        else:
            accuracy = self._compute_filling_rate(extracted_df)

        self._metrics["extraction_accuracy"] = round(accuracy, 4)
        self._details["extraction_accuracy"] = {
            "description": "Fraction of target schema columns with valid extracted values",
            "value": round(accuracy, 4),
            "rows": len(extracted_df),
            "columns": len(extracted_df.columns),
        }
        return accuracy

    def compute_schema_mapping_accuracy(self, df: pd.DataFrame,
                                         schema_columns: list[str]) -> float:
        mapped = [c for c in schema_columns if c in df.columns]
        accuracy = len(mapped) / len(schema_columns) if schema_columns else 0.0
        unmapped = [c for c in schema_columns if c not in df.columns]
        self._metrics["schema_mapping_accuracy"] = round(accuracy, 4)
        self._details["schema_mapping_accuracy"] = {
            "description": "Fraction of schema columns successfully mapped",
            "value": round(accuracy, 4),
            "mapped": len(mapped),
            "unmapped_count": len(unmapped),
            "unmapped": unmapped[:10],
        }
        return accuracy

    def compute_missing_data_rate(self, df: pd.DataFrame) -> float:
        if df.empty:
            rate = 1.0
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                rate = 0.0
            else:
                rate = float(numeric_df.isna().mean().mean())
        self._metrics["missing_data_rate"] = round(rate, 4)
        self._details["missing_data_rate"] = {
            "description": "Fraction of numeric cells that are missing",
            "value": round(rate, 4),
            "total_cells": int(df.select_dtypes(include=[np.number]).shape[0] * df.select_dtypes(include=[np.number]).shape[1]),
            "missing_cells": int(df.select_dtypes(include=[np.number]).isna().sum().sum()),
        }
        return rate

    def compute_validation_pass_rate(self, df: pd.DataFrame,
                                      validation_issues: Optional[list] = None) -> float:
        if validation_issues is None:
            validation_issues = []
        total_rows = len(df)
        issue_rows = len(set(
            issue.get("row", 0) for issue in validation_issues if isinstance(issue, dict)
        ))
        pass_rate = 1.0 - (issue_rows / total_rows) if total_rows > 0 else 1.0
        self._metrics["validation_pass_rate"] = round(max(0.0, pass_rate), 4)
        self._details["validation_pass_rate"] = {
            "description": "Fraction of rows passing all validation rules",
            "value": round(max(0.0, pass_rate), 4),
            "total_rows": total_rows,
            "issue_rows": issue_rows,
            "issue_count": len(validation_issues),
        }
        return pass_rate

    def compute_feature_coverage(self, original_df: pd.DataFrame,
                                  engineered_df: pd.DataFrame) -> float:
        original_cols = set(original_df.columns)
        engineered_cols = set(engineered_df.columns)
        new_features = engineered_cols - original_cols
        coverage = len(new_features) / max(len(original_cols), 1)
        self._metrics["feature_coverage"] = round(min(coverage, 1.0), 4)
        self._details["feature_coverage"] = {
            "description": "Ratio of engineered features to original columns",
            "value": round(min(coverage, 1.0), 4),
            "original_columns": len(original_cols),
            "engineered_columns": len(engineered_cols),
            "new_features": len(new_features),
        }
        return coverage

    def compute_recommendation_confidence(self, df: pd.DataFrame) -> float:
        conf_col = "Recommendation_Confidence"
        if conf_col not in df.columns:
            self._metrics["recommendation_confidence"] = 0.0
            self._details["recommendation_confidence"] = {
                "description": "Average recommendation confidence score",
                "value": 0.0,
                "note": "No recommendation confidence column found",
            }
            return 0.0
        vals = pd.to_numeric(df[conf_col], errors="coerce").dropna()
        avg_conf = float(vals.mean()) if len(vals) > 0 else 0.0
        self._metrics["recommendation_confidence"] = round(avg_conf, 4)
        self._details["recommendation_confidence"] = {
            "description": "Average recommendation confidence score",
            "value": round(avg_conf, 4),
            "min": round(float(vals.min()), 4) if len(vals) > 0 else 0.0,
            "max": round(float(vals.max()), 4) if len(vals) > 0 else 0.0,
            "count": len(vals),
        }
        return avg_conf

    def compute_reckoner_completeness(self, df: pd.DataFrame) -> float:
        key_cols = ["Crop", "Nitrogen", "Phosphorus", "Potassium",
                     "Fertilizer_Name", "Dose", "Application_Interval"]
        present = [c for c in key_cols if c in df.columns]
        if not present:
            completeness = 0.0
        else:
            filled = 0
            total = 0
            for col in present:
                total += len(df)
                filled += df[col].notna().sum()
            completeness = filled / total if total > 0 else 0.0
        self._metrics["reckoner_completeness"] = round(completeness, 4)
        self._details["reckoner_completeness"] = {
            "description": "Fraction of key reckoner fields that are filled",
            "value": round(completeness, 4),
            "key_columns_checked": len(present),
            "total_key_columns": len(key_cols),
        }
        return completeness

    def compute_pipeline_success_rate(self, agent_results: list[dict]) -> float:
        if not agent_results:
            self._metrics["pipeline_success_rate"] = 0.0
            self._details["pipeline_success_rate"] = {
                "description": "Fraction of pipeline agents that succeeded",
                "value": 0.0,
                "note": "No agent results provided",
            }
            return 0.0
        success = sum(1 for r in agent_results if r.get("status") == "success")
        rate = success / len(agent_results)
        self._metrics["pipeline_success_rate"] = round(rate, 4)
        self._details["pipeline_success_rate"] = {
            "description": "Fraction of pipeline agents that succeeded",
            "value": round(rate, 4),
            "total_agents": len(agent_results),
            "successful": success,
            "failed": len(agent_results) - success,
        }
        return rate

    def compute_all(self, df: pd.DataFrame, schema_columns: list[str],
                     original_df: Optional[pd.DataFrame] = None,
                     agent_results: Optional[list[dict]] = None,
                     validation_issues: Optional[list] = None) -> dict:
        self.compute_extraction_accuracy(df)
        self.compute_schema_mapping_accuracy(df, schema_columns)
        self.compute_missing_data_rate(df)
        self.compute_validation_pass_rate(df, validation_issues)
        if original_df is not None:
            self.compute_feature_coverage(original_df, df)
        else:
            self._metrics["feature_coverage"] = 0.0
            self._details["feature_coverage"] = {"description": "No original df provided", "value": 0.0}
        self.compute_recommendation_confidence(df)
        self.compute_reckoner_completeness(df)
        if agent_results is not None:
            self.compute_pipeline_success_rate(agent_results)
        else:
            self._metrics["pipeline_success_rate"] = 0.0
            self._details["pipeline_success_rate"] = {"description": "No agent results provided", "value": 0.0}
        return self._metrics

    def _compute_filling_rate(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        total = df.shape[0] * df.shape[1]
        filled = df.notna().sum().sum()
        return float(filled / total) if total > 0 else 0.0

    def generate_html(self, title: str = "AAIF Pipeline Dashboard") -> str:
        rows = ""
        for name in METRIC_NAMES:
            val = self._metrics.get(name, 0.0)
            pct = val * 100
            detail = self._details.get(name, {})
            desc = detail.get("description", "")
            color = "#4CAF50" if val >= 0.8 else "#FF9800" if val >= 0.5 else "#F44336"
            rows += f"""
            <tr>
                <td style="padding:12px;border-bottom:1px solid #eee;font-weight:500">{name.replace('_', ' ').title()}</td>
                <td style="padding:12px;border-bottom:1px solid #eee;text-align:center">
                    <div style="background:{color};color:white;padding:4px 12px;border-radius:12px;display:inline-block">
                        {pct:.1f}%
                    </div>
                </td>
                <td style="padding:12px;border-bottom:1px solid #eee;color:#666">{desc}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
.container {{ max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); padding: 32px; }}
h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
th {{ text-align: left; padding: 12px; background: #f8f9fa; border-bottom: 2px solid #dee2e6; }}
.summary {{ margin-top: 24px; padding: 16px; background: #f0f7ff; border-radius: 8px; }}
</style></head><body>
<div class="container">
<h1>{title}</h1>
<p style="color:#666">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<table>
<thead><tr><th>Metric</th><th>Score</th><th>Description</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<div class="summary">
<strong>Overall Score:</strong> {self._overall_score()*100:.1f}%
</div></div></body></html>"""
        return html

    def save_dashboard(self, filename: str = "dashboard.html") -> Path:
        html = self.generate_html()
        path = self.output_dir / filename
        path.write_text(html, encoding="utf-8")
        return path

    def save_metrics_json(self, filename: str = "metrics.json") -> Path:
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._metrics,
            "details": self._details,
            "overall_score": self._overall_score(),
        }
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def _overall_score(self) -> float:
        if not self._metrics:
            return 0.0
        vals = list(self._metrics.values())
        return round(sum(vals) / len(vals), 4)

    def summary_text(self) -> str:
        lines = ["=" * 60, "AAIF Pipeline Evaluation Summary", "=" * 60]
        for name in METRIC_NAMES:
            val = self._metrics.get(name, 0.0)
            lines.append(f"  {name.replace('_', ' ').title():.<40s} {val*100:.1f}%")
        lines.append(f"  {'Overall Score':.<40s} {self._overall_score()*100:.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)
