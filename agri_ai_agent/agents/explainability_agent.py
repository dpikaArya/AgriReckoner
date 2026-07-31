"""
Explainability Agent — AAIF v2.0.

For every prediction and recommendation generates a full explanation:
model used, prediction confidence, feature importance, top-10 variables,
activated fuzzy rules, evidence papers, supporting treatments, crops,
soil/climate conditions, data quality, extraction confidence,
reason for recommendation, alternatives, and uncertainty.
Produces one JSON report per prediction under reports/explainability/.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract


class ExplainabilityAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ExplainabilityAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.log.info("=" * 60)
        self.log.info("ExplainabilityAgent: Generating prediction explanations")
        self.log.info("=" * 60)

        pipeline_results = kwargs.get("pipeline_results", {})
        model_metrics = kwargs.get("model_metrics", {})
        feature_importances = kwargs.get("feature_importances", {})

        explanations = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            explanation = self._explain_row(
                row, idx, df, pipeline_results, model_metrics, feature_importances
            )
            explanations.append(explanation)

        self._save_explanations(explanations)
        self._save_feature_importance_csv(explanations)
        self._generate_summary_dashboard(explanations, df)

        self.log.info("Generated %d explanations", len(explanations))

        self._last_explanations = explanations
        self._last_avg_confidence = self._avg_confidence(explanations)
        return df

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        base = super()._build_output(df, **kwargs)
        if hasattr(self, "_last_explanations"):
            base["explanation_count"] = len(self._last_explanations)
            base["avg_confidence"] = self._last_avg_confidence
        return base

    def _explain_row(
        self,
        row: pd.Series,
        idx: int,
        df: pd.DataFrame,
        pipeline_results: dict,
        model_metrics: dict,
        feature_importances: dict,
    ) -> dict:
        crop = str(row.get("Crop", "Unknown"))
        source_paper = str(row.get("Source_Paper", "Unknown"))
        predicted_yield = self._safe_float(row.get("Predicted_Yield"))
        target_yield = self._safe_float(row.get("Target_Yield"))
        actual_yield = self._safe_float(row.get("Yield_per_Hectare"))
        confidence = self._safe_float(
            row.get("Confidence_Score", row.get("Recommendation_Confidence", 0.0))
        )
        fertilizer = str(row.get("Recommended_Fertilizer", row.get("Fertilizer_Name", "Unknown")))
        dose = self._safe_float(row.get("Recommended_Dose", row.get("Dose", 0.0)))

        top_features = self._get_top_features(feature_importances, idx, n=10)
        fuzzy_rules = self._get_activated_fuzzy_rules(row)
        uncertainty = self._compute_uncertainty(row, predicted_yield, actual_yield)

        explanation = {
            "prediction_id": f"PRED_{idx:05d}",
            "timestamp": datetime.now().isoformat(),
            "crop": crop,
            "source_paper": source_paper,
            "prediction": {
                "model_used": self._detect_model(pipeline_results),
                "predicted_yield": predicted_yield,
                "target_yield": target_yield,
                "actual_yield": actual_yield,
                "confidence": confidence,
                "uncertainty": uncertainty,
            },
            "feature_importance": {
                "top_10_variables": top_features,
                "all_features": feature_importances.get("global", {}),
            },
            "fuzzy_rules": {
                "activated_rules": fuzzy_rules,
                "rule_count": len(fuzzy_rules),
            },
            "evidence": {
                "papers_used": self._get_evidence_papers(row),
                "supporting_treatments": self._get_supporting_treatments(row),
                "supporting_crops": self._get_supporting_crops(row, df),
                "soil_conditions": self._get_soil_conditions(row),
                "climate_conditions": self._get_climate_conditions(row),
            },
            "data_quality": {
                "data_quality_score": self._compute_data_quality_score(row),
                "extraction_confidence": self._get_extraction_confidence(row),
                "missing_fields": self._get_missing_fields(row),
            },
            "recommendation": {
                "fertilizer": fertilizer,
                "dose": dose,
                "application_interval": self._safe_float(row.get("Application_Interval", 0)),
                "reason_for_recommendation": self._get_recommendation_reason(row),
                "alternatives": self._get_alternatives(row),
            },
        }
        return explanation

    def _safe_float(self, val) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        try:
            return round(float(val), 4)
        except (ValueError, TypeError):
            return 0.0

    def _detect_model(self, pipeline_results: dict) -> str:
        training = pipeline_results.get("training")
        if training is not None:
            if isinstance(training, AgentContract):
                artifacts = training.artifacts
            elif isinstance(training, dict):
                artifacts = training.get("artifacts", [])
            else:
                artifacts = []
            for a in artifacts:
                if "xgboost" in str(a).lower():
                    return "XGBoost"
                elif "random_forest" in str(a).lower() or "rf" in str(a).lower():
                    return "Random Forest"
                elif "gradient" in str(a).lower():
                    return "Gradient Boosting"
                elif "ridge" in str(a).lower():
                    return "Ridge Regression"
        return "Ensemble"

    def _get_top_features(self, feature_importances: dict, idx: int, n: int = 10) -> list[dict]:
        row_imp = feature_importances.get(idx, feature_importances.get("global", {}))
        if not row_imp:
            return []
        sorted_features = sorted(row_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
        return [
            {"feature": name, "importance": round(float(imp), 4)} for name, imp in sorted_features
        ]

    def _get_activated_fuzzy_rules(self, row: pd.Series) -> list[dict]:
        rules = []
        fuzzy_cols = [c for c in row.index if c.startswith("Fuzzy_")]
        for col in fuzzy_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                rules.append(
                    {
                        "rule": col,
                        "output": str(val),
                        "category": col.replace("Fuzzy_", ""),
                    }
                )
        n_action = str(row.get("Fuzzy_N_Action", ""))
        p_action = str(row.get("Fuzzy_P_Action", ""))
        k_action = str(row.get("Fuzzy_K_Action", ""))
        risk = str(row.get("Fuzzy_Risk", ""))
        if n_action and n_action != "nan":
            rules.append({"rule": "N_Deficiency", "output": n_action, "category": "Nitrogen"})
        if p_action and p_action != "nan":
            rules.append({"rule": "P_Deficiency", "output": p_action, "category": "Phosphorus"})
        if k_action and k_action != "nan":
            rules.append({"rule": "K_Deficiency", "output": k_action, "category": "Potassium"})
        if risk and risk != "nan":
            rules.append({"rule": "Risk_Assessment", "output": risk, "category": "Risk"})
        return rules

    def _compute_uncertainty(self, row: pd.Series, predicted: float, actual: float) -> dict:
        confidence = self._safe_float(
            row.get("Confidence_Score", row.get("Recommendation_Confidence", 0.5))
        )
        pred_std = 0.0
        if predicted > 0 and actual > 0:
            pred_std = abs(predicted - actual) / predicted
        uncertainty_score = max(0.0, 1.0 - confidence + pred_std * 0.3)
        return {
            "uncertainty_score": round(min(uncertainty_score, 1.0), 4),
            "confidence_interval_width": round(predicted * (1 - confidence) * 2, 4)
            if predicted > 0
            else 0.0,
            "prediction_std": round(pred_std, 4),
        }

    def _get_evidence_papers(self, row: pd.Series) -> list[str]:
        source = str(row.get("Source_Paper", ""))
        if source and source != "nan":
            return [source]
        return []

    def _get_supporting_treatments(self, row: pd.Series) -> list[dict]:
        treatments = []
        fert = str(row.get("Fertilizer_Name", row.get("Recommended_Fertilizer", "")))
        dose = self._safe_float(row.get("Dose", row.get("Recommended_Dose", 0)))
        if fert and fert != "nan":
            treatments.append({"fertilizer": fert, "dose_kg_ha": dose})
        return treatments

    def _get_supporting_crops(self, row: pd.Series, df: pd.DataFrame) -> list[str]:
        crop = str(row.get("Crop", ""))
        if crop and crop != "nan" and "Crop" in df.columns:
            return df["Crop"].dropna().unique().tolist()[:5]
        return []

    def _get_soil_conditions(self, row: pd.Series) -> dict:
        return {
            "ph": self._safe_float(row.get("Soil_pH")),
            "nitrogen": self._safe_float(row.get("Nitrogen")),
            "phosphorus": self._safe_float(row.get("Phosphorus")),
            "potassium": self._safe_float(row.get("Potassium")),
            "organic_carbon": self._safe_float(row.get("Organic_Carbon")),
        }

    def _get_climate_conditions(self, row: pd.Series) -> dict:
        return {
            "rainfall_mm": self._safe_float(row.get("Rainfall")),
            "temp_max_c": self._safe_float(row.get("Temperature_Max")),
            "temp_min_c": self._safe_float(row.get("Temperature_Min")),
            "humidity_pct": self._safe_float(row.get("Humidity")),
        }

    def _compute_data_quality_score(self, row: pd.Series) -> float:
        important = [
            "Crop",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Soil_pH",
            "Yield_per_Hectare",
            "Rainfall",
        ]
        present = sum(1 for c in important if c in row.index and pd.notna(row.get(c)))
        return round(present / len(important), 4) if important else 0.0

    def _get_extraction_confidence(self, row: pd.Series) -> float:
        conf = row.get("Provenance_Confidence", row.get("Extraction_Confidence", 0.5))
        return self._safe_float(conf)

    def _get_missing_fields(self, row: pd.Series) -> list[str]:
        missing = []
        important = [
            "Crop",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Soil_pH",
            "Yield_per_Hectare",
            "Rainfall",
            "Fertilizer_Name",
            "Dose",
        ]
        for col in important:
            if col in row.index and pd.isna(row.get(col)):
                missing.append(col)
        return missing

    def _get_recommendation_reason(self, row: pd.Series) -> str:
        summary = str(row.get("Recommendation_Summary", ""))
        if summary and summary != "nan":
            return summary
        fert = str(row.get("Recommended_Fertilizer", row.get("Fertilizer_Name", "")))
        crop = str(row.get("Crop", ""))
        confidence = self._safe_float(
            row.get("Confidence_Score", row.get("Recommendation_Confidence", 0))
        )
        if fert and fert != "nan":
            return f"{fert} recommended for {crop} with {confidence * 100:.0f}% confidence based on soil analysis and research evidence."
        return "Insufficient data for recommendation reasoning."

    def _get_alternatives(self, row: pd.Series) -> list[dict]:
        alt_col = "Top_Alternatives"
        if alt_col in row.index and pd.notna(row.get(alt_col)):
            val = row[alt_col]
            if isinstance(val, str):
                try:
                    alts = json.loads(val)
                    if isinstance(alts, list):
                        return alts[:3]
                except json.JSONDecodeError:
                    pass
        return []

    def _save_explanations(self, explanations: list[dict]):
        out = self.settings.REPORTS_DIR / "explainability"
        out.mkdir(parents=True, exist_ok=True)
        for exp in explanations:
            path = out / f"{exp['prediction_id']}.json"
            path.write_text(json.dumps(exp, indent=2, default=str), encoding="utf-8")
        summary_path = out / "prediction_explanations.json"
        summary_path.write_text(json.dumps(explanations, indent=2, default=str), encoding="utf-8")
        if self.contract is not None:
            self.contract.artifacts.append(str(summary_path))

    def _save_feature_importance_csv(self, explanations: list[dict]):
        out = self.settings.REPORTS_DIR / "explainability"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for exp in explanations:
            pred_id = exp["prediction_id"]
            for feat in exp.get("feature_importance", {}).get("top_10_variables", []):
                rows.append(
                    {
                        "prediction_id": pred_id,
                        "crop": exp.get("crop", ""),
                        "feature": feat["feature"],
                        "importance": feat["importance"],
                    }
                )
        df = (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["prediction_id", "crop", "feature", "importance"])
        )
        path = out / "feature_importance.csv"
        df.to_csv(path, index=False)
        if self.contract is not None:
            self.contract.artifacts.append(str(path))

    def _generate_summary_dashboard(self, explanations: list[dict], df: pd.DataFrame):
        out = self.settings.REPORTS_DIR / "explainability"
        out.mkdir(parents=True, exist_ok=True)
        confidences = [e["prediction"]["confidence"] for e in explanations]
        uncertainties = [e["prediction"]["uncertainty"]["uncertainty_score"] for e in explanations]
        quality_scores = [e["data_quality"]["data_quality_score"] for e in explanations]

        dashboard = {
            "total_predictions": len(explanations),
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "avg_uncertainty": round(float(np.mean(uncertainties)), 4) if uncertainties else 0.0,
            "avg_data_quality": round(float(np.mean(quality_scores)), 4) if quality_scores else 0.0,
            "crop_distribution": {},
            "model_used": explanations[0]["prediction"]["model_used"]
            if explanations
            else "Unknown",
            "timestamp": datetime.now().isoformat(),
        }
        for exp in explanations:
            crop = exp.get("crop", "Unknown")
            dashboard["crop_distribution"][crop] = dashboard["crop_distribution"].get(crop, 0) + 1

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AAIF Explainability Dashboard</title>
<style>body{{font-family:sans-serif;margin:40px;background:#f5f5f5}}
.container{{max-width:800px;margin:0 auto;background:white;border-radius:12px;padding:32px;box-shadow:0 2px 12px rgba(0,0,0,.08)}}
h1{{color:#333;border-bottom:2px solid #2196F3;padding-bottom:10px}}
.metric{{display:inline-block;margin:10px 20px 10px 0;padding:15px 20px;background:#f8f9fa;border-radius:8px;text-align:center}}
.metric .value{{font-size:24px;font-weight:bold;color:#2196F3}}
.metric .label{{font-size:12px;color:#666;margin-top:4px}}</style></head>
<body><div class="container">
<h1>AAIF Explainability Dashboard</h1>
<p>Generated: {dashboard["timestamp"]}</p>
<div class="metric"><div class="value">{dashboard["total_predictions"]}</div><div class="label">Predictions</div></div>
<div class="metric"><div class="value">{dashboard["avg_confidence"] * 100:.1f}%</div><div class="label">Avg Confidence</div></div>
<div class="metric"><div class="value">{dashboard["avg_uncertainty"] * 100:.1f}%</div><div class="label">Avg Uncertainty</div></div>
<div class="metric"><div class="value">{dashboard["avg_data_quality"] * 100:.1f}%</div><div class="label">Data Quality</div></div>
<div class="metric"><div class="value">{dashboard["model_used"]}</div><div class="label">Model</div></div>
<h2>Crop Distribution</h2>
<ul>{"".join(f"<li>{c}: {n}</li>" for c, n in dashboard["crop_distribution"].items())}</ul>
</div></body></html>"""
        path = out / "explainability_dashboard.html"
        path.write_text(html, encoding="utf-8")
        if self.contract is not None:
            self.contract.artifacts.append(str(path))

    def _avg_confidence(self, explanations: list[dict]) -> float:
        if not explanations:
            return 0.0
        vals = [e["prediction"]["confidence"] for e in explanations]
        return round(float(np.mean(vals)), 4)
