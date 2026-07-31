"""
Recommendation Agent v2 — Phase 9.

Pipeline: prediction → knowledge → evidence ranking → fuzzy reasoning →
confidence → supporting papers → alternatives.

Returns top-3 recommendations per row with risk/economic/environmental scores.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent

CROP_FERTILIZER_PREFS = {
    "Rice": ["Urea", "DAP", "MOP", "Bio NPK"],
    "Wheat": ["Urea", "DAP", "MOP", "10-26-26"],
    "Maize": ["Urea", "DAP", "12-32-16", "Bio NPK"],
    "Cotton": ["Urea", "DAP", "MOP", "15-15-15"],
    "Sugarcane": ["Urea", "DAP", "MOP", "20-20-0"],
    "Tomato": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Potato": ["Urea", "DAP", "15-15-15", "MOP"],
    "Onion": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Chilli": ["Urea", "DAP", "MOP", "Bio NPK"],
    "Soybean": ["DAP", "Bio NPK", "Urea", "MOP"],
    "Groundnut": ["DAP", "Urea", "CaCO3", "Bio NPK"],
    "Pea": ["DAP", "Bio NPK", "MOP", "Urea"],
    "Sunflower": ["Urea", "DAP", "MOP", "10-26-26"],
    "Mustard": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Black Gram": ["DAP", "Urea", "Bio NPK", "MOP"],
    "Green Gram": ["DAP", "Urea", "Bio NPK", "MOP"],
    "Bengal Gram": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Rapeseed": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Lentil": ["DAP", "Bio NPK", "Urea", "MOP"],
    "Cauliflower": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Cabbage": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Carrot": ["DAP", "Urea", "MOP", "Bio NPK"],
    "Spinach": ["Urea", "DAP", "Bio NPK", "MOP"],
}

NUTRIENT_DEFICIENCY_ACTION = {
    "low": {"N": "Urea", "P": "DAP", "K": "MOP"},
    "medium": {"N": "15-15-15", "P": "DAP", "K": "MOP"},
    "high": {"N": None, "P": None, "K": None},
}

FERTILIZER_COST_PER_KG = {
    "Urea": 6.0,
    "DAP": 28.0,
    "MOP": 22.0,
    "SSP": 5.0,
    "CAN": 12.0,
    "SOP": 30.0,
    "10-26-26": 20.0,
    "12-32-16": 22.0,
    "15-15-15": 18.0,
    "20-20-0": 16.0,
    "Bio NPK": 8.0,
    "Ammonium Sulphate": 10.0,
    "Potassium Sulphate": 25.0,
}


class RecommendationAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "RecommendationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        top_n = kwargs.get("top_n", 3)

        df = self._recommend_fertilizer(df)
        df = self._recommend_dose(df)
        df = self._recommend_interval(df)
        df = self._compute_yield_increase(df)
        df = self._compute_economic_score(df)
        df = self._compute_environmental_score(df)
        df = self._compute_risk_score(df)
        df = self._final_confidence(df)
        df = self._generate_top_alternatives(df, top_n=top_n)
        df = self._generate_summary(df)

        n_rec = df["Recommended_Fertilizer"].notna().sum()
        self.log.info(
            "Recommendations generated for %d/%d rows (top %d alternatives)", n_rec, len(df), top_n
        )

        report = self._build_report(df, top_n)
        self.save_text_artifact(report, "recommendation_report_v2.md")

        self.dataframe = df
        return df

    def _init_col(self, df: pd.DataFrame, col: str, dtype: type, default=None) -> None:
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)
        if default is not None:
            df[col] = df[col].fillna(default)

    def _recommend_fertilizer(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Fertilizer", object)
        self._init_col(df, "Fertilizer_NPK", object)

        for idx in df.index:
            if pd.notna(df.at[idx, "Recommended_Fertilizer"]):
                continue

            crop = str(df.at[idx, "Crop"]).strip() if "Crop" in df.columns else ""
            fert = df.at[idx, "Fertilizer_Name"] if "Fertilizer_Name" in df.columns else None
            n_status = df.at[idx, "N_Status"] if "N_Status" in df.columns else None
            p_status = df.at[idx, "P_Status"] if "P_Status" in df.columns else None
            k_status = df.at[idx, "K_Status"] if "K_Status" in df.columns else None

            if n_status == "low":
                df.at[idx, "Recommended_Fertilizer"] = "Urea"
                if pd.isna(df.at[idx, "Fertilizer_NPK"]):
                    df.at[idx, "Fertilizer_NPK"] = "46-0-0"
            elif p_status == "low":
                df.at[idx, "Recommended_Fertilizer"] = "DAP"
                if pd.isna(df.at[idx, "Fertilizer_NPK"]):
                    df.at[idx, "Fertilizer_NPK"] = "18-46-0"
            elif k_status == "low":
                df.at[idx, "Recommended_Fertilizer"] = "MOP"
                if pd.isna(df.at[idx, "Fertilizer_NPK"]):
                    df.at[idx, "Fertilizer_NPK"] = "0-0-60"
            elif pd.notna(fert):
                df.at[idx, "Recommended_Fertilizer"] = fert
            elif crop in CROP_FERTILIZER_PREFS:
                df.at[idx, "Recommended_Fertilizer"] = CROP_FERTILIZER_PREFS[crop][0]
            else:
                df.at[idx, "Recommended_Fertilizer"] = "Bio NPK"

        return df

    def _recommend_dose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Dose", float)

        for idx in df.index:
            if pd.notna(df.at[idx, "Recommended_Dose"]):
                continue

            base = df.at[idx, "Dose"] if "Dose" in df.columns else 100
            if pd.isna(base):
                base = 100

            n_status = df.at[idx, "N_Status"] if "N_Status" in df.columns else "medium"
            rain = df.at[idx, "Rainfall"] if "Rainfall" in df.columns else None

            dose = base
            if n_status == "low":
                dose *= 1.25
            elif n_status == "high":
                dose *= 0.75

            if pd.notna(rain) and rain > 1200:
                dose *= 0.90
            elif pd.notna(rain) and rain < 400:
                dose *= 1.10

            ph = df.at[idx, "Soil_pH"] if "Soil_pH" in df.columns else None
            if pd.notna(ph) and ph < 5.5:
                dose *= 1.05
            elif pd.notna(ph) and ph > 7.5:
                dose *= 0.95

            df.at[idx, "Recommended_Dose"] = round(dose, 1)

        return df

    def _recommend_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Application_Interval", float)

        for idx in df.index:
            if pd.notna(df.at[idx, "Recommended_Application_Interval"]):
                continue

            base = (
                df.at[idx, "Application_Interval"] if "Application_Interval" in df.columns else 10
            )
            if pd.isna(base):
                base = 10

            rainfall = df.at[idx, "Rainfall"] if "Rainfall" in df.columns else None
            temp = (
                df.at[idx, "Average_Temperature"] if "Average_Temperature" in df.columns else None
            )

            interval = base
            if pd.notna(rainfall) and rainfall > 1000:
                interval *= 1.5
            elif pd.notna(rainfall) and rainfall < 500:
                interval *= 0.75

            if pd.notna(temp) and temp > 35:
                interval *= 0.85
            elif pd.notna(temp) and temp < 15:
                interval *= 1.15

            df.at[idx, "Recommended_Application_Interval"] = round(interval, 1)

        return df

    def _compute_yield_increase(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Expected_Yield_Increase", float)
        self._init_col(df, "Expected_Yield_Increase_Pct", float)

        for idx in df.index:
            if pd.notna(df.at[idx, "Expected_Yield_Increase"]):
                continue

            predicted = df.at[idx, "Predicted_Yield"] if "Predicted_Yield" in df.columns else None
            baseline = (
                df.at[idx, "Yield_per_Hectare"] if "Yield_per_Hectare" in df.columns else None
            )

            if pd.notna(predicted) and pd.notna(baseline) and baseline > 0:
                increase = predicted - baseline
                df.at[idx, "Expected_Yield_Increase"] = round(increase, 2)
                df.at[idx, "Expected_Yield_Increase_Pct"] = round((increase / baseline) * 100, 1)
            elif pd.notna(predicted):
                df.at[idx, "Expected_Yield_Increase"] = round(predicted * 0.1, 2)
                df.at[idx, "Expected_Yield_Increase_Pct"] = 10.0

        return df

    def _compute_economic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Economic_Score", float)

        for idx in df.index:
            fert = (
                df.at[idx, "Recommended_Fertilizer"]
                if "Recommended_Fertilizer" in df.columns
                else None
            )
            dose = df.at[idx, "Recommended_Dose"] if "Recommended_Dose" in df.columns else None
            yield_inc = (
                df.at[idx, "Expected_Yield_Increase"]
                if "Expected_Yield_Increase" in df.columns
                else None
            )

            if pd.isna(fert) or pd.isna(dose):
                continue

            cost_per_kg = FERTILIZER_COST_PER_KG.get(str(fert), 15.0)
            total_cost = cost_per_kg * (dose / 1000.0)

            revenue_per_ton = 25000
            expected_revenue = 0
            if pd.notna(yield_inc):
                expected_revenue = (yield_inc / 1000.0) * revenue_per_ton

            if total_cost > 0 and expected_revenue > 0:
                roi = expected_revenue / total_cost
                score = min(1.0, max(0.0, roi / 10.0))
            elif expected_revenue > 0:
                score = 0.7
            else:
                score = 0.5

            df.at[idx, "Economic_Score"] = round(score, 3)

        return df

    def _compute_environmental_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Environmental_Score", float)

        for idx in df.index:
            fert = (
                str(df.at[idx, "Recommended_Fertilizer"])
                if "Recommended_Fertilizer" in df.columns
                else ""
            )
            dose = df.at[idx, "Recommended_Dose"] if "Recommended_Dose" in df.columns else None

            organic_ferts = {"Bio NPK", "Jaivik Khad", "Vermicompost", "FYM"}
            score = 0.5

            if fert in organic_ferts:
                score = 0.9
            elif "Bio" in fert or "organic" in fert.lower():
                score = 0.8
            else:
                if pd.notna(dose):
                    if dose < 50:
                        score = 0.7
                    elif dose < 100:
                        score = 0.5
                    elif dose < 200:
                        score = 0.4
                    else:
                        score = 0.3

            df.at[idx, "Environmental_Score"] = round(score, 3)

        return df

    def _compute_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Risk_Score", float)

        for idx in df.index:
            conf = df.at[idx, "Confidence_Score"] if "Confidence_Score" in df.columns else None
            econ = df.at[idx, "Economic_Score"] if "Economic_Score" in df.columns else None
            env = df.at[idx, "Environmental_Score"] if "Environmental_Score" in df.columns else None

            scores = [s for s in [conf, econ, env] if pd.notna(s)]
            if not scores:
                continue

            avg = np.mean(scores)
            risk = round(1.0 - avg, 3)
            df.at[idx, "Risk_Score"] = max(0.0, min(1.0, risk))

        return df

    def _final_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Confidence_Score", float)

        for idx in df.index:
            if pd.notna(df.at[idx, "Confidence_Score"]):
                continue

            score = 0.3
            checks = [
                ("Predicted_Yield", 0.2),
                ("Recommended_Fertilizer", 0.1),
                ("Recommended_Dose", 0.1),
                ("Recommended_Application_Interval", 0.1),
                ("Yield_per_Hectare", 0.1),
                ("Soil_pH", 0.05),
                ("Nitrogen", 0.05),
            ]
            for col, weight in checks:
                if col in df.columns and pd.notna(df.at[idx, col]):
                    score += weight

            n_missing = sum(
                1
                for col in ["Soil_pH", "Nitrogen", "Rainfall"]
                if col in df.columns and pd.isna(df.at[idx, col])
            )
            score -= n_missing * 0.05

            df.at[idx, "Confidence_Score"] = round(max(0.1, min(1.0, score)), 3)

        return df

    def _generate_top_alternatives(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        self._init_col(df, "Top_Alternatives", object)

        for idx in df.index:
            crop = str(df.at[idx, "Crop"]).strip() if "Crop" in df.columns else ""
            current = (
                df.at[idx, "Recommended_Fertilizer"]
                if "Recommended_Fertilizer" in df.columns
                else None
            )
            n_status = df.at[idx, "N_Status"] if "N_Status" in df.columns else "medium"
            p_status = df.at[idx, "P_Status"] if "P_Status" in df.columns else "medium"
            k_status = df.at[idx, "K_Status"] if "K_Status" in df.columns else "medium"

            candidates = []
            if crop in CROP_FERTILIZER_PREFS:
                candidates = CROP_FERTILIZER_PREFS[crop]
            else:
                candidates = ["Urea", "DAP", "MOP", "15-15-15", "Bio NPK", "10-26-26"]

            scored = []
            for fert in candidates:
                if fert == current:
                    continue
                score = 0.5
                if n_status == "low" and fert == "Urea":
                    score += 0.3
                if p_status == "low" and fert == "DAP":
                    score += 0.3
                if k_status == "low" and fert == "MOP":
                    score += 0.3
                if "Bio" in fert:
                    score += 0.1

                scored.append({"fertilizer": fert, "score": round(score, 3)})

            scored.sort(key=lambda x: x["score"], reverse=True)
            alts = scored[:top_n]
            df.at[idx, "Top_Alternatives"] = json.dumps(alts, default=str)

        return df

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommendation_Summary", object)

        for idx in df.index:
            parts = []
            conditions = self._describe_conditions(df, idx)
            if conditions:
                parts.append("Current: " + "; ".join(conditions) + ".")

            fert = df.at[idx, "Recommended_Fertilizer"]
            dose = df.at[idx, "Recommended_Dose"]
            interval = df.at[idx, "Recommended_Application_Interval"]

            rec_parts = []
            if pd.notna(fert):
                rec_parts.append(str(fert))
            if pd.notna(dose):
                rec_parts.append(f"{dose:.1f} kg/ha")
            if pd.notna(interval):
                rec_parts.append(f"every {interval:.0f} days")
            if rec_parts:
                parts.append("Apply " + " ".join(rec_parts) + ".")

            increase = df.at[idx, "Expected_Yield_Increase"]
            pct = df.at[idx, "Expected_Yield_Increase_Pct"]
            if pd.notna(increase):
                sym = "+" if increase >= 0 else ""
                pct_str = f" ({sym}{pct:.1f}%)" if pd.notna(pct) else ""
                parts.append(f"Expected yield {sym}{increase:.1f} kg/ha{pct_str} vs baseline.")

            conf = df.at[idx, "Confidence_Score"]
            if pd.notna(conf):
                c_pct = conf * 100
                label = "High" if c_pct >= 70 else "Moderate" if c_pct >= 40 else "Low"
                parts.append(f"{label} confidence ({c_pct:.0f}%).")

            econ = df.at[idx, "Economic_Score"]
            if pd.notna(econ):
                e_label = "Favorable" if econ >= 0.6 else "Moderate" if econ >= 0.4 else "Marginal"
                parts.append(f"Economic outlook: {e_label} ({econ:.0%}).")

            env = df.at[idx, "Environmental_Score"]
            if pd.notna(env):
                env_label = (
                    "Low impact" if env >= 0.7 else "Moderate" if env >= 0.5 else "Higher impact"
                )
                parts.append(f"Environmental: {env_label} ({env:.0%}).")

            risk = df.at[idx, "Risk_Score"]
            if pd.notna(risk):
                r_label = (
                    "Low risk" if risk <= 0.3 else "Moderate risk" if risk <= 0.6 else "Higher risk"
                )
                parts.append(f"{r_label} ({risk:.0%}).")

            df.at[idx, "Recommendation_Summary"] = " ".join(parts)

        return df

    def _describe_conditions(self, df: pd.DataFrame, idx) -> list[str]:
        desc = []
        checks = [
            ("Nitrogen", "ppm", 50),
            ("Phosphorus", "ppm", 15),
            ("Potassium", "ppm", 100),
            ("Rainfall", "mm", 500),
        ]
        for col, unit, low_thresh in checks:
            val = df.at[idx, col] if col in df.columns else None
            if pd.notna(val):
                if val < low_thresh:
                    desc.append(f"{col} {val:.0f}{unit} (low)")
                else:
                    desc.append(f"{col} {val:.0f}{unit}")

        ph = df.at[idx, "Soil_pH"] if "Soil_pH" in df.columns else None
        if pd.notna(ph):
            ph_desc = f"Soil pH {ph:.1f}"
            if ph < 5.5:
                ph_desc += " (acidic)"
            elif ph > 7.5:
                ph_desc += " (alkaline)"
            desc.append(ph_desc)

        temp = df.at[idx, "Average_Temperature"] if "Average_Temperature" in df.columns else None
        if pd.notna(temp):
            desc.append(f"Temperature {temp:.1f}°C")

        return desc

    def _build_report(self, df: pd.DataFrame, top_n: int) -> str:
        lines = [
            "# Recommendation Report v2",
            f"Generated: {datetime.now().isoformat()}",
            f"Rows: {len(df)} | Top-{top_n} alternatives per row",
            "",
            "## Summary Statistics",
        ]

        if "Recommended_Fertilizer" in df.columns:
            vc = df["Recommended_Fertilizer"].value_counts()
            lines.append("\n### Recommended Fertilizer Distribution")
            for fert, count in vc.items():
                lines.append(f"  - {fert}: {count}")

        if "Confidence_Score" in df.columns:
            conf = df["Confidence_Score"].dropna()
            if len(conf) > 0:
                lines.extend(
                    [
                        "\n### Confidence Score Distribution",
                        f"  - Mean: {conf.mean():.3f}",
                        f"  - Median: {conf.median():.3f}",
                        f"  - Min: {conf.min():.3f}",
                        f"  - Max: {conf.max():.3f}",
                    ]
                )

        if "Economic_Score" in df.columns:
            econ = df["Economic_Score"].dropna()
            if len(econ) > 0:
                lines.extend(
                    [
                        "\n### Economic Score",
                        f"  - Mean: {econ.mean():.3f}",
                    ]
                )

        if "Environmental_Score" in df.columns:
            env = df["Environmental_Score"].dropna()
            if len(env) > 0:
                lines.extend(
                    [
                        "\n### Environmental Score",
                        f"  - Mean: {env.mean():.3f}",
                    ]
                )

        if "Risk_Score" in df.columns:
            risk = df["Risk_Score"].dropna()
            if len(risk) > 0:
                low = (risk <= 0.3).sum()
                med = ((risk > 0.3) & (risk <= 0.6)).sum()
                high = (risk > 0.6).sum()
                lines.extend(
                    [
                        "\n### Risk Distribution",
                        f"  - Low risk (≤30%): {low}",
                        f"  - Moderate (30-60%): {med}",
                        f"  - High risk (>60%): {high}",
                    ]
                )

        lines.extend(
            [
                "",
                "## Sample Recommendations",
                "",
            ]
        )
        sample = df.head(min(5, len(df)))
        for i, row in sample.iterrows():
            crop = row.get("Crop", "Unknown")
            fert = row.get("Recommended_Fertilizer", "N/A")
            dose = row.get("Recommended_Dose", "N/A")
            conf = row.get("Confidence_Score", "N/A")
            lines.append(f"**Row {i}** ({crop}): {fert} @ {dose} kg/ha (conf={conf})")

        lines.append("\n*Report generated by RecommendationAgent v2*")
        return "\n".join(lines)

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "recommendation_complete": True,
        }
