"""
Recommendation Agent
Combines ML predictions + fuzzy reasoning into final fertilizer
recommendations with human-readable summaries.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ades.agents.base_agent import BaseAgent


class RecommendationAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "RecommendationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        # Step 1: Determine recommended fertilizer
        df = self._recommend_fertilizer(df)

        # Step 2: Determine recommended dose
        df = self._recommend_dose(df)

        # Step 3: Determine application interval
        df = self._recommend_interval(df)

        # Step 4: Compute expected yield increase
        df = self._compute_yield_increase(df)

        # Step 5: Final confidence score
        df = self._final_confidence(df)

        # Step 6: Generate summary
        df = self._generate_summary(df)

        n_rec = df["Recommended_Fertilizer"].notna().sum()
        self.log.info("Recommendations generated for %d/%d rows", n_rec, len(df))

        self.dataframe = df
        return df

    def _init_col(self, df: pd.DataFrame, col: str, dtype: type) -> None:
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)
        elif df[col].dtype != dtype:
            df[col] = df[col].astype(dtype)

    def _recommend_fertilizer(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Fertilizer", object)

        for row_idx in df.index:
            if pd.notna(df.at[row_idx, "Recommended_Fertilizer"]):
                continue

            fert = df.at[row_idx, "Fertilizer_Name"] if "Fertilizer_Name" in df.columns else None
            n_status = df.at[row_idx, "N_Status"] if "N_Status" in df.columns else None
            p_status = df.at[row_idx, "P_Status"] if "P_Status" in df.columns else None

            if n_status == "low":
                df.at[row_idx, "Recommended_Fertilizer"] = "Urea"
            elif p_status == "low":
                df.at[row_idx, "Recommended_Fertilizer"] = "DAP"
            elif pd.notna(fert):
                df.at[row_idx, "Recommended_Fertilizer"] = fert
            else:
                df.at[row_idx, "Recommended_Fertilizer"] = "Bio NPK"

        return df

    def _recommend_dose(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Dose", float)

        for row_idx in df.index:
            if pd.notna(df.at[row_idx, "Recommended_Dose"]):
                continue

            base = df.at[row_idx, "Dose"] if "Dose" in df.columns else 100
            if pd.isna(base):
                base = 100

            n_status = df.at[row_idx, "N_Status"] if "N_Status" in df.columns else "medium"

            if n_status == "low":
                df.at[row_idx, "Recommended_Dose"] = base * 1.25
            elif n_status == "high":
                df.at[row_idx, "Recommended_Dose"] = base * 0.75
            else:
                df.at[row_idx, "Recommended_Dose"] = base

        return df

    def _recommend_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommended_Application_Interval", float)

        for row_idx in df.index:
            if pd.notna(df.at[row_idx, "Recommended_Application_Interval"]):
                continue

            base = df.at[row_idx, "Application_Interval"] if "Application_Interval" in df.columns else 10
            if pd.isna(base):
                base = 10

            rainfall = df.at[row_idx, "Rainfall"] if "Rainfall" in df.columns else None

            if pd.notna(rainfall) and rainfall > 1000:
                df.at[row_idx, "Recommended_Application_Interval"] = base * 1.5
            elif pd.notna(rainfall) and rainfall < 500:
                df.at[row_idx, "Recommended_Application_Interval"] = base * 0.75
            else:
                df.at[row_idx, "Recommended_Application_Interval"] = base

        return df

    def _compute_yield_increase(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Expected_Yield_Increase", float)

        for row_idx in df.index:
            if pd.notna(df.at[row_idx, "Expected_Yield_Increase"]):
                continue

            predicted = df.at[row_idx, "Predicted_Yield"] if "Predicted_Yield" in df.columns else None
            baseline = df.at[row_idx, "Yield_per_Hectare"] if "Yield_per_Hectare" in df.columns else None

            if pd.notna(predicted) and pd.notna(baseline) and baseline > 0:
                df.at[row_idx, "Expected_Yield_Increase"] = predicted - baseline
            elif pd.notna(predicted):
                df.at[row_idx, "Expected_Yield_Increase"] = predicted * 0.1

        return df

    def _final_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Confidence_Score", float)

        for row_idx in df.index:
            if pd.notna(df.at[row_idx, "Confidence_Score"]):
                continue

            score = 0.5
            has_prediction = pd.notna(df.at[row_idx, "Predicted_Yield"])
            has_rec = pd.notna(df.at[row_idx, "Recommended_Fertilizer"])
            has_dose = pd.notna(df.at[row_idx, "Recommended_Dose"])
            has_interval = pd.notna(df.at[row_idx, "Recommended_Application_Interval"])

            if has_prediction:
                score += 0.2
            if has_rec:
                score += 0.1
            if has_dose:
                score += 0.1
            if has_interval:
                score += 0.1
            if has_prediction and has_rec and has_dose and has_interval:
                score = 0.85

            df.at[row_idx, "Confidence_Score"] = round(min(score, 1.0), 3)

        return df

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        self._init_col(df, "Recommendation_Summary", object)

        for row_idx in df.index:
            parts = []
            conditions = self._describe_conditions(df, row_idx)
            if conditions:
                parts.append("Current: " + "; ".join(conditions) + ".")

            fert = df.at[row_idx, "Recommended_Fertilizer"]
            dose = df.at[row_idx, "Recommended_Dose"]
            interval = df.at[row_idx, "Recommended_Application_Interval"]

            rec_parts = []
            if pd.notna(fert):
                rec_parts.append(str(fert))
            if pd.notna(dose):
                rec_parts.append(f"{dose:.1f} units")
            if pd.notna(interval):
                rec_parts.append(f"every {interval:.0f} days")
            if rec_parts:
                parts.append("Apply " + " ".join(rec_parts) + ".")

            increase = df.at[row_idx, "Expected_Yield_Increase"]
            if pd.notna(increase):
                symbol = "+" if increase >= 0 else ""
                parts.append(f"Expected yield {symbol}{increase:.1f} vs baseline.")

            conf = df.at[row_idx, "Confidence_Score"]
            if pd.notna(conf):
                pct = conf * 100
                label = "High" if pct >= 70 else "Moderate" if pct >= 40 else "Low"
                parts.append(f"{label} confidence ({pct:.0f}%).")

            df.at[row_idx, "Recommendation_Summary"] = " ".join(parts)

        return df

    def _describe_conditions(self, df: pd.DataFrame, row_idx: int) -> list[str]:
        desc = []
        checks = [
            ("Nitrogen", "ppm", 50), ("Phosphorus", "ppm", 15),
            ("Potassium", "ppm", 100), ("Rainfall", "mm", 500),
        ]
        for col, unit, low_thresh in checks:
            val = df.at[row_idx, col] if col in df.columns else None
            if pd.notna(val):
                if val < low_thresh:
                    desc.append(f"{col} {val:.0f}{unit} (low)")
                else:
                    desc.append(f"{col} {val:.0f}{unit}")

        ph = df.at[row_idx, "Soil_pH"] if "Soil_pH" in df.columns else None
        if pd.notna(ph):
            desc.append(f"Soil pH {ph:.1f}")
            if ph < 5.5:
                desc[-1] += " (acidic)"
            elif ph > 7.5:
                desc[-1] += " (alkaline)"

        return desc

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "recommendation_complete": True,
        }
