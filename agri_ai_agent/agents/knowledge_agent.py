"""
Knowledge Agent
Central domain knowledge provider for crop, soil, weather, and fertilizer
information. Other agents query this agent for thresholds, optimal ranges,
and domain-specific lookups.
"""

from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent


CROP_PH_RANGES = {
    "Rice": (5.0, 6.5), "Wheat": (6.0, 7.5), "Maize": (5.5, 7.0),
    "Sugarcane": (5.5, 7.5), "Cotton": (5.5, 7.0), "Groundnut": (5.5, 7.0),
    "Potato": (5.0, 6.5), "Tomato": (5.5, 7.0), "Onion": (6.0, 7.0),
    "Chilli": (5.5, 7.0), "Brinjal": (5.5, 7.0), "Cabbage": (6.0, 7.0),
    "Cauliflower": (6.0, 7.0), "Okra": (6.0, 7.5), "Pea": (6.0, 7.5),
    "Bean": (6.0, 7.0), "Soybean": (6.0, 7.0), "Sunflower": (6.0, 7.5),
    "Mustard": (5.5, 7.0), "Sorghum": (5.5, 7.5), "Pearl Millet": (5.5, 7.5),
    "Finger Millet": (5.0, 6.5), "Barley": (6.0, 7.5), "Oat": (5.5, 7.0),
}

FERTILIZER_NPK = {
    "Urea": (46, 0, 0), "DAP": (18, 46, 0), "MOP": (0, 0, 60),
    "SOP": (0, 0, 50), "SSP": (0, 16, 0), "CAN": (26, 0, 0),
    "UAN": (32, 0, 0), "10-26-26": (10, 26, 26), "12-32-16": (12, 32, 16),
    "20-20-0": (20, 20, 0), "15-15-15": (15, 15, 15),
    "Bio NPK": (5, 5, 5), "Jaivik Khad": (3, 2, 2),
    "Pori Potash": (0, 0, 15), "Ammonium Sulphate": (21, 0, 0),
    "Potassium Sulphate": (0, 0, 50),
}

SOIL_TEXTURE_PROPS = {
    "Sandy": {"whc": 0.6, "infiltration": "high", "om": "low", "aeration": "high"},
    "Loamy": {"whc": 1.0, "infiltration": "moderate", "om": "medium", "aeration": "moderate"},
    "Clay": {"whc": 1.4, "infiltration": "low", "om": "high", "aeration": "low"},
    "Silt": {"whc": 1.1, "infiltration": "moderate", "om": "medium", "aeration": "moderate"},
    "Sandy Loam": {"whc": 0.8, "infiltration": "moderate", "om": "medium", "aeration": "high"},
    "Clay Loam": {"whc": 1.2, "infiltration": "low", "om": "high", "aeration": "low"},
}

NUTRIENT_THRESHOLDS = {
    "Nitrogen": {"low": 50, "medium": 100, "high": 150},
    "Phosphorus": {"low": 15, "medium": 30, "high": 50},
    "Potassium": {"low": 100, "medium": 200, "high": 300},
    "Organic_Carbon": {"low": 0.4, "medium": 0.75, "high": 1.0},
}

WEATHER_THRESHOLDS = {
    "Rainfall": {"low": 500, "medium": 1000, "high": 1500},
    "Temperature_Max": {"cool": 25, "moderate": 32, "hot": 38},
}

GROWTH_STAGES = {
    "seedling": (0, 30), "vegetative": (20, 65),
    "flowering": (50, 85), "maturity": (75, 120),
}


def npk_label(n_pct: float, p_pct: float, k_pct: float) -> str:
    if n_pct >= 20 and p_pct >= 20 and k_pct >= 20:
        return "balanced"
    if n_pct >= 20:
        return "nitrogen_rich"
    if p_pct >= 20:
        return "phosphorus_rich"
    if k_pct >= 20:
        return "potassium_rich"
    return "organic"


class KnowledgeAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "KnowledgeAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()

        if "Crop" in df.columns:
            df["Optimal_pH_Min"] = df["Crop"].map(
                lambda c: CROP_PH_RANGES.get(str(c), (5.5, 7.0))[0]
            )
            df["Optimal_pH_Max"] = df["Crop"].map(
                lambda c: CROP_PH_RANGES.get(str(c), (5.5, 7.0))[1]
            )

        if "Nitrogen" in df.columns:
            df["N_Status"] = df["Nitrogen"].apply(
                lambda v: self._nutrient_status(v, "Nitrogen")
            )
        if "Phosphorus" in df.columns:
            df["P_Status"] = df["Phosphorus"].apply(
                lambda v: self._nutrient_status(v, "Phosphorus")
            )
        if "Potassium" in df.columns:
            df["K_Status"] = df["Potassium"].apply(
                lambda v: self._nutrient_status(v, "Potassium")
            )

        if "Fertilizer_Name" in df.columns:
            df["Fertilizer_NPK"] = df["Fertilizer_Name"].apply(
                lambda f: self._fertilizer_npk(str(f)) if pd.notna(f) else "0-0-0"
            )
            df["Fertilizer_Type"] = df["Fertilizer_Name"].apply(
                lambda f: self._fertilizer_type(str(f)) if pd.notna(f) else "unknown"
            )

        self.log.info("Knowledge annotations applied to %d rows", len(df))
        self.dataframe = df
        return df

    def _nutrient_status(self, value: float, nutrient: str) -> str:
        if pd.isna(value):
            return "unknown"
        thresholds = NUTRIENT_THRESHOLDS.get(nutrient, {})
        if value <= thresholds.get("low", 0):
            return "low"
        if value <= thresholds.get("medium", 50):
            return "medium"
        return "high"

    def _fertilizer_npk(self, name: str) -> str:
        npk = FERTILIZER_NPK.get(name)
        if npk:
            return f"{npk[0]}-{npk[1]}-{npk[2]}"
        return "0-0-0"

    def _fertilizer_type(self, name: str) -> str:
        npk = FERTILIZER_NPK.get(name)
        if npk:
            return npk_label(*npk)
        return "unknown"

    @staticmethod
    def nutrient_status(value: float, nutrient: str) -> str:
        if pd.isna(value):
            return "unknown"
        thresholds = NUTRIENT_THRESHOLDS.get(nutrient, {})
        if value <= thresholds.get("low", 0):
            return "low"
        if value <= thresholds.get("medium", 50):
            return "medium"
        return "high"

    @staticmethod
    def optimal_ph(crop: str) -> tuple[float, float]:
        return CROP_PH_RANGES.get(crop, (5.5, 7.0))

    @staticmethod
    def fertilizer_npk(name: str) -> tuple[int, int, int]:
        return FERTILIZER_NPK.get(name, (0, 0, 0))

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "annotations_added": ["Optimal_pH_Min", "Optimal_pH_Max",
                                  "N_Status", "P_Status", "K_Status",
                                  "Fertilizer_NPK", "Fertilizer_Type"],
        }
