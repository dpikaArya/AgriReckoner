"""
Schema Population Agent
Infers missing schema fields from available data using domain rules.
Only populates when confidence >95%.
Derives Yield_per_Hectare from Yield_per_Plot + Plot_size,
Organic_Matter from Organic_Carbon, and other safe derivations.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.config.settings import AgriAISettings


DERIVATION_RULES: list[dict[str, Any]] = [
    {
        "name": "yield_plot_to_hectare",
        "target": "Yield_per_Hectare",
        "sources": ["Yield_per_Plot", "Plot_Size"],
        "confidence": 0.98,
        "method": lambda df: df["Yield_per_Plot"] * (10000 / df["Plot_Size"].replace(0, np.nan)),
        "description": "Yield_per_Hectare = Yield_per_Plot * (10000 / Plot_Size_m2)",
    },
    {
        "name": "yield_acre_to_hectare",
        "target": "Yield_per_Hectare",
        "sources": ["Yield_per_Acre"],
        "confidence": 0.99,
        "method": lambda df: df["Yield_per_Acre"] * 2.471,
        "description": "Yield_per_Hectare = Yield_per_Acre * 2.471",
    },
    {
        "name": "yield_plot_to_acre",
        "target": "Yield_per_Acre",
        "sources": ["Yield_per_Plot", "Plot_Size"],
        "confidence": 0.98,
        "method": lambda df: df["Yield_per_Plot"] * (4046.86 / df["Plot_Size"].replace(0, np.nan)),
        "description": "Yield_per_Acre = Yield_per_Plot * (4046.86 / Plot_Size_m2)",
    },
    {
        "name": "organic_matter_from_carbon",
        "target": "Organic_Matter",
        "sources": ["Organic_Carbon"],
        "confidence": 0.97,
        "method": lambda df: df["Organic_Carbon"] * 1.724,
        "description": "Organic_Matter = Organic_Carbon * 1.724 (Van Bemmelen factor)",
    },
    {
        "name": "organic_carbon_from_matter",
        "target": "Organic_Carbon",
        "sources": ["Organic_Matter"],
        "confidence": 0.97,
        "method": lambda df: df["Organic_Matter"] / 1.724,
        "description": "Organic_Carbon = Organic_Matter / 1.724",
    },
    {
        "name": "plant_height_from_30_60",
        "target": "Plant_Height_cm",
        "sources": ["Plant_Height_30_cm", "Plant_Height_60_cm"],
        "confidence": 0.95,
        "method": lambda df: (df["Plant_Height_30_cm"] + df["Plant_Height_60_cm"]) / 2,
        "description": "Plant_Height_cm = average of 30d and 60d measurements",
    },
    {
        "name": "biomass_from_shoot_root",
        "target": "Biomass_Yield",
        "sources": ["Shoot_Biomass_g", "Root_Biomass_g"],
        "confidence": 0.99,
        "method": lambda df: df["Shoot_Biomass_g"] + df["Root_Biomass_g"],
        "description": "Biomass_Yield = Shoot_Biomass + Root_Biomass",
    },
    {
        "name": "leaf_area_index",
        "target": "Leaf_Area_cm2",
        "sources": ["Leaf_Area_30_cm2", "Leaf_Area_60_cm2"],
        "confidence": 0.95,
        "method": lambda df: df[[c for c in ["Leaf_Area_30_cm2", "Leaf_Area_60_cm2"] if c in df.columns]].max(axis=1),
        "description": "Leaf_Area_cm2 = max of available leaf area measurements",
    },
    {
        "name": "growing_degree_days",
        "target": "Growing_Degree_Days",
        "sources": ["Average_Temperature", "Growth_Duration_Days"],
        "confidence": 0.96,
        "method": lambda df: (df["Average_Temperature"] - 10).clip(lower=0) * df["Growth_Duration_Days"],
        "description": "GDD = (T_avg - T_base) * Duration, base=10C",
    },
    {
        "name": "yield_per_plant_from_plot",
        "target": "Yield_per_Plant",
        "sources": ["Yield_per_Plot", "Plot_Size"],
        "confidence": 0.97,
        "method": lambda df: df["Yield_per_Plot"] * df["Plot_Size"],
        "description": "Yield_per_Plant = Yield_per_Plot * Plot_Size (m2)",
    },
    {
        "name": "temp_x_rainfall_interaction",
        "target": "Temp_x_Rainfall",
        "sources": ["Average_Temperature", "Rainfall"],
        "confidence": 0.99,
        "method": lambda df: df["Average_Temperature"] * df["Rainfall"],
        "description": "Temp_x_Rainfall = T_avg * Rainfall",
    },
    {
        "name": "nxp_interaction",
        "target": "N_x_P",
        "sources": ["Nitrogen", "Phosphorus"],
        "confidence": 0.99,
        "method": lambda df: df["Nitrogen"] * df["Phosphorus"],
        "description": "N_x_P = N * P interaction",
    },
]


class SchemaPopulationAgent(BaseAgent):
    def __init__(self, settings: Optional[AgriAISettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.population_report: dict[str, Any] = {}
        self.min_confidence = 0.95

    @property
    def agent_name(self) -> str:
        return "SchemaPopulationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        self.min_confidence = kwargs.get("min_confidence", self.min_confidence)
        df = df.copy()

        derivations_applied: list[dict[str, Any]] = []

        for rule in DERIVATION_RULES:
            if rule["confidence"] < self.min_confidence:
                continue

            target = rule["target"]
            sources = rule["sources"]

            if target in df.columns and df[target].notna().any():
                continue

            missing_sources = [s for s in sources if s not in df.columns]
            if missing_sources:
                continue

            available_rows = df[sources].notna().all(axis=1)
            if not available_rows.any():
                continue

            try:
                derived = rule["method"](df)
                df.loc[available_rows, target] = derived[available_rows]
                n_filled = int(available_rows.sum())

                derivations_applied.append({
                    "rule": rule["name"],
                    "target": target,
                    "sources": sources,
                    "rows_filled": n_filled,
                    "confidence": rule["confidence"],
                    "description": rule["description"],
                })

                self.log.info(
                    "[%s] Applied %s: filled %d rows for %s (conf=%.2f)",
                    self.agent_name, rule["name"], n_filled, target, rule["confidence"],
                )
            except Exception as e:
                self.log.warning(
                    "[%s] Rule %s failed: %s", self.agent_name, rule["name"], e,
                )

        self.population_report = {
            "rules_evaluated": len(DERIVATION_RULES),
            "rules_applied": len(derivations_applied),
            "min_confidence_threshold": self.min_confidence,
            "derivations": derivations_applied,
        }

        return df

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        base = super()._build_output(df, **kwargs)
        base["population_report"] = self.population_report
        if df is not None and not df.empty:
            base["total_columns"] = len(df.columns)
            base["columns_with_data"] = int(df.notna().any().sum())
        return base
