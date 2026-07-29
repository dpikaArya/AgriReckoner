import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("CropEnvironmentInteraction")


class CropEnvironmentInteraction:
    def __init__(self):
        self.generated_features_: list[str] = []

    def compute_all(
        self, df: pd.DataFrame, target: Optional[str] = None
    ) -> pd.DataFrame:
        result = df.copy()
        result = self.crop_duration_temperature(result)
        result = self.rainfall_critical_growth(result)
        result = self.soil_climate_interaction(result)
        result = self.nutrient_yield_response(result, target)
        result = self.crop_water_productivity(result)
        return result

    def crop_duration_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])
        duration = self._find_col(df, [
            "Crop_Duration_days", "Duration_days", "Growing_Period", "Crop_Duration"
        ])
        gdd = self._find_col(df, ["GDD_Base10", "Growing_Degree_Days"])

        if t_avg is not None and duration is not None:
            df["Crop_Duration_x_Temp"] = df[duration] * df[t_avg]
            self.generated_features_.append("Crop_Duration_x_Temp")

        if gdd is not None and duration is not None:
            df["Thermal_Time_Efficiency"] = df[gdd] / df[duration].clip(lower=1)
            self.generated_features_.append("Thermal_Time_Efficiency")

        if t_avg is not None:
            df["Temp_Stress_Accumulated"] = (df[t_avg] - 35).clip(lower=0).cumsum() if len(df) > 1 else (df[t_avg] - 35).clip(lower=0)
            self.generated_features_.append("Temp_Stress_Accumulated")

        return df

    def rainfall_critical_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        rainfall = self._find_col(df, [
            "Rainfall_mm", "Rainfall", "Precipitation", "Annual_Rainfall"
        ])
        if rainfall is None:
            return df

        df["Rainfall_Critical_Stage"] = df[rainfall] * 0.6
        self.generated_features_.append("Rainfall_Critical_Stage")

        df["Rainfall_Flowering_Stage"] = df[rainfall] * 0.3
        self.generated_features_.append("Rainfall_Flowering_Stage")

        df["Rainfall_Ripening_Stage"] = df[rainfall] * 0.1
        self.generated_features_.append("Rainfall_Ripening_Stage")

        rain_mean = df[rainfall].mean()
        rain_std = df[rainfall].std()
        if rain_std > 0:
            df["Rainfall_Anomaly_Index"] = ((df[rainfall] - rain_mean) / rain_std).abs()
            self.generated_features_.append("Rainfall_Anomaly_Index")

        return df

    def soil_climate_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        fertility = self._find_col(df, [
            "Soil_Fertility_Index", "Fertility_Index"
        ])
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])
        rainfall = self._find_col(df, [
            "Rainfall_mm", "Rainfall", "Precipitation"
        ])

        if fertility is not None and t_avg is not None:
            df["Soil_Climate_Interaction"] = df[fertility] * (
                df[t_avg] / df[t_avg].max()
            ).clip(0, 1)
            self.generated_features_.append("Soil_Climate_Interaction")

        if fertility is not None and rainfall is not None:
            df["Soil_Rainfall_Interaction"] = df[fertility] * (
                df[rainfall] / df[rainfall].max()
            ).clip(0, 1)
            self.generated_features_.append("Soil_Rainfall_Interaction")

        if t_avg is not None and rainfall is not None:
            df["Climate_Moisture_Index"] = (df[rainfall] / (df[t_avg] + 1)).clip(0, 100)
            self.generated_features_.append("Climate_Moisture_Index")

        return df

    def nutrient_yield_response(
        self, df: pd.DataFrame, target: Optional[str] = None
    ) -> pd.DataFrame:
        n = self._find_col(df, ["Nitrogen_kg_ha", "N", "Total_Nitrogen"])
        p = self._find_col(df, ["Phosphorus_kg_ha", "P", "Available_Phosphorus"])
        k = self._find_col(df, ["Potassium_kg_ha", "K", "Available_Potassium"])

        if target is not None and n is not None and target in df.columns:
            df["N_Use_Efficiency_Est"] = df[target] / (df[n] + 1e-6)
            self.generated_features_.append("N_Use_Efficiency_Est")

        if n is not None and p is not None and k is not None:
            df["NPK_Interaction"] = df[n] * df[p] * df[k]
            self.generated_features_.append("NPK_Interaction")

        if n is not None and p is not None:
            df["N_P_Synergy"] = df[n] * df[p] / (df[n] + df[p] + 1e-6)
            self.generated_features_.append("N_P_Synergy")

        return df

    def crop_water_productivity(self, df: pd.DataFrame) -> pd.DataFrame:
        rainfall = self._find_col(df, [
            "Rainfall_mm", "Rainfall", "Precipitation", "Annual_Rainfall"
        ])
        yield_col = self._find_col(df, ["Yield_per_Hectare", "Yield", "Grain_Yield"])

        if rainfall is not None and yield_col is not None:
            df["Water_Use_Efficiency_Est"] = df[yield_col] / (df[rainfall] + 1e-6)
            self.generated_features_.append("Water_Use_Efficiency_Est")

        if rainfall is not None:
            df["Rainfall_Productivity"] = 1 / (df[rainfall] + 1e-6)
            self.generated_features_.append("Rainfall_Productivity")

        return df

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None
