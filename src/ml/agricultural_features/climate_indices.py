import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ClimateIndices")


class ClimateIndices:
    def __init__(self):
        self.generated_features_: list[str] = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result = self.growing_degree_days(result)
        result = self.heat_stress_index(result)
        result = self.rainfall_deficit_index(result)
        result = self.moisture_stress_index(result)
        result = self.temperature_suitability_index(result)
        return result

    def growing_degree_days(self, df: pd.DataFrame) -> pd.DataFrame:
        t_max = self._find_col(df, ["Temperature_Max", "Temp_max", "Max_Temp"])
        t_min = self._find_col(df, ["Temperature_Min", "Temp_min", "Min_Temp"])
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])
        base_temp = 10.0

        if t_avg is not None:
            df["GDD_Base10"] = (df[t_avg] - base_temp).clip(lower=0)
            self.generated_features_.append("GDD_Base10")
        elif t_max is not None and t_min is not None:
            t_avg_calc = (df[t_max] + df[t_min]) / 2
            df["GDD_Base10"] = (t_avg_calc - base_temp).clip(lower=0)
            self.generated_features_.append("GDD_Base10")

        if t_avg is not None:
            for base in [5, 8, 12]:
                name = f"GDD_Base{base}"
                df[name] = (df[t_avg] - base).clip(lower=0)
                self.generated_features_.append(name)

        return df

    def heat_stress_index(self, df: pd.DataFrame) -> pd.DataFrame:
        t_max = self._find_col(df, ["Temperature_Max", "Temp_max", "Max_Temp"])
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])

        if t_max is not None:
            df["Heat_Stress_Index"] = ((df[t_max] - 30) / 15).clip(0, 1)
            self.generated_features_.append("Heat_Stress_Index")
            df["Heat_Stress_Binary"] = (df[t_max] > 35).astype(float)
            self.generated_features_.append("Heat_Stress_Binary")

        if t_avg is not None:
            df["Heat_Stress_Moderate"] = ((df[t_avg] - 25) / 20).clip(0, 1)
            self.generated_features_.append("Heat_Stress_Moderate")

        return df

    def rainfall_deficit_index(self, df: pd.DataFrame) -> pd.DataFrame:
        rainfall = self._find_col(df, [
            "Rainfall_mm", "Rainfall", "Precipitation", "Rain", "Annual_Rainfall"
        ])
        if rainfall is None:
            return df

        df["Rainfall_Deficit_Index"] = 1 - (df[rainfall] / df[rainfall].max()).clip(0, 1)
        self.generated_features_.append("Rainfall_Deficit_Index")

        drought_threshold = df[rainfall].quantile(0.33)
        df["Drought_Risk"] = (df[rainfall] < drought_threshold).astype(float)
        self.generated_features_.append("Drought_Risk")

        excess_threshold = df[rainfall].quantile(0.90)
        df["Excess_Rainfall_Risk"] = (df[rainfall] > excess_threshold).astype(float)
        self.generated_features_.append("Excess_Rainfall_Risk")

        return df

    def moisture_stress_index(self, df: pd.DataFrame) -> pd.DataFrame:
        rainfall = self._find_col(df, ["Rainfall_mm", "Rainfall", "Precipitation"])
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])
        if rainfall is not None and t_avg is not None:
            pet = 0.0023 * 17.8 * (df[t_avg] + 17.8) * (df[rainfall] + 1).pow(0.5)
            df["Moisture_Stress_Index"] = (1 - (df[rainfall] / (pet + 1)).clip(0, 1)).clip(0, 1)
            self.generated_features_.append("Moisture_Stress_Index")
        return df

    def temperature_suitability_index(self, df: pd.DataFrame) -> pd.DataFrame:
        t_avg = self._find_col(df, ["Average_Temperature", "Temp_avg", "Avg_Temp"])
        if t_avg is None:
            return df

        t_opt = 25.0
        t_range = 15.0
        df["Temp_Suitability_Index"] = np.exp(
            -(((df[t_avg] - t_opt) / t_range) ** 2)
        )
        self.generated_features_.append("Temp_Suitability_Index")

        for crop, (t_min, t_max, t_opt_c) in [
            ("Wheat", (10, 30, 20)),
            ("Rice", (20, 38, 30)),
            ("Maize", (15, 35, 25)),
            ("Cotton", (20, 40, 30)),
            ("Potato", (10, 25, 18)),
        ]:
            name = f"Temp_Suitability_{crop}"
            df[name] = np.exp(-(((df[t_avg] - t_opt_c) / ((t_max - t_min) / 2)) ** 2))
            self.generated_features_.append(name)

        return df

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None
