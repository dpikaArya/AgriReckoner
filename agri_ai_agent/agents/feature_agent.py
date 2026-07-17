"""
Consolidated Feature Agent: Feature Engineering + Leakage Detection
+ Categorical Encoding + Statistical Diagnostics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.config.schema import (
    POST_HARVEST_VARIABLES, PRE_HARVEST_MEASUREMENTS, NON_FEATURE_COLS,
)

CATEGORICAL_ENCODING_MAP = {
    "Crop": "Crop_Code",
    "Season": "Season_Code",
    "Variety": "Variety_Code",
    "Fertilizer_Name": "Fertilizer_Code",
    "Country": "Country_Code",
}

EXTRA_CATEGORICALS = {
    "Soil_Texture": "Soil_Texture_Code",
    "Soil_Type": "Soil_Texture_Code",
}


class FeatureAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "FeatureAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        self._engineer_features(df)
        self._detect_leakage(df)
        self._encode_categoricals(df)
        self._run_diagnostics(df)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> None:
        features_added = []
        tbase = 10.0

        if "Temperature_Max" in df.columns and "Temperature_Min" in df.columns:
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
            tmean = (tmax + tmin) / 2.0

            if "Growing_Degree_Days" not in df.columns:
                df["Growing_Degree_Days"] = np.maximum(0, tmean - tbase)
                features_added.append("Growing_Degree_Days")

            if "Heat_Units" not in df.columns:
                df["Heat_Units"] = np.maximum(0, tmax - tbase)
                features_added.append("Heat_Units")

            if "Temp_squared" not in df.columns:
                df["Temp_squared"] = tmean ** 2
                features_added.append("Temp_squared")

            optimal_temp = 25.0
            if "Stress_Index" not in df.columns:
                df["Stress_Index"] = np.abs(tmean - optimal_temp) / optimal_temp
                features_added.append("Stress_Index")

            if "Temp_Range" not in df.columns:
                df["Temp_Range"] = tmax - tmin
                features_added.append("Temp_Range")

            if "Temp_CV" not in df.columns:
                df["Temp_CV"] = np.where(tmean > 0, (tmax - tmin) / tmean, 0)
                features_added.append("Temp_CV")

            for t in [10, 15, 20, 25, 30, 35]:
                name = f"Temp_Above_{t}"
                if name not in df.columns:
                    df[name] = np.maximum(0, tmean - t)
                    features_added.append(name)

        if "Rainfall" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Rainfall_Anomaly" not in df.columns:
                df["Rainfall_Anomaly"] = rainfall - rainfall.mean()
                features_added.append("Rainfall_Anomaly")

            if "Rainfall_log" not in df.columns:
                df["Rainfall_log"] = np.log1p(rainfall.clip(lower=0))
                features_added.append("Rainfall_log")

            if "Rainfall_sqrt" not in df.columns:
                df["Rainfall_sqrt"] = np.sqrt(rainfall.clip(lower=0))
                features_added.append("Rainfall_sqrt")

            if "Rainfall_squared" not in df.columns:
                df["Rainfall_squared"] = rainfall ** 2
                features_added.append("Rainfall_squared")

            if "Rainfall_Bin" not in df.columns:
                df["Rainfall_Bin"] = pd.cut(rainfall, bins=[0, 200, 500, 1000, 5000, 10000],
                                            labels=[0, 1, 2, 3, 4]).astype(float)
                features_added.append("Rainfall_Bin")

        if "Rainfall" in df.columns and "Temperature_Max" in df.columns:
            tmean_series = (pd.to_numeric(df.get("Temperature_Max", 0), errors="coerce") +
                           pd.to_numeric(df.get("Temperature_Min", 0), errors="coerce")) / 2.0
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Temp_x_Rainfall" not in df.columns:
                df["Temp_x_Rainfall"] = tmean_series * rainfall
                features_added.append("Temp_x_Rainfall")

        if "Nitrogen" in df.columns and "Phosphorus" in df.columns:
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "N_x_P" not in df.columns:
                df["N_x_P"] = n * p
                features_added.append("N_x_P")

        if "Nitrogen" in df.columns:
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "N_log" not in df.columns:
                df["N_log"] = np.log1p(n.clip(lower=0))
                features_added.append("N_log")
            if "N_sqrt" not in df.columns:
                df["N_sqrt"] = np.sqrt(n.clip(lower=0))
                features_added.append("N_sqrt")

        if "Phosphorus" in df.columns:
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "P_log" not in df.columns:
                df["P_log"] = np.log1p(p.clip(lower=0))
                features_added.append("P_log")

        if "Potassium" in df.columns:
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "K_log" not in df.columns:
                df["K_log"] = np.log1p(k.clip(lower=0))
                features_added.append("K_log")

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Potassium"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "NPK_sum" not in df.columns:
                df["NPK_sum"] = n + p + k
                features_added.append("NPK_sum")
            if "NPK_ratio_N" not in df.columns:
                total = (n + p + k).replace(0, np.nan)
                df["NPK_ratio_N"] = n / total
                features_added.append("NPK_ratio_N")
            if "NPK_ratio_P" not in df.columns:
                df["NPK_ratio_P"] = p / total
                features_added.append("NPK_ratio_P")
            if "NPK_ratio_K" not in df.columns:
                df["NPK_ratio_K"] = k / total
                features_added.append("NPK_ratio_K")
            if "N_x_K" not in df.columns:
                df["N_x_K"] = n * k
                features_added.append("N_x_K")
            if "P_x_K" not in df.columns:
                df["P_x_K"] = p * k
                features_added.append("P_x_K")
            if "N_plus_P" not in df.columns:
                df["N_plus_P"] = n + p
                features_added.append("N_plus_P")
            if "N_plus_K" not in df.columns:
                df["N_plus_K"] = n + k
                features_added.append("N_plus_K")
            if "P_plus_K" not in df.columns:
                df["P_plus_K"] = p + k
                features_added.append("P_plus_K")

        if "Soil_pH" in df.columns:
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "Soil_pH_squared" not in df.columns:
                df["Soil_pH_squared"] = ph ** 2
                features_added.append("Soil_pH_squared")
            if "Soil_pH_neutral" not in df.columns:
                df["Soil_pH_neutral"] = np.abs(ph - 7.0)
                features_added.append("Soil_pH_neutral")
            if "Soil_pH_acidic" not in df.columns:
                df["Soil_pH_acidic"] = np.where(ph < 6.5, 1, 0)
                features_added.append("Soil_pH_acidic")
            if "Soil_pH_alkaline" not in df.columns:
                df["Soil_pH_alkaline"] = np.where(ph > 7.5, 1, 0)
                features_added.append("Soil_pH_alkaline")

        if "Organic_Carbon" in df.columns:
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "OC_log" not in df.columns:
                df["OC_log"] = np.log1p(oc.clip(lower=0))
                features_added.append("OC_log")

        if all(c in df.columns for c in ["Organic_Carbon", "Nitrogen"]):
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "C_N_ratio" not in df.columns:
                df["C_N_ratio"] = oc / n.replace(0, np.nan)
                features_added.append("C_N_ratio")

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "N_P_ratio" not in df.columns:
                df["N_P_ratio"] = n / p.replace(0, np.nan)
                features_added.append("N_P_ratio")

        if all(c in df.columns for c in ["Phosphorus", "Potassium"]):
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "P_K_ratio" not in df.columns:
                df["P_K_ratio"] = p / k.replace(0, np.nan)
                features_added.append("P_K_ratio")

        if "Biomass_Yield" in df.columns and "Yield_per_Hectare" in df.columns:
            biomass = pd.to_numeric(df["Biomass_Yield"], errors="coerce")
            yield_ha = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                if "Harvest_Index_Calc" not in df.columns:
                    df["Harvest_Index_Calc"] = np.where(biomass > 0, yield_ha / biomass, np.nan)
                    features_added.append("Harvest_Index_Calc")
                if "Biomass_log" not in df.columns:
                    df["Biomass_log"] = np.log1p(biomass.clip(lower=0))
                    features_added.append("Biomass_log")

        if "Nitrogen" in df.columns and "Yield_per_Hectare" in df.columns:
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            yield_ha = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                if "Nitrogen_Use_Efficiency" not in df.columns:
                    df["Nitrogen_Use_Efficiency"] = np.where(n > 0, yield_ha / n, np.nan)
                    features_added.append("Nitrogen_Use_Efficiency")

        if "Rainfall" in df.columns and "Yield_per_Hectare" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            yield_ha = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                if "Water_Use_Efficiency" not in df.columns:
                    df["Water_Use_Efficiency"] = np.where(rainfall > 0, yield_ha / rainfall, np.nan)
                    features_added.append("Water_Use_Efficiency")

        if "Humidity" in df.columns and "Temperature_Max" in df.columns:
            hum = pd.to_numeric(df["Humidity"], errors="coerce")
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            if "Disease_Risk_Index" not in df.columns:
                df["Disease_Risk_Index"] = np.where(
                    (hum > 80) & (tmax > 25), 1.0,
                    np.where((hum > 60) & (tmax > 20), 0.5, 0.0),
                )
                features_added.append("Disease_Risk_Index")

        if "Yield_per_Plot" in df.columns:
            if "Yield_per_Plot_Calc" not in df.columns:
                df["Yield_per_Plot_Calc"] = pd.to_numeric(df["Yield_per_Plot"], errors="coerce")
                features_added.append("Yield_per_Plot_Calc")

            if "Yield_per_Hectare" not in df.columns and "Plot_Size" in df.columns:
                plot_size = pd.to_numeric(df["Plot_Size"], errors="coerce")
                yield_plot = pd.to_numeric(df["Yield_per_Plot"], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    if "Yield_per_Hectare_Calc" not in df.columns:
                        df["Yield_per_Hectare_Calc"] = np.where(
                            plot_size > 0, yield_plot / plot_size * 10000, np.nan,
                        )
                        features_added.append("Yield_per_Hectare_Calc")

        if "Fruit_Number" in df.columns and "Fruit_Weight" in df.columns:
            fn = pd.to_numeric(df["Fruit_Number"], errors="coerce")
            fw = pd.to_numeric(df["Fruit_Weight"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                if "Yield_per_Plant" not in df.columns:
                    df["Yield_per_Plant"] = np.where(fn > 0, fn * fw, np.nan)
                    features_added.append("Yield_per_Plant")

        if "Plant_Height_cm" in df.columns:
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            if "Height_log" not in df.columns:
                df["Height_log"] = np.log1p(ph.clip(lower=0))
                features_added.append("Height_log")
            if "Height_sqrt" not in df.columns:
                df["Height_sqrt"] = np.sqrt(ph.clip(lower=0))
                features_added.append("Height_sqrt")

        if "SPAD" in df.columns:
            spad = pd.to_numeric(df["SPAD"], errors="coerce")
            if "SPAD_log" not in df.columns:
                df["SPAD_log"] = np.log1p(spad.clip(lower=0))
                features_added.append("SPAD_log")
            if "SPAD_category" not in df.columns:
                df["SPAD_category"] = pd.cut(spad, bins=[0, 20, 35, 50, 100],
                                             labels=[0, 1, 2, 3]).astype(float)
                features_added.append("SPAD_category")

        if all(c in df.columns for c in ["Plant_Height_cm", "Leaf_Area_cm2"]):
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            la = pd.to_numeric(df["Leaf_Area_cm2"], errors="coerce")
            if "Height_x_LeafArea" not in df.columns:
                df["Height_x_LeafArea"] = ph * la
                features_added.append("Height_x_LeafArea")

        if all(c in df.columns for c in ["Plant_Height_cm", "Shoot_Biomass_g"]):
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            sb = pd.to_numeric(df["Shoot_Biomass_g"], errors="coerce")
            if "Height_x_ShootBiomass" not in df.columns:
                df["Height_x_ShootBiomass"] = ph * sb
                features_added.append("Height_x_ShootBiomass")

        for col in ["Rainfall", "Temperature_Max", "Temperature_Min", "Average_Temperature"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                col_name = f"{col}_7d_MA"
                if col_name not in df.columns:
                    df[col_name] = vals.rolling(window=7, min_periods=1).mean()
                    features_added.append(col_name)

        if "Average_Temperature" in df.columns:
            tavg = pd.to_numeric(df["Average_Temperature"], errors="coerce")
            if "Temp_x_N" in df.columns:
                pass
            elif "Nitrogen" in df.columns:
                n = pd.to_numeric(df["Nitrogen"], errors="coerce")
                if "Temp_x_N" not in df.columns:
                    df["Temp_x_N"] = tavg * n
                    features_added.append("Temp_x_N")

        if "Rainfall" in df.columns and "Nitrogen" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Rainfall_x_N" not in df.columns:
                df["Rainfall_x_N"] = rainfall * n
                features_added.append("Rainfall_x_N")

        if "Rainfall" in df.columns and "Phosphorus" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "Rainfall_x_P" not in df.columns:
                df["Rainfall_x_P"] = rainfall * p
                features_added.append("Rainfall_x_P")

        if "Rainfall" in df.columns and "Potassium" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "Rainfall_x_K" not in df.columns:
                df["Rainfall_x_K"] = rainfall * k
                features_added.append("Rainfall_x_K")

        if "Organic_Carbon" in df.columns and "Soil_pH" in df.columns:
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "OC_x_pH" not in df.columns:
                df["OC_x_pH"] = oc * ph
                features_added.append("OC_x_pH")

        if "EC" in df.columns and "Soil_pH" in df.columns:
            ec = pd.to_numeric(df["EC"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "EC_x_pH" not in df.columns:
                df["EC_x_pH"] = ec * ph
                features_added.append("EC_x_pH")

        for col in ["Yield_per_Hectare", "Plant_Height_cm", "Biomass_Yield", "SPAD",
                     "Shoot_Biomass_g", "Root_Biomass_g", "Leaf_Area_cm2",
                     "Fruit_Weight", "100_Seed_Weight", "Harvest_Index"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                log_name = f"{col}_log"
                if log_name not in df.columns and vals.min() >= 0:
                    df[log_name] = np.log1p(vals.clip(lower=0))
                    features_added.append(log_name)

        if "Yield_per_Hectare" in df.columns:
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            if "Yield_category" not in df.columns:
                df["Yield_category"] = pd.cut(yh, bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
                features_added.append("Yield_category")

        if "EC" in df.columns:
            ec = pd.to_numeric(df["EC"], errors="coerce")
            if "EC_log" not in df.columns:
                df["EC_log"] = np.log1p(ec.clip(lower=0))
                features_added.append("EC_log")
            if "EC_squared" not in df.columns:
                df["EC_squared"] = ec ** 2
                features_added.append("EC_squared")
            if "EC_category" not in df.columns:
                df["EC_category"] = pd.cut(ec, bins=[0, 0.5, 1.5, 4, 100],
                                           labels=[0, 1, 2, 3]).astype(float)
                features_added.append("EC_category")

        if "Rainfall" in df.columns and "Soil_pH" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "Rainfall_x_pH" not in df.columns:
                df["Rainfall_x_pH"] = rainfall * ph
                features_added.append("Rainfall_x_pH")

        if "Rainfall" in df.columns and "Organic_Carbon" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "Rainfall_x_OC" not in df.columns:
                df["Rainfall_x_OC"] = rainfall * oc
                features_added.append("Rainfall_x_OC")

        if all(c in df.columns for c in ["Nitrogen", "Soil_pH"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "N_x_pH" not in df.columns:
                df["N_x_pH"] = n * ph
                features_added.append("N_x_pH")

        if all(c in df.columns for c in ["Phosphorus", "Soil_pH"]):
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "P_x_pH" not in df.columns:
                df["P_x_pH"] = p * ph
                features_added.append("P_x_pH")

        if all(c in df.columns for c in ["Potassium", "Soil_pH"]):
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "K_x_pH" not in df.columns:
                df["K_x_pH"] = k * ph
                features_added.append("K_x_pH")

        if "Yield_per_Hectare" in df.columns and "Nitrogen" in df.columns:
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Yield_x_N" not in df.columns:
                df["Yield_x_N"] = yh * n
                features_added.append("Yield_x_N")

        if "Yield_per_Hectare" in df.columns and "Soil_pH" in df.columns:
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "Yield_x_pH" not in df.columns:
                df["Yield_x_pH"] = yh * ph
                features_added.append("Yield_x_pH")

        if "Plant_Height_cm" in df.columns and "Nitrogen" in df.columns:
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Height_x_N" not in df.columns:
                df["Height_x_N"] = ph * n
                features_added.append("Height_x_N")

        if "Plant_Height_cm" in df.columns and "Rainfall" in df.columns:
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Height_x_Rainfall" not in df.columns:
                df["Height_x_Rainfall"] = ph * rainfall
                features_added.append("Height_x_Rainfall")

        if "Plant_Height_cm" in df.columns and "Soil_pH" in df.columns:
            ph_col = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            soil_ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "Height_x_pH" not in df.columns:
                df["Height_x_pH"] = ph_col * soil_ph
                features_added.append("Height_x_pH")

        if all(c in df.columns for c in ["Temperature_Max", "Nitrogen"]):
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Tmax_x_N" not in df.columns:
                df["Tmax_x_N"] = tmax * n
                features_added.append("Tmax_x_N")

        if all(c in df.columns for c in ["Temperature_Max", "Rainfall"]):
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Tmax_x_Rainfall" not in df.columns:
                df["Tmax_x_Rainfall"] = tmax * rainfall
                features_added.append("Tmax_x_Rainfall")

        if all(c in df.columns for c in ["Temperature_Min", "Rainfall"]):
            tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Tmin_x_Rainfall" not in df.columns:
                df["Tmin_x_Rainfall"] = tmin * rainfall
                features_added.append("Tmin_x_Rainfall")

        if all(c in df.columns for c in ["Humidity", "Rainfall"]):
            hum = pd.to_numeric(df["Humidity"], errors="coerce")
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Humidity_x_Rainfall" not in df.columns:
                df["Humidity_x_Rainfall"] = hum * rainfall
                features_added.append("Humidity_x_Rainfall")

        if all(c in df.columns for c in ["Humidity", "Nitrogen"]):
            hum = pd.to_numeric(df["Humidity"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Humidity_x_N" not in df.columns:
                df["Humidity_x_N"] = hum * n
                features_added.append("Humidity_x_N")

        if all(c in df.columns for c in ["Yield_per_Hectare", "Biomass_Yield"]):
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            bm = pd.to_numeric(df["Biomass_Yield"], errors="coerce")
            if "Yield_x_Biomass" not in df.columns:
                df["Yield_x_Biomass"] = yh * bm
                features_added.append("Yield_x_Biomass")

        if "Humidity" in df.columns:
            hum = pd.to_numeric(df["Humidity"], errors="coerce")
            if "Humidity_squared" not in df.columns:
                df["Humidity_squared"] = hum ** 2
                features_added.append("Humidity_squared")
            if "Humidity_log" not in df.columns:
                df["Humidity_log"] = np.log1p(hum.clip(lower=0))
                features_added.append("Humidity_log")

        if "Organic_Carbon" in df.columns:
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "OC_squared" not in df.columns:
                df["OC_squared"] = oc ** 2
                features_added.append("OC_squared")

        if "EC" in df.columns and "Organic_Carbon" in df.columns:
            ec = pd.to_numeric(df["EC"], errors="coerce")
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "EC_x_OC" not in df.columns:
                df["EC_x_OC"] = ec * oc
                features_added.append("EC_x_OC")

        if all(c in df.columns for c in ["Nitrogen", "Organic_Carbon"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "N_x_OC" not in df.columns:
                df["N_x_OC"] = n * oc
                features_added.append("N_x_OC")

        if all(c in df.columns for c in ["Phosphorus", "Organic_Carbon"]):
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "P_x_OC" not in df.columns:
                df["P_x_OC"] = p * oc
                features_added.append("P_x_OC")

        if all(c in df.columns for c in ["Potassium", "Organic_Carbon"]):
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            oc = pd.to_numeric(df["Organic_Carbon"], errors="coerce")
            if "K_x_OC" not in df.columns:
                df["K_x_OC"] = k * oc
                features_added.append("K_x_OC")

        if "Leaf_Area_cm2" in df.columns:
            la = pd.to_numeric(df["Leaf_Area_cm2"], errors="coerce")
            if "LA_log" not in df.columns:
                df["LA_log"] = np.log1p(la.clip(lower=0))
                features_added.append("LA_log")
            if "LA_squared" not in df.columns:
                df["LA_squared"] = la ** 2
                features_added.append("LA_squared")

        if "Shoot_Biomass_g" in df.columns:
            sb = pd.to_numeric(df["Shoot_Biomass_g"], errors="coerce")
            if "ShootBiomass_log" not in df.columns:
                df["ShootBiomass_log"] = np.log1p(sb.clip(lower=0))
                features_added.append("ShootBiomass_log")

        if "Root_Biomass_g" in df.columns:
            rb = pd.to_numeric(df["Root_Biomass_g"], errors="coerce")
            if "RootBiomass_log" not in df.columns:
                df["RootBiomass_log"] = np.log1p(rb.clip(lower=0))
                features_added.append("RootBiomass_log")

        if all(c in df.columns for c in ["Shoot_Biomass_g", "Root_Biomass_g"]):
            sb = pd.to_numeric(df["Shoot_Biomass_g"], errors="coerce")
            rb = pd.to_numeric(df["Root_Biomass_g"], errors="coerce")
            total = sb + rb
            if "Shoot_Root_Ratio" not in df.columns:
                df["Shoot_Root_Ratio"] = sb / rb.replace(0, np.nan)
                features_added.append("Shoot_Root_Ratio")
            if "Root_Shoot_Ratio" not in df.columns:
                df["Root_Shoot_Ratio"] = rb / sb.replace(0, np.nan)
                features_added.append("Root_Shoot_Ratio")
            if "Root_pct" not in df.columns:
                df["Root_pct"] = rb / total.replace(0, np.nan)
                features_added.append("Root_pct")

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Soil_pH"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "N_x_P_x_pH" not in df.columns:
                df["N_x_P_x_pH"] = n * p * ph
                features_added.append("N_x_P_x_pH")

        if "Fruit_Weight" in df.columns:
            fw = pd.to_numeric(df["Fruit_Weight"], errors="coerce")
            if "FruitWeight_log" not in df.columns:
                df["FruitWeight_log"] = np.log1p(fw.clip(lower=0))
                features_added.append("FruitWeight_log")
            if "FruitWeight_squared" not in df.columns:
                df["FruitWeight_squared"] = fw ** 2
                features_added.append("FruitWeight_squared")

        if "100_Seed_Weight" in df.columns:
            sw = pd.to_numeric(df["100_Seed_Weight"], errors="coerce")
            if "SeedWeight_log" not in df.columns:
                df["SeedWeight_log"] = np.log1p(sw.clip(lower=0))
                features_added.append("SeedWeight_log")

        if "Harvest_Index" in df.columns:
            hi = pd.to_numeric(df["Harvest_Index"], errors="coerce")
            if "HI_squared" not in df.columns:
                df["HI_squared"] = hi ** 2
                features_added.append("HI_squared")
            if "HI_log" not in df.columns:
                df["HI_log"] = np.log1p(hi.clip(lower=0))
                features_added.append("HI_log")

        if "Rainfall" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Rainfall_100_bin" not in df.columns:
                df["Rainfall_100_bin"] = (rainfall / 100).round() * 100
                features_added.append("Rainfall_100_bin")

        if "Average_Temperature" in df.columns:
            tavg = pd.to_numeric(df["Average_Temperature"], errors="coerce")
            if "Tavg_squared" not in df.columns:
                df["Tavg_squared"] = tavg ** 2
                features_added.append("Tavg_squared")
            if "Tavg_log" not in df.columns:
                df["Tavg_log"] = np.log1p(tavg.clip(lower=0))
                features_added.append("Tavg_log")

        if "Nitrogen" in df.columns:
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "N_squared" not in df.columns:
                df["N_squared"] = n ** 2
                features_added.append("N_squared")
            if "N_category" not in df.columns:
                df["N_category"] = pd.cut(n, bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
                features_added.append("N_category")

        if "Phosphorus" in df.columns:
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "P_squared" not in df.columns:
                df["P_squared"] = p ** 2
                features_added.append("P_squared")
            if "P_category" not in df.columns:
                df["P_category"] = pd.cut(p, bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
                features_added.append("P_category")

        if "Potassium" in df.columns:
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "K_squared" not in df.columns:
                df["K_squared"] = k ** 2
                features_added.append("K_squared")
            if "K_category" not in df.columns:
                df["K_category"] = pd.cut(k, bins=5, labels=[0, 1, 2, 3, 4]).astype(float)
                features_added.append("K_category")

        if "Plant_Height_cm" in df.columns:
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            if "Height_squared" not in df.columns:
                df["Height_squared"] = ph ** 2
                features_added.append("Height_squared")

        if "SPAD" in df.columns:
            spad = pd.to_numeric(df["SPAD"], errors="coerce")
            if "SPAD_squared" not in df.columns:
                df["SPAD_squared"] = spad ** 2
                features_added.append("SPAD_squared")

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Potassium", "Soil_pH"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "NPK_x_pH" not in df.columns:
                df["NPK_x_pH"] = (n + p + k) * ph
                features_added.append("NPK_x_pH")

        if all(c in df.columns for c in ["Nitrogen", "Rainfall"]):
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "N_Rainfall_ratio" not in df.columns:
                df["N_Rainfall_ratio"] = n / rainfall.replace(0, np.nan)
                features_added.append("N_Rainfall_ratio")

        if all(c in df.columns for c in ["Yield_per_Hectare", "Plant_Height_cm"]):
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            if "Yield_per_Height" not in df.columns:
                df["Yield_per_Height"] = yh / ph.replace(0, np.nan)
                features_added.append("Yield_per_Height")

        if all(c in df.columns for c in ["Biomass_Yield", "Nitrogen"]):
            bm = pd.to_numeric(df["Biomass_Yield"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Biomass_x_N" not in df.columns:
                df["Biomass_x_N"] = bm * n
                features_added.append("Biomass_x_N")
            if "Biomass_N_ratio" not in df.columns:
                df["Biomass_N_ratio"] = bm / n.replace(0, np.nan)
                features_added.append("Biomass_N_ratio")

        if all(c in df.columns for c in ["Biomass_Yield", "Soil_pH"]):
            bm = pd.to_numeric(df["Biomass_Yield"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            if "Biomass_x_pH" not in df.columns:
                df["Biomass_x_pH"] = bm * ph
                features_added.append("Biomass_x_pH")

        if all(c in df.columns for c in ["SPAD", "Nitrogen"]):
            spad = pd.to_numeric(df["SPAD"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "SPAD_x_N" not in df.columns:
                df["SPAD_x_N"] = spad * n
                features_added.append("SPAD_x_N")

        if all(c in df.columns for c in ["Leaf_Area_cm2", "Nitrogen"]):
            la = pd.to_numeric(df["Leaf_Area_cm2"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "LA_x_N" not in df.columns:
                df["LA_x_N"] = la * n
                features_added.append("LA_x_N")

        if all(c in df.columns for c in ["Plant_Height_cm", "Phosphorus"]):
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "Height_x_P" not in df.columns:
                df["Height_x_P"] = ph * p
                features_added.append("Height_x_P")

        if all(c in df.columns for c in ["Plant_Height_cm", "Potassium"]):
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            k = pd.to_numeric(df["Potassium"], errors="coerce")
            if "Height_x_K" not in df.columns:
                df["Height_x_K"] = ph * k
                features_added.append("Height_x_K")

        if all(c in df.columns for c in ["Yield_per_Hectare", "Soil_pH", "Nitrogen"]):
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            ph = pd.to_numeric(df["Soil_pH"], errors="coerce")
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            if "Yield_pH_N_interaction" not in df.columns:
                df["Yield_pH_N_interaction"] = yh * ph * n
                features_added.append("Yield_pH_N_interaction")

        if all(c in df.columns for c in ["Plant_Height_cm", "SPAD"]):
            ph = pd.to_numeric(df["Plant_Height_cm"], errors="coerce")
            spad = pd.to_numeric(df["SPAD"], errors="coerce")
            if "Height_x_SPAD" not in df.columns:
                df["Height_x_SPAD"] = ph * spad
                features_added.append("Height_x_SPAD")

        self.log.info("Engineered %d features: %s", len(features_added), features_added)

    def _detect_leakage(self, df: pd.DataFrame) -> None:
        leaked = []
        safe = []
        non_feature = []
        feature_labels = []

        for col in df.columns:
            base_col = col
            for suffix in ["_Calc", "_7d_MA", "_squared"]:
                if col.endswith(suffix):
                    base_col = col[: -len(suffix)]
                    break
            if base_col in POST_HARVEST_VARIABLES:
                leaked.append(col)
                feature_labels.append((col, "POST_HARVEST"))
            elif col in NON_FEATURE_COLS:
                non_feature.append(col)
                feature_labels.append((col, "NON_FEATURE"))
            elif base_col in PRE_HARVEST_MEASUREMENTS:
                safe.append(col)
                feature_labels.append((col, "PRE_HARVEST_MEASUREMENT"))
            else:
                safe.append(col)
                feature_labels.append((col, "AVAILABLE_BEFORE_PREDICTION"))

        leak_df = pd.DataFrame(feature_labels, columns=["Feature", "Availability"])

        if "Feature_Available_Before_Prediction" not in df.columns:
            df["Feature_Available_Before_Prediction"] = leak_df["Availability"].map(
                lambda x: x == "AVAILABLE_BEFORE_PREDICTION"
            ).astype(str).str.upper()

        self.log.info("Leakage: %d leaked, %d non-feature, %d safe", len(leaked), len(non_feature), len(safe))

        if leaked:
            self.save_artifact(pd.DataFrame({"leaked_feature": leaked}), "Leakage_Report.csv")
        self.save_artifact(leak_df, "Feature_Availability_Check.csv")

    def _encode_categoricals(self, df: pd.DataFrame) -> None:
        encoding_records = []
        combined_map = {**CATEGORICAL_ENCODING_MAP, **EXTRA_CATEGORICALS}

        for src_col, tgt_col in combined_map.items():
            if src_col in df.columns and tgt_col not in df.columns:
                le = LabelEncoder()
                valid = df[src_col].dropna().unique()
                le.fit(valid)
                df[tgt_col] = df[src_col].map(
                    lambda x: le.transform([x])[0] if pd.notna(x) else pd.NA
                )
                for i, cls in enumerate(le.classes_):
                    encoding_records.append({
                        "original_column": src_col,
                        "encoded_column": tgt_col,
                        "original_value": cls,
                        "encoded_value": int(i),
                    })
                self.log.info("Encoded %s -> %s (%d classes)", src_col, tgt_col, len(le.classes_))

        if encoding_records:
            self.save_artifact(pd.DataFrame(encoding_records), "Encoding_Map.csv")
        else:
            self.log.info("No categorical columns found for encoding")

    def _run_diagnostics(self, df: pd.DataFrame) -> None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            self.log.warning("No numeric columns found for diagnostics")
            return

        stats_rows = []
        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            n = len(vals)
            mean = float(vals.mean())
            median = float(vals.median())
            variance = float(vals.var())
            std = float(vals.std())
            skew = float(vals.skew()) if n > 2 else 0.0
            kurt = float(vals.kurtosis()) if n > 2 else 0.0
            missing = int(df[col].isna().sum())
            missing_pct = round(missing / len(df) * 100, 2) if len(df) > 0 else 0

            norm_stat, norm_p = 0.0, 1.0
            if n > 3:
                try:
                    sample = vals.sample(min(5000, n), random_state=42)
                    norm_stat, norm_p = stats.shapiro(sample)
                except Exception:
                    pass

            stats_rows.append({
                "Variable": col,
                "N": n,
                "Mean": round(mean, 4),
                "Median": round(median, 4),
                "Variance": round(variance, 4),
                "Std_Dev": round(std, 4),
                "Min": round(float(vals.min()), 4),
                "Max": round(float(vals.max()), 4),
                "Skewness": round(skew, 4),
                "Kurtosis": round(kurt, 4),
                "Missing_Count": missing,
                "Missing_Pct": missing_pct,
                "Normality_Stat": round(float(norm_stat), 4),
                "Normality_p": round(float(norm_p), 4),
                "Q1": round(float(vals.quantile(0.25)), 4),
                "Q3": round(float(vals.quantile(0.75)), 4),
                "IQR": round(float(vals.quantile(0.75) - vals.quantile(0.25)), 4),
            })

        if stats_rows:
            self.save_artifact(pd.DataFrame(stats_rows), "Statistical_Profile.csv")

        if len(num_cols) > 1:
            corr_df = df[num_cols].corr(method="pearson")
            self.save_artifact(corr_df, "Correlation_Matrix.csv")

        vif_df = self._calculate_vif(df, num_cols) if len(num_cols) >= 2 else pd.DataFrame(
            {"Variable": num_cols, "VIF": [1.0]}
        )
        self.save_artifact(vif_df, "VIF_Report.csv")

        self.save_artifact(self._generate_missing_profile(df), "Missing_Value_Profile.csv")
        self.save_artifact(self._generate_outlier_profile(df, num_cols), "Outlier_Profile.csv")

        high_vif = vif_df[vif_df["VIF"] > 10]["Variable"].tolist() if not vif_df.empty else []
        self.log.info("Stats: %d variables, %d high-VIF (>10)", len(num_cols), len(high_vif))

    def _calculate_vif(self, df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
        vif_data = []
        for col in num_cols:
            others = [c for c in num_cols if c != col]
            if len(others) < 1:
                vif_data.append({"Variable": col, "VIF": 1.0})
                continue
            sub = df[others].dropna()
            target = df[col].dropna()
            common = sub.index.intersection(target.index)
            if len(common) < 10:
                vif_data.append({"Variable": col, "VIF": float("inf")})
                continue
            X = sub.loc[common].fillna(sub.loc[common].median())
            y = target.loc[common].fillna(target.loc[common].median())
            try:
                lr = LinearRegression()
                lr.fit(X, y)
                r2 = lr.score(X, y)
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
            except Exception:
                vif = float("inf")
            vif_data.append({"Variable": col, "VIF": round(vif, 4)})
        return pd.DataFrame(vif_data)

    def _generate_missing_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            total = len(df)
            pct = round(missing / total * 100, 2) if total > 0 else 0
            rows.append({
                "Column": col,
                "Total": total,
                "Missing": missing,
                "Missing_Pct": pct,
                "Available": total - missing,
                "Available_Pct": round(100 - pct, 2),
            })
        return pd.DataFrame(rows)

    def _generate_outlier_profile(self, df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
        rows = []
        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) > 3:
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_count = int(((vals < lower) | (vals > upper)).sum())
                    outlier_pct = round(outlier_count / len(vals) * 100, 2)
                    rows.append({
                        "Column": col,
                        "Q1": round(q1, 4),
                        "Q3": round(q3, 4),
                        "IQR": round(iqr, 4),
                        "Lower_Fence": round(lower, 4),
                        "Upper_Fence": round(upper, 4),
                        "Outlier_Count": outlier_count,
                        "Outlier_Pct": outlier_pct,
                    })
        return pd.DataFrame(rows)

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "features_engineered": True,
            "leakage_checked": True,
            "encoding_complete": True,
            "diagnostics_complete": True,
        }
