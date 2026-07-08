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


POST_HARVEST_VARIABLES = {
    "Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat",
    "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content",
    "Iron_Content", "Copper_Content", "Zinc_Content",
    "Manganese_Content", "Sulphur_Content",
    "100_Seed_Weight", "Root_Weight", "Pod_Weight",
    "Harvest_Index", "Biomass_Yield",
    "Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare",
    "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm",
    "Spike_Length", "Seeds_per_Spike",
    "Target_Yield", "Target_Fertilizer",
    "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
    "Yield_per_Plant", "Yield_per_Plot_Calc", "Yield_per_Hectare_Calc",
    "Nitrogen_Use_Efficiency", "Water_Use_Efficiency",
    "Harvest_Index_Calc",
}

PRE_HARVEST_MEASUREMENTS = {
    "Leaf_Number", "Tillers", "Branches", "Nodes", "Flowers",
    "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm",
    "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2",
    "SPAD", "Moisture_Content", "Dry_Matter",
    "Shoot_Length_cm", "Root_Length_cm",
    "Stem_Diameter_mm", "Root_Diameter_mm",
    "Plant_Height_cm",
}

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

        if "Rainfall" in df.columns and "Temperature_Max" in df.columns:
            tmean = (pd.to_numeric(df["Temperature_Max"], errors="coerce") +
                     pd.to_numeric(df["Temperature_Min"], errors="coerce")) / 2.0
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Temp_x_Rainfall" not in df.columns:
                df["Temp_x_Rainfall"] = tmean * rainfall
                features_added.append("Temp_x_Rainfall")

        if "Nitrogen" in df.columns and "Phosphorus" in df.columns:
            n = pd.to_numeric(df["Nitrogen"], errors="coerce")
            p = pd.to_numeric(df["Phosphorus"], errors="coerce")
            if "N_x_P" not in df.columns:
                df["N_x_P"] = n * p
                features_added.append("N_x_P")

        if "Rainfall" in df.columns:
            rainfall = pd.to_numeric(df["Rainfall"], errors="coerce")
            if "Rainfall_Anomaly" not in df.columns:
                df["Rainfall_Anomaly"] = rainfall - rainfall.mean()
                features_added.append("Rainfall_Anomaly")

        if "Biomass_Yield" in df.columns and "Yield_per_Hectare" in df.columns:
            biomass = pd.to_numeric(df["Biomass_Yield"], errors="coerce")
            yield_ha = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                if "Harvest_Index_Calc" not in df.columns:
                    df["Harvest_Index_Calc"] = np.where(biomass > 0, yield_ha / biomass, np.nan)
                    features_added.append("Harvest_Index_Calc")

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

        for col in ["Rainfall", "Temperature_Max", "Temperature_Min", "Average_Temperature"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                col_name = f"{col}_7d_MA"
                if col_name not in df.columns:
                    df[col_name] = vals.rolling(window=7, min_periods=1).mean()
                    features_added.append(col_name)

        self.log.info("Engineered %d features: %s", len(features_added), features_added)

    def _detect_leakage(self, df: pd.DataFrame) -> None:
        leaked = []
        safe = []
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

        self.log.info("Leakage: %d leaked, %d safe", len(leaked), len(safe))

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
