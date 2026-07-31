"""
Consolidated Feature Agent: Feature Engineering + Leakage Detection
+ Categorical Encoding + Statistical Diagnostics.

All engineered features are built in a dict and joined in a single
pd.concat to avoid DataFrame fragmentation (PerformanceWarning).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.schema import (
    NON_FEATURE_COLS,
    PRE_HARVEST_MEASUREMENTS,
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
        new_cols = self._engineer_features(df)
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        self._detect_leakage(df, target_col=kwargs.get("target_col"))
        self._encode_categoricals(df)
        self._run_diagnostics(df)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        new: dict[str, pd.Series] = {}
        tbase = 10.0

        def _num(col):
            return pd.to_numeric(df[col], errors="coerce")

        def _add(name, series):
            if name not in df.columns and name not in new:
                new[name] = series

        if "Temperature_Max" in df.columns and "Temperature_Min" in df.columns:
            tmax = _num("Temperature_Max")
            tmin = _num("Temperature_Min")
            tmean = (tmax + tmin) / 2.0

            _add("Growing_Degree_Days", np.maximum(0, tmean - tbase))
            _add("Heat_Units", np.maximum(0, tmax - tbase))
            _add("Temp_squared", tmean**2)
            _add("Stress_Index", np.abs(tmean - 25.0) / 25.0)
            _add("Temp_Range", tmax - tmin)
            _add("Temp_CV", np.where(tmean > 0, (tmax - tmin) / tmean, 0))

            for t in [10, 15, 20, 25, 30, 35]:
                _add(f"Temp_Above_{t}", np.maximum(0, tmean - t))

        if "Rainfall" in df.columns:
            rainfall = _num("Rainfall")
            _add("Rainfall_Anomaly", rainfall - rainfall.mean())
            _add("Rainfall_log", np.log1p(rainfall.clip(lower=0)))
            _add("Rainfall_sqrt", np.sqrt(rainfall.clip(lower=0)))
            _add("Rainfall_squared", rainfall**2)
            _add(
                "Rainfall_Bin",
                pd.cut(
                    rainfall, bins=[0, 200, 500, 1000, 5000, 10000], labels=[0, 1, 2, 3, 4]
                ).astype(float),
            )
            _add("Rainfall_100_bin", (rainfall / 100).round() * 100)

        if "Rainfall" in df.columns and "Temperature_Max" in df.columns:
            tmean_s = (_num("Temperature_Max") + _num("Temperature_Min")) / 2.0
            _add("Temp_x_Rainfall", tmean_s * _num("Rainfall"))

        if "Nitrogen" in df.columns and "Phosphorus" in df.columns:
            _add("N_x_P", _num("Nitrogen") * _num("Phosphorus"))

        if "Nitrogen" in df.columns:
            n = _num("Nitrogen")
            _add("N_log", np.log1p(n.clip(lower=0)))
            _add("N_sqrt", np.sqrt(n.clip(lower=0)))
            _add("N_squared", n**2)
            _add("N_category", pd.cut(n, bins=5, labels=[0, 1, 2, 3, 4]).astype(float))

        if "Phosphorus" in df.columns:
            p = _num("Phosphorus")
            _add("P_log", np.log1p(p.clip(lower=0)))
            _add("P_squared", p**2)
            _add("P_category", pd.cut(p, bins=5, labels=[0, 1, 2, 3, 4]).astype(float))

        if "Potassium" in df.columns:
            k = _num("Potassium")
            _add("K_log", np.log1p(k.clip(lower=0)))
            _add("K_squared", k**2)
            _add("K_category", pd.cut(k, bins=5, labels=[0, 1, 2, 3, 4]).astype(float))

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Potassium"]):
            n, p, k = _num("Nitrogen"), _num("Phosphorus"), _num("Potassium")
            total = (n + p + k).replace(0, np.nan)
            _add("NPK_sum", n + p + k)
            _add("NPK_ratio_N", n / total)
            _add("NPK_ratio_P", p / total)
            _add("NPK_ratio_K", k / total)
            _add("N_x_K", n * k)
            _add("P_x_K", p * k)
            _add("N_plus_P", n + p)
            _add("N_plus_K", n + k)
            _add("P_plus_K", p + k)

        if "Soil_pH" in df.columns:
            ph = _num("Soil_pH")
            _add("Soil_pH_squared", ph**2)
            _add("Soil_pH_neutral", np.abs(ph - 7.0))
            _add("Soil_pH_acidic", np.where(ph < 6.5, 1, 0))
            _add("Soil_pH_alkaline", np.where(ph > 7.5, 1, 0))

        if "Organic_Carbon" in df.columns:
            _add("OC_log", np.log1p(_num("Organic_Carbon").clip(lower=0)))
            _add("OC_squared", _num("Organic_Carbon") ** 2)

        if all(c in df.columns for c in ["Organic_Carbon", "Nitrogen"]):
            _add("C_N_ratio", _num("Organic_Carbon") / _num("Nitrogen").replace(0, np.nan))

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus"]):
            _add("N_P_ratio", _num("Nitrogen") / _num("Phosphorus").replace(0, np.nan))

        if all(c in df.columns for c in ["Phosphorus", "Potassium"]):
            _add("P_K_ratio", _num("Phosphorus") / _num("Potassium").replace(0, np.nan))

        if "Biomass_Yield" in df.columns:
            bm = _num("Biomass_Yield")
            _add("Biomass_log", np.log1p(bm.clip(lower=0)))
            _add("Biomass_sqrt", np.sqrt(bm.clip(lower=0)))
            _add("Biomass_squared", bm**2)

        if "Humidity" in df.columns and "Temperature_Max" in df.columns:
            hum, tmax = _num("Humidity"), _num("Temperature_Max")
            _add(
                "Disease_Risk_Index",
                np.where(
                    (hum > 80) & (tmax > 25), 1.0, np.where((hum > 60) & (tmax > 20), 0.5, 0.0)
                ),
            )
            _add("Humidity_squared", hum**2)
            _add("Humidity_log", np.log1p(hum.clip(lower=0)))

        if (
            "Yield_per_Plot" in df.columns
            and "Yield_per_Hectare" not in df.columns
            and "Plot_Size" in df.columns
        ):
            ps, yp = _num("Plot_Size"), _num("Yield_per_Plot")
            with np.errstate(divide="ignore", invalid="ignore"):
                _add("Yield_per_Hectare_Calc", np.where(ps > 0, yp / ps * 10000, np.nan))

        if "Fruit_Number" in df.columns and "Fruit_Weight" in df.columns:
            fn, fw = _num("Fruit_Number"), _num("Fruit_Weight")
            with np.errstate(divide="ignore", invalid="ignore"):
                _add("Yield_per_Plant", np.where(fn > 0, fn * fw, np.nan))

        if "Plant_Height_cm" in df.columns:
            ph = _num("Plant_Height_cm")
            _add("Height_log", np.log1p(ph.clip(lower=0)))
            _add("Height_sqrt", np.sqrt(ph.clip(lower=0)))
            _add("Height_squared", ph**2)

        if "SPAD" in df.columns:
            spad = _num("SPAD")
            _add("SPAD_log", np.log1p(spad.clip(lower=0)))
            _add("SPAD_squared", spad**2)
            _add(
                "SPAD_category",
                pd.cut(spad, bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float),
            )

        if all(c in df.columns for c in ["Plant_Height_cm", "Leaf_Area_cm2"]):
            _add("Height_x_LeafArea", _num("Plant_Height_cm") * _num("Leaf_Area_cm2"))

        if all(c in df.columns for c in ["Plant_Height_cm", "Shoot_Biomass_g"]):
            _add("Height_x_ShootBiomass", _num("Plant_Height_cm") * _num("Shoot_Biomass_g"))

        for col in ["Rainfall", "Temperature_Max", "Temperature_Min", "Average_Temperature"]:
            if col in df.columns:
                _add(f"{col}_7d_MA", _num(col).rolling(window=7, min_periods=1).mean())

        if "Average_Temperature" in df.columns and "Nitrogen" in df.columns:
            _add("Temp_x_N", _num("Average_Temperature") * _num("Nitrogen"))

        if "Rainfall" in df.columns and "Nitrogen" in df.columns:
            _add("Rainfall_x_N", _num("Rainfall") * _num("Nitrogen"))

        if "Rainfall" in df.columns and "Phosphorus" in df.columns:
            _add("Rainfall_x_P", _num("Rainfall") * _num("Phosphorus"))

        if "Rainfall" in df.columns and "Potassium" in df.columns:
            _add("Rainfall_x_K", _num("Rainfall") * _num("Potassium"))

        if "Organic_Carbon" in df.columns and "Soil_pH" in df.columns:
            _add("OC_x_pH", _num("Organic_Carbon") * _num("Soil_pH"))

        if "EC" in df.columns and "Soil_pH" in df.columns:
            _add("EC_x_pH", _num("EC") * _num("Soil_pH"))

        for col in [
            "Plant_Height_cm",
            "SPAD",
            "Shoot_Biomass_g",
            "Root_Biomass_g",
            "Leaf_Area_cm2",
            "Fruit_Weight",
            "100_Seed_Weight",
            "Harvest_Index",
        ]:
            if col in df.columns:
                vals = _num(col)
                if vals.min() >= 0:
                    _add(f"{col}_log", np.log1p(vals.clip(lower=0)))

        if "EC" in df.columns:
            ec = _num("EC")
            _add("EC_log", np.log1p(ec.clip(lower=0)))
            _add("EC_squared", ec**2)
            _add(
                "EC_category",
                pd.cut(ec, bins=[0, 0.5, 1.5, 4, 100], labels=[0, 1, 2, 3]).astype(float),
            )

        if "Rainfall" in df.columns and "Soil_pH" in df.columns:
            _add("Rainfall_x_pH", _num("Rainfall") * _num("Soil_pH"))

        if "Rainfall" in df.columns and "Organic_Carbon" in df.columns:
            _add("Rainfall_x_OC", _num("Rainfall") * _num("Organic_Carbon"))

        if all(c in df.columns for c in ["Nitrogen", "Soil_pH"]):
            _add("N_x_pH", _num("Nitrogen") * _num("Soil_pH"))

        if all(c in df.columns for c in ["Phosphorus", "Soil_pH"]):
            _add("P_x_pH", _num("Phosphorus") * _num("Soil_pH"))

        if all(c in df.columns for c in ["Potassium", "Soil_pH"]):
            _add("K_x_pH", _num("Potassium") * _num("Soil_pH"))

        if "Plant_Height_cm" in df.columns and "Nitrogen" in df.columns:
            _add("Height_x_N", _num("Plant_Height_cm") * _num("Nitrogen"))

        if "Plant_Height_cm" in df.columns and "Rainfall" in df.columns:
            _add("Height_x_Rainfall", _num("Plant_Height_cm") * _num("Rainfall"))

        if "Plant_Height_cm" in df.columns and "Soil_pH" in df.columns:
            _add("Height_x_pH", _num("Plant_Height_cm") * _num("Soil_pH"))

        if all(c in df.columns for c in ["Temperature_Max", "Nitrogen"]):
            _add("Tmax_x_N", _num("Temperature_Max") * _num("Nitrogen"))

        if all(c in df.columns for c in ["Temperature_Max", "Rainfall"]):
            _add("Tmax_x_Rainfall", _num("Temperature_Max") * _num("Rainfall"))

        if all(c in df.columns for c in ["Temperature_Min", "Rainfall"]):
            _add("Tmin_x_Rainfall", _num("Temperature_Min") * _num("Rainfall"))

        if all(c in df.columns for c in ["Humidity", "Rainfall"]):
            _add("Humidity_x_Rainfall", _num("Humidity") * _num("Rainfall"))

        if all(c in df.columns for c in ["Humidity", "Nitrogen"]):
            _add("Humidity_x_N", _num("Humidity") * _num("Nitrogen"))

        if all(c in df.columns for c in ["EC", "Organic_Carbon"]):
            _add("EC_x_OC", _num("EC") * _num("Organic_Carbon"))

        if all(c in df.columns for c in ["Nitrogen", "Organic_Carbon"]):
            _add("N_x_OC", _num("Nitrogen") * _num("Organic_Carbon"))

        if all(c in df.columns for c in ["Phosphorus", "Organic_Carbon"]):
            _add("P_x_OC", _num("Phosphorus") * _num("Organic_Carbon"))

        if all(c in df.columns for c in ["Potassium", "Organic_Carbon"]):
            _add("K_x_OC", _num("Potassium") * _num("Organic_Carbon"))

        if "Leaf_Area_cm2" in df.columns:
            la = _num("Leaf_Area_cm2")
            _add("LA_log", np.log1p(la.clip(lower=0)))
            _add("LA_squared", la**2)

        if "Shoot_Biomass_g" in df.columns:
            _add("ShootBiomass_log", np.log1p(_num("Shoot_Biomass_g").clip(lower=0)))

        if "Root_Biomass_g" in df.columns:
            _add("RootBiomass_log", np.log1p(_num("Root_Biomass_g").clip(lower=0)))

        if all(c in df.columns for c in ["Shoot_Biomass_g", "Root_Biomass_g"]):
            sb, rb = _num("Shoot_Biomass_g"), _num("Root_Biomass_g")
            total = sb + rb
            _add("Shoot_Root_Ratio", sb / rb.replace(0, np.nan))
            _add("Root_Shoot_Ratio", rb / sb.replace(0, np.nan))
            _add("Root_pct", rb / total.replace(0, np.nan))

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Soil_pH"]):
            _add("N_x_P_x_pH", _num("Nitrogen") * _num("Phosphorus") * _num("Soil_pH"))

        if "Fruit_Weight" in df.columns:
            fw = _num("Fruit_Weight")
            _add("FruitWeight_log", np.log1p(fw.clip(lower=0)))
            _add("FruitWeight_sqrt", np.sqrt(fw.clip(lower=0)))
            _add("FruitWeight_squared", fw**2)

        if "100_Seed_Weight" in df.columns:
            sw = _num("100_Seed_Weight")
            _add("SeedWeight_log", np.log1p(sw.clip(lower=0)))
            _add("SeedWeight_squared", sw**2)

        if "Harvest_Index" in df.columns:
            hi = _num("Harvest_Index")
            _add("HI_squared", hi**2)
            _add("HI_log", np.log1p(hi.clip(lower=0)))

        if "Average_Temperature" in df.columns:
            tavg = _num("Average_Temperature")
            _add("Tavg_squared", tavg**2)
            _add("Tavg_log", np.log1p(tavg.clip(lower=0)))

        if all(c in df.columns for c in ["Nitrogen", "Phosphorus", "Potassium", "Soil_pH"]):
            n, p, k, ph = _num("Nitrogen"), _num("Phosphorus"), _num("Potassium"), _num("Soil_pH")
            _add("NPK_x_pH", (n + p + k) * ph)

        if all(c in df.columns for c in ["Nitrogen", "Rainfall"]):
            _add("N_Rainfall_ratio", _num("Nitrogen") / _num("Rainfall").replace(0, np.nan))

        if all(c in df.columns for c in ["Biomass_Yield", "Nitrogen"]):
            bm, n = _num("Biomass_Yield"), _num("Nitrogen")
            _add("Biomass_x_N", bm * n)
            _add("Biomass_N_ratio", bm / n.replace(0, np.nan))

        if all(c in df.columns for c in ["Biomass_Yield", "Soil_pH"]):
            _add("Biomass_x_pH", _num("Biomass_Yield") * _num("Soil_pH"))

        if all(c in df.columns for c in ["SPAD", "Nitrogen"]):
            _add("SPAD_x_N", _num("SPAD") * _num("Nitrogen"))

        if all(c in df.columns for c in ["Leaf_Area_cm2", "Nitrogen"]):
            _add("LA_x_N", _num("Leaf_Area_cm2") * _num("Nitrogen"))

        if all(c in df.columns for c in ["Plant_Height_cm", "Phosphorus"]):
            _add("Height_x_P", _num("Plant_Height_cm") * _num("Phosphorus"))

        if all(c in df.columns for c in ["Plant_Height_cm", "Potassium"]):
            _add("Height_x_K", _num("Plant_Height_cm") * _num("Potassium"))

        if all(c in df.columns for c in ["Plant_Height_cm", "SPAD"]):
            _add("Height_x_SPAD", _num("Plant_Height_cm") * _num("SPAD"))

        self.log.info("Engineered %d features", len(new))
        return new

    def _detect_leakage(self, df: pd.DataFrame, target_col: str | None = None) -> None:
        from agri_ai_agent.ml.leakage import is_leaky_feature, strip_engineered

        leaked = []
        safe = []
        non_feature = []
        feature_labels = []

        for col in df.columns:
            if col in NON_FEATURE_COLS:
                non_feature.append(col)
                feature_labels.append((col, "NON_FEATURE"))
            elif is_leaky_feature(col, target_col):
                leaked.append(col)
                feature_labels.append((col, "POST_HARVEST_OR_DERIVED"))
            elif strip_engineered(col) in PRE_HARVEST_MEASUREMENTS:
                safe.append(col)
                feature_labels.append((col, "PRE_HARVEST_MEASUREMENT"))
            else:
                safe.append(col)
                feature_labels.append((col, "AVAILABLE_BEFORE_PREDICTION"))

        leak_df = pd.DataFrame(feature_labels, columns=["Feature", "Availability"])

        if "Feature_Available_Before_Prediction" not in df.columns:
            df["Feature_Available_Before_Prediction"] = (
                leak_df["Availability"]
                .map(lambda x: x == "AVAILABLE_BEFORE_PREDICTION")
                .astype(str)
                .str.upper()
            )

        self.log.info(
            "Leakage: %d leaked, %d non-feature, %d safe", len(leaked), len(non_feature), len(safe)
        )

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
                    lambda x, le=le: le.transform([x])[0] if pd.notna(x) else pd.NA
                )
                for i, cls in enumerate(le.classes_):
                    encoding_records.append(
                        {
                            "original_column": src_col,
                            "encoded_column": tgt_col,
                            "original_value": cls,
                            "encoded_value": int(i),
                        }
                    )
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
                except Exception as e:
                    self.log.debug("Shapiro test failed for %s: %s", col, e)

            stats_rows.append(
                {
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
                }
            )

        if stats_rows:
            self.save_artifact(pd.DataFrame(stats_rows), "Statistical_Profile.csv")

        if len(num_cols) > 1:
            corr_df = df[num_cols].corr(method="pearson")
            self.save_artifact(corr_df, "Correlation_Matrix.csv")

        vif_df = (
            self._calculate_vif(df, num_cols)
            if len(num_cols) >= 2
            else pd.DataFrame({"Variable": num_cols, "VIF": [1.0]})
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
            except Exception as e:
                self.log.debug("VIF computation failed for %s: %s", col, e)
                vif = float("inf")
            vif_data.append({"Variable": col, "VIF": round(vif, 4)})
        return pd.DataFrame(vif_data)

    def _generate_missing_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            total = len(df)
            pct = round(missing / total * 100, 2) if total > 0 else 0
            rows.append(
                {
                    "Column": col,
                    "Total": total,
                    "Missing": missing,
                    "Missing_Pct": pct,
                    "Available": total - missing,
                    "Available_Pct": round(100 - pct, 2),
                }
            )
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
                    rows.append(
                        {
                            "Column": col,
                            "Q1": round(q1, 4),
                            "Q3": round(q3, 4),
                            "IQR": round(iqr, 4),
                            "Lower_Fence": round(lower, 4),
                            "Upper_Fence": round(upper, 4),
                            "Outlier_Count": outlier_count,
                            "Outlier_Pct": outlier_pct,
                        }
                    )
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
