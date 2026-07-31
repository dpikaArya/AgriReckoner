import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("SoilIndices")


class SoilIndices:
    def __init__(self):
        self.generated_features_: list[str] = []

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result = self.soil_fertility_index(result)
        result = self.soil_ph_suitability(result)
        result = self.organic_carbon_score(result)
        result = self.npk_balance_index(result)
        result = self.micronutrient_availability(result)
        return result

    def soil_fertility_index(self, df: pd.DataFrame) -> pd.DataFrame:
        oc = self._find_col(df, ["Organic_Carbon_pct", "OC", "Organic_Carbon"])
        n = self._find_col(df, ["Nitrogen_kg_ha", "N", "Total_Nitrogen"])
        p = self._find_col(df, ["Phosphorus_kg_ha", "P", "Available_Phosphorus"])
        k = self._find_col(df, ["Potassium_kg_ha", "K", "Available_Potassium"])
        ph = self._find_col(df, ["Soil_pH", "pH", "p H"])

        components = []
        if oc is not None:
            oc_score = (df[oc] / 1.5).clip(0, 1)
            components.append(oc_score)
        if n is not None:
            n_score = (df[n] / 200).clip(0, 1)
            components.append(n_score)
        if p is not None:
            p_score = (df[p] / 50).clip(0, 1)
            components.append(p_score)
        if k is not None:
            k_score = (df[k] / 300).clip(0, 1)
            components.append(k_score)
        if ph is not None:
            ph_score = 1 - ((df[ph] - 6.5) / 3.5).abs().clip(0, 1)
            components.append(ph_score)

        if components:
            df["Soil_Fertility_Index"] = pd.concat(components, axis=1).mean(axis=1)
            self.generated_features_.append("Soil_Fertility_Index")
        return df

    def soil_ph_suitability(self, df: pd.DataFrame) -> pd.DataFrame:
        ph = self._find_col(df, ["Soil_pH", "pH", "p H"])
        if ph is None:
            return df

        df["Soil_pH_Suitability_Score"] = 1 - ((df[ph] - 6.5) / 3.5).abs().clip(0, 1)
        self.generated_features_.append("Soil_pH_Suitability_Score")

        for crop, (ph_min, ph_max, ph_opt) in [
            ("Wheat", (5.5, 8.0, 6.5)),
            ("Rice", (5.0, 7.5, 6.0)),
            ("Maize", (5.5, 7.5, 6.5)),
            ("Potato", (5.0, 6.5, 5.8)),
            ("Cotton", (5.5, 8.0, 6.5)),
        ]:
            name = f"pH_Suitability_{crop}"
            df[name] = 1 - (
                (df[ph] - ph_opt) / max(ph_max - ph_opt, ph_opt - ph_min, 0.1)
            ).abs().clip(0, 1)
            self.generated_features_.append(name)

        return df

    def organic_carbon_score(self, df: pd.DataFrame) -> pd.DataFrame:
        oc = self._find_col(df, ["Organic_Carbon_pct", "OC", "Organic_Carbon"])
        if oc is None:
            return df

        bins = [-np.inf, 0.4, 0.75, 1.5, np.inf]
        labels = [0.2, 0.5, 0.8, 1.0]
        df["OC_Score"] = pd.cut(df[oc], bins=bins, labels=labels, right=False).astype(float)
        self.generated_features_.append("OC_Score")

        thresholds = [0.5, 1.0, 1.5]
        for t in thresholds:
            name = f"OC_Above_{str(t).replace('.', '_')}"
            df[name] = (df[oc] >= t).astype(float)
            self.generated_features_.append(name)

        return df

    def npk_balance_index(self, df: pd.DataFrame) -> pd.DataFrame:
        n = self._find_col(df, ["Nitrogen_kg_ha", "N", "Total_Nitrogen"])
        p = self._find_col(df, ["Phosphorus_kg_ha", "P", "Available_Phosphorus"])
        k = self._find_col(df, ["Potassium_kg_ha", "K", "Available_Potassium"])

        if n is not None and p is not None and k is not None:
            ideal_npk = np.array([1.0, 0.5, 1.0])
            df["NPK_Balance_Index"] = df.apply(
                lambda row: self._npk_balance(row[n], row[p], row[k], ideal_npk),
                axis=1,
            )
            self.generated_features_.append("NPK_Balance_Index")

        if n is not None and k is not None:
            df["N_K_Ratio"] = (df[n] / (df[k] + 1e-6)).clip(0, 10)
            self.generated_features_.append("N_K_Ratio")

        return df

    def micronutrient_availability(self, df: pd.DataFrame) -> pd.DataFrame:
        ph = self._find_col(df, ["Soil_pH", "pH", "p H"])
        oc = self._find_col(df, ["Organic_Carbon_pct", "OC", "Organic_Carbon"])

        if ph is not None:
            df["Zn_Availability_Score"] = np.exp(-((df[ph] - 6.0) ** 2) / 4)
            self.generated_features_.append("Zn_Availability_Score")
            df["Fe_Availability_Score"] = np.exp(-((df[ph] - 5.5) ** 2) / 6)
            self.generated_features_.append("Fe_Availability_Score")
            df["Mn_Availability_Score"] = np.exp(-((df[ph] - 5.8) ** 2) / 5)
            self.generated_features_.append("Mn_Availability_Score")

        if oc is not None:
            df["Micro_Nutrient_Score"] = (df[oc] / 1.0).clip(0, 1)
            self.generated_features_.append("Micro_Nutrient_Score")

        if ph is not None and oc is not None:
            df["Integrated_Fertility_Score"] = (
                self._ph_score(df[ph]) * 0.4 + (df[oc] / 1.5).clip(0, 1) * 0.6
            )
            self.generated_features_.append("Integrated_Fertility_Score")

        return df

    def _npk_balance(self, n_val, p_val, k_val, ideal: np.ndarray) -> float:
        eps = 1e-6
        actual = np.array([n_val, p_val, k_val])
        total = actual.sum()
        if total < eps:
            return 0.0
        actual_norm = actual / total
        diff = np.abs(actual_norm - (ideal / ideal.sum()))
        return float(max(0, 1 - diff.sum()))

    def _ph_score(self, ph: pd.Series) -> pd.Series:
        return (1 - ((ph - 6.5) / 3.5).abs()).clip(0, 1)

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None
