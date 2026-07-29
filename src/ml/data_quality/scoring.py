import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("DatasetScorer")


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DatasetScorer:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scores_: Optional[pd.DataFrame] = None

    def score_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty dataset for scoring")
            return pd.DataFrame()

        scores = pd.DataFrame(index=df.index)
        scores["crop_score"] = self._score_crop(df)
        scores["location_score"] = self._score_location(df)
        scores["soil_score"] = self._score_soil(df)
        scores["climate_score"] = self._score_climate(df)
        scores["management_score"] = self._score_management(df)
        scores["yield_score"] = self._score_yield(df)

        score_cols = [
            "crop_score", "location_score", "soil_score",
            "climate_score", "management_score", "yield_score",
        ]
        scores["total_score"] = scores[score_cols].sum(axis=1) / len(score_cols)
        scores["tier"] = scores["total_score"].apply(self._assign_tier)
        self.scores_ = scores

        self._save_results(df, scores)
        return scores

    def _score_crop(self, df: pd.DataFrame) -> pd.Series:
        crop_cols = [
            "Crop", "Crop_Type", "Crop_Name", "Crop_Variety",
            "Crop_Season", "Season", "Crop_Duration_days",
        ]
        present = [c for c in crop_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        non_null = df[present].notna().sum(axis=1)
        return (non_null / len(present)).clip(0, 1)

    def _score_location(self, df: pd.DataFrame) -> pd.Series:
        loc_cols = [
            "Location", "Region", "Country", "State", "District",
            "Latitude", "Longitude", "Altitude_m", "Agro_Ecological_Zone",
        ]
        present = [c for c in loc_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        non_null = df[present].notna().sum(axis=1)
        return (non_null / len(present)).clip(0, 1)

    def _score_soil(self, df: pd.DataFrame) -> pd.Series:
        soil_cols = [
            "Soil_pH", "Soil_Texture", "Soil_Type", "Organic_Carbon_pct",
            "Nitrogen_kg_ha", "Phosphorus_kg_ha", "Potassium_kg_ha",
            "Cation_Exchange_Capacity", "Soil_Depth_cm",
        ]
        present = [c for c in soil_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        non_null = df[present].notna().sum(axis=1)
        return (non_null / len(present)).clip(0, 1)

    def _score_climate(self, df: pd.DataFrame) -> pd.Series:
        climate_cols = [
            "Average_Temperature", "Temperature_Max", "Temperature_Min",
            "Rainfall_mm", "Humidity_pct", "Solar_Radiation",
            "Wind_Speed", "Evapotranspiration",
        ]
        present = [c for c in climate_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        non_null = df[present].notna().sum(axis=1)
        return (non_null / len(present)).clip(0, 1)

    def _score_management(self, df: pd.DataFrame) -> pd.Series:
        mgmt_cols = [
            "Fertilizer_Name", "Fertilizer_Type", "Dose_kg_acre",
            "Irrigation", "Planting_Date", "Harvest_Date",
            "Planting_Density", "Row_Spacing_cm",
        ]
        present = [c for c in mgmt_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        non_null = df[present].notna().sum(axis=1)
        return (non_null / len(present)).clip(0, 1)

    def _score_yield(self, df: pd.DataFrame) -> pd.Series:
        yield_cols = [
            "Yield_per_Hectare", "Yield_per_Plot", "Grain_Yield",
            "Biomass_yield", "Fruit_Yield", "Seed_Yield",
        ]
        present = [c for c in yield_cols if c in df.columns]
        if not present:
            return pd.Series(0.0, index=df.index)
        has_yield = df[present].notna().any(axis=1).astype(float)
        multiple = df[present].notna().sum(axis=1)
        return (has_yield + (multiple / len(present)) * 0.5).clip(0, 1)

    def _assign_tier(self, score: float) -> str:
        if score >= 0.8:
            return QualityTier.HIGH.value
        elif score >= 0.5:
            return QualityTier.MEDIUM.value
        return QualityTier.LOW.value

    def _save_results(self, df: pd.DataFrame, scores: pd.DataFrame):
        result = df.copy()
        for col in scores.columns:
            result[col] = scores[col]

        high = result[result["tier"] == QualityTier.HIGH.value]
        med = result[result["tier"] == QualityTier.MEDIUM.value]
        low = result[result["tier"] == QualityTier.LOW.value]

        tier_counts = {
            "high": len(high),
            "medium": len(med),
            "low": len(low),
        }

        if not high.empty:
            gold_path = Path("data") / "processed" / "gold_standard_dataset.csv"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            high.to_csv(gold_path, index=False)
            logger.info("Saved gold standard dataset (%d records) to %s", len(high), gold_path)

        labeled_path = self.output_dir / "dataset_quality_scores.csv"
        result.to_csv(labeled_path, index=False)
        logger.info("Saved quality scores to %s", labeled_path)

        summary = {
            "total_records": len(result),
            "tier_counts": tier_counts,
            "high_quality_pct": round(len(high) / max(len(result), 1) * 100, 2),
            "average_score": round(scores["total_score"].mean(), 4),
            "score_distribution": {
                col: {
                    "mean": round(scores[col].mean(), 4),
                    "std": round(scores[col].std(), 4),
                }
                for col in scores.columns if col not in ["total_score", "tier"]
            },
        }
        summary_path = self.output_dir / "dataset_quality_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Dataset quality summary: %s", tier_counts)

    def get_high_quality_mask(self, scores: pd.DataFrame) -> pd.Series:
        return scores["tier"] == QualityTier.HIGH.value

    def get_weighted_sample_weights(self, scores: pd.DataFrame) -> pd.Series:
        weights = scores["total_score"].copy()
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        weights = 0.5 + weights * 0.5
        return weights
