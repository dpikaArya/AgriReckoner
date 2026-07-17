"""
Agricultural Ontology Agent
Builds and maintains ontology dictionaries for column name mapping,
unit normalization, and agricultural terminology resolution.
Auto-updates when new synonyms are encountered.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.contracts.messages import AgentContract
from agri_ai_agent.config.settings import AgriAISettings


COLONYM_MAP: dict[str, list[str]] = {
    "Yield_per_Hectare": [
        "yield", "yield_per_ha", "yield/ha", "yield_kg_ha", "grain_yield",
        "crop_yield", "productivity", "yield_kg_ha-1", "yield t/ha",
        "yield (kg/ha)", "yield kg ha-1", "dry matter yield",
    ],
    "Yield_per_Plot": [
        "yield_per_plot", "plot_yield", "yield (kg/plot)", "yield g_plot",
    ],
    "Yield_per_Acre": [
        "yield_per_acre", "yield/acre", "yield kg_acre",
    ],
    "Plant_Height_cm": [
        "plant_height", "height", "height_cm", "height (cm)",
        "vegetative_height", "crop_height", "stem_height", "standing_height",
    ],
    "Shoot_Length_cm": [
        "shoot_length", "shoot_length_cm", "shoot (cm)", "shoot_height",
    ],
    "Root_Length_cm": [
        "root_length", "root_length_cm", "root (cm)", "root_depth",
    ],
    "Stem_Diameter_mm": [
        "stem_diameter", "stem_diameter_mm", "stem dia (mm)",
    ],
    "Root_Diameter_mm": [
        "root_diameter", "root_diameter_mm", "root dia (mm)",
    ],
    "Leaf_Area_cm2": [
        "leaf_area", "la", "la (cm2)", "leaf_area_cm2", "total_leaf_area",
        "leaf area cm-2", "specific_leaf_area",
    ],
    "Leaf_Number": [
        "leaf_number", "num_leaves", "leaves", "leaf_count",
        "number_of_leaves", "total_leaves",
    ],
    "SPAD": [
        "spad", "spad_value", "chlorophyll", "spad_reading",
        "spad value", "chlorophyll_content",
    ],
    "Shoot_Biomass_g": [
        "shoot_biomass", "shoot_weight", "shoot (g)", "aerial_biomass",
        "above_ground_biomass", "shoot_biomass_g",
    ],
    "Root_Biomass_g": [
        "root_biomass", "root_weight", "root (g)", "root_biomass_g",
    ],
    "Biomass_Yield": [
        "biomass", "biomass_yield", "total_biomass", "dry_biomass",
        "biomass (kg/ha)", "biomass_kg_ha",
    ],
    "Dry_Matter": [
        "dry_matter", "dm", "dry_matter_content", "dry matter %",
    ],
    "Moisture_Content": [
        "moisture", "moisture_content", "water_content", "moisture %",
    ],
    "Tillers": [
        "tillers", "tiller_count", "num_tillers", "tillers_per_plant",
    ],
    "Branches": [
        "branches", "branch_count", "num_branches", "branching",
    ],
    "Nodes": [
        "nodes", "node_count", "num_nodes",
    ],
    "Flowers": [
        "flowers", "flower_count", "num_flowers", "flowering",
    ],
    "Fruit_Number": [
        "fruit_number", "num_fruits", "fruit_count", "fruits_per_plant",
    ],
    "Fruit_Weight": [
        "fruit_weight", "fruit_mass", "fruit (g)", "average_fruit_weight",
    ],
    "Fruit_Diameter_mm": [
        "fruit_diameter", "fruit_size", "fruit dia (mm)",
    ],
    "100_Seed_Weight": [
        "100_seed_weight", "hsw", "1000_grain_weight", "seed_weight_100",
        "hundred_seed_weight", "tgw", "thousand_grain_weight",
    ],
    "Spike_Length": [
        "spike_length", "ear_length", "spike (cm)",
    ],
    "Seeds_per_Spike": [
        "seeds_per_spike", "grains_per_spike", "kernels_per_ear",
    ],
    "Harvest_Index": [
        "harvest_index", "hi", "harvest index", "hi ratio",
    ],
    "Protein": [
        "protein", "protein_content", "crude_protein", "protein %",
        "protein_content_%",
    ],
    "Ash": [
        "ash", "ash_content", "ash %", "mineral_content",
    ],
    "Gluten": [
        "gluten", "gluten_content", "gluten %", "wet_gluten",
    ],
    "Fiber": [
        "fiber", "fiber_content", "crude_fiber", "fiber %",
        "dietary_fiber",
    ],
    "Carbohydrates": [
        "carbohydrates", "carbs", "carb_content", "total_carbohydrates",
    ],
    "Fat": [
        "fat", "fat_content", "crude_fat", "oil_content", "fat %",
    ],
    "Soil_pH": [
        "soil_ph", "ph", "soil_ph_value", "ph_value", "soil ph",
        "ph (1:2.5)", "ph water",
    ],
    "EC": [
        "ec", "electrical_conductivity", "ec_ds_m", "ec (ds/m)",
        "salinity",
    ],
    "Organic_Carbon": [
        "organic_carbon", "oc", "soc", "organic_carbon_%",
        "soil_organic_carbon", "organic carbon %",
    ],
    "Organic_Matter": [
        "organic_matter", "om", "organic_matter_%", "som",
        "soil_organic_matter",
    ],
    "Nitrogen": [
        "nitrogen", "n", "total_nitrogen", "n_content", "n_kg_ha",
        "available_nitrogen", "nitrogen_kg_ha", "soil_nitrogen",
    ],
    "Phosphorus": [
        "phosphorus", "p", "total_phosphorus", "p_content", "p_kg_ha",
        "available_phosphorus", "phosphorus_kg_ha", "soil_phosphorus",
    ],
    "Potassium": [
        "potassium", "k", "total_potassium", "k_content", "k_kg_ha",
        "available_potassium", "potassium_kg_ha", "soil_potassium",
    ],
    "Sulphur": [
        "sulphur", "s", "sulfur", "sulphur_content",
    ],
    "Iron": [
        "iron", "fe", "iron_content", "fe_ppm",
    ],
    "Copper": [
        "copper", "cu", "copper_content", "cu_ppm",
    ],
    "Manganese": [
        "manganese", "mn", "manganese_content", "mn_ppm",
    ],
    "Zinc": [
        "zinc", "zn", "zinc_content", "zn_ppm",
    ],
    "Calcium": [
        "calcium", "ca", "calcium_content", "ca_ppm",
    ],
    "Magnesium": [
        "magnesium", "mg", "magnesium_content", "mg_ppm",
    ],
    "Boron": [
        "boron", "b", "boron_content", "b_ppm",
    ],
    "Dose": [
        "dose", "dosage", "application_rate", "rate", "dose_kg_ha",
        "dose (kg/ha)", "rate_kg_ha",
    ],
    "Fertilizer_Name": [
        "fertilizer", "fertilizer_name", "fert_type", "fertilizer_type",
        "nutrient_source", "product",
    ],
    "Application_Method": [
        "application_method", "method", "application", "placement",
        "foliar", "soil_application",
    ],
    "Application_Interval": [
        "application_interval", "interval", "frequency", "timing",
        "days_between",
    ],
    "Temperature_Max": [
        "temp_max", "maximum_temperature", "tmax", "max_temp",
        "temperature_max_c", "tmax_c",
    ],
    "Temperature_Min": [
        "temp_min", "minimum_temperature", "tmin", "min_temp",
        "temperature_min_c", "tmin_c",
    ],
    "Average_Temperature": [
        "avg_temp", "mean_temperature", "temperature", "temp_avg",
        "average_temp", "mean_temp",
    ],
    "Rainfall": [
        "rainfall", "precipitation", "rain", "rainfall_mm",
        "total_rainfall", "precip",
    ],
    "Humidity": [
        "humidity", "relative_humidity", "rh", "humidity_%",
    ],
    "Variety": [
        "variety", "cultivar", "cv", "var", "cultivar_name",
        "variety_name",
    ],
    "Season": [
        "season", "cropping_season", "growing_season", "season_name",
    ],
    "Location": [
        "location", "site", "place", "station", "locality",
        "experimental_site",
    ],
    "Country": [
        "country", "country_name", "nation",
    ],
    "Latitude": [
        "latitude", "lat", "lat_dd", "lat_deg",
    ],
    "Longitude": [
        "longitude", "lon", "lng", "lon_dd", "lon_deg",
    ],
    "Altitude": [
        "altitude", "elevation", "asl", "altitude_m",
    ],
    "Design": [
        "design", "experimental_design", "design_type", "layout",
        "trial_design",
    ],
    "Replications": [
        "replications", "reps", "replicates", "n_reps", "number_of_reps",
    ],
    "Plot_Size": [
        "plot_size", "plot_area", "plot (m2)", "plot_size_m2",
    ],
    "Spacing_Row": [
        "row_spacing", "spacing_row", "row_spacing_cm", "spacing (cm)",
    ],
    "Spacing_Plant": [
        "plant_spacing", "spacing_plant", "intra_row_spacing",
        "plant_spacing_cm",
    ],
    "Growth_Duration_Days": [
        "growth_duration", "duration", "days_to_maturity", "crop_duration",
        "duration_days",
    ],
    "Growth_Stage": [
        "growth_stage", "stage", "phenological_stage", "development_stage",
    ],
    "Sample_Size": [
        "sample_size", "n", "sample_n", "observations",
    ],
    "Target_Yield": [
        "target_yield", "yield_target", "expected_yield",
    ],
    "Target_Nitrogen": [
        "target_nitrogen", "n_target", "recommended_nitrogen",
    ],
    "Target_Phosphorus": [
        "target_phosphorus", "p_target", "recommended_phosphorus",
    ],
    "Target_Potassium": [
        "target_potassium", "k_target", "recommended_potassium",
    ],
}

UNIT_SYNONYMS: dict[str, list[str]] = {
    "kg_per_ha": ["kg/ha", "kg ha-1", "kg ha", "kg/ha.", "kg ha.", "kg ha⁻¹"],
    "t_per_ha": ["t/ha", "t ha-1", "t ha", "ton/ha", "tonnes/ha"],
    "q_per_ha": ["q/ha", "q ha-1", "q ha", "quintal/ha"],
    "kg_per_acre": ["kg/acre", "kg acre-1", "kg acre"],
    "g_per_plot": ["g/plot", "g plot-1", "g plot"],
    "cm": ["cm", "centimeter", "centimetre"],
    "mm": ["mm", "millimeter", "millimetre"],
    "m": ["m", "meter", "metre"],
    "g": ["g", "gram", "grams"],
    "mg": ["mg", "milligram"],
    "kg": ["kg", "kilogram", "kilograms"],
    "ppm": ["ppm", "mg/kg", "mg kg-1", "mg kg"],
    "ds_per_m": ["dS/m", "ds/m", "dS m-1", "ds m"],
    "percent": ["%", "percent", "pct"],
    "celsius": ["°C", "C", "deg C", "degree C", "degrees C"],
    "mm_day": ["mm/day", "mm day-1", "mm d-1"],
}


class OntologyAgent(BaseAgent):
    def __init__(self, settings: Optional[AgriAISettings] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.colonymap = dict(COLONYM_MAP)
        self.unit_synonyms = dict(UNIT_SYNONYMS)
        self.new_synonyms: list[dict[str, str]] = []
        self.ontology_path = self.settings.OUTPUT_DIR / "ontology_map.json"

    @property
    def agent_name(self) -> str:
        return "OntologyAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = df.copy()
        mapping_report: dict[str, Any] = {
            "columns_mapped": 0,
            "columns_unmapped": 0,
            "synonyms_added": 0,
            "mapping": {},
        }

        rename_map: dict[str, str] = {}
        unmapped: list[str] = []

        for col in df.columns:
            canonical = self._resolve_column(col)
            if canonical and canonical != col:
                rename_map[col] = canonical
                mapping_report["mapping"][col] = canonical
                mapping_report["columns_mapped"] += 1
            elif canonical:
                mapping_report["mapping"][col] = col
                mapping_report["columns_mapped"] += 1
            else:
                unmapped.append(col)
                mapping_report["columns_unmapped"] += 1

        if rename_map:
            df = df.rename(columns=rename_map)

        df = self._normalize_column_names(df)
        df = self._standardize_units(df)

        self._learn_new_synonyms(df)

        mapping_report["synonyms_added"] = len(self.new_synonyms)
        self._save_ontology(mapping_report)

        return df

    def _resolve_column(self, col_name: str) -> Optional[str]:
        normalized = self._normalize_name(col_name)

        for canonical, synonyms in self.colonymap.items():
            if normalized == self._normalize_name(canonical):
                return canonical
            for syn in synonyms:
                if normalized == self._normalize_name(syn):
                    return canonical

        return None

    def _normalize_name(self, name: str) -> str:
        n = str(name).lower().strip()
        n = re.sub(r"\s+", "_", n)
        n = n.replace("-", "_").replace(".", "").replace("(", "").replace(")", "")
        n = n.replace("/", "_").replace("%", "pct").replace("°", "")
        n = re.sub(r"_+", "_", n)
        return n

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        new_columns: list[str] = []
        seen: set[str] = set()
        for col in df.columns:
            normalized = col
            if normalized in seen:
                i = 1
                while f"{normalized}_{i}" in seen:
                    i += 1
                normalized = f"{normalized}_{i}"
            seen.add(normalized)
            new_columns.append(normalized)
        df.columns = new_columns
        return df

    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_cols = [
            "Dose", "Nitrogen", "Phosphorus", "Potassium",
            "Yield_per_Hectare", "Yield_per_Acre",
        ]
        for col in unit_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _learn_new_synonyms(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            normalized = self._normalize_name(col)
            is_known = False
            for canonical, synonyms in self.colonymap.items():
                if normalized == self._normalize_name(canonical):
                    is_known = True
                    break
                for syn in synonyms:
                    if normalized == self._normalize_name(syn):
                        is_known = True
                        break
                if is_known:
                    break

            if not is_known and len(col) > 2:
                best_canonical = self._guess_canonical(col, df)
                if best_canonical:
                    self.colonymap.setdefault(best_canonical, []).append(col)
                    self.new_synonyms.append({
                        "new_synonym": col,
                        "mapped_to": best_canonical,
                    })

    def _guess_canonical(self, col_name: str, df: pd.DataFrame) -> Optional[str]:
        normalized = self._normalize_name(col_name)

        for canonical in self.colonymap:
            canon_norm = self._normalize_name(canonical)
            if normalized.startswith(canon_norm[:4]) and len(canon_norm[:4]) >= 4:
                return canonical

        if col_name in df.columns:
            series = df[col_name]
            if pd.api.types.is_numeric_dtype(series):
                vals = series.dropna()
                if len(vals) > 0:
                    mean_val = vals.mean()
                    if 0 < mean_val < 20:
                        return "Soil_pH"
                    elif 100 < mean_val < 10000:
                        return "Yield_per_Hectare"
                    elif 10 < mean_val < 200:
                        return "Plant_Height_cm"

        return None

    def _save_ontology(self, mapping_report: dict) -> None:
        output_dir = self.settings.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        ontology_data = {
            "colonymap": {k: v for k, v in self.colonymap.items()},
            "unit_synonyms": self.unit_synonyms,
            "new_synonyms": self.new_synonyms,
            "mapping_report": mapping_report,
        }

        with open(self.ontology_path, "w", encoding="utf-8") as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)

    def resolve_value(self, value: str, unit_type: str = "general") -> Optional[float]:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s in ("-", "–", "—", "ns", "NA", "N/A", ".", "nd", "ND"):
            return None
        s = re.sub(r"[†‡*]", "", s)
        s = re.sub(r"\s*±\s*.*", "", s)
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            match = re.search(r"(\d+\.?\d*)", s)
            return float(match.group(1)) if match else None

    def get_canonical_columns(self) -> list[str]:
        return list(self.colonymap.keys())

    def get_synonyms(self, canonical: str) -> list[str]:
        return self.colonymap.get(canonical, [])

    def add_synonym(self, canonical: str, synonym: str) -> None:
        if canonical not in self.colonymap:
            self.colonymap[canonical] = []
        if synonym not in self.colonymap[canonical]:
            self.colonymap[canonical].append(synonym)
            self.new_synonyms.append({
                "new_synonym": synonym,
                "mapped_to": canonical,
            })
