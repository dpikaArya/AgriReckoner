from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

from agri_ai_agent.config.schema import UAMS_COLUMNS, VARIANT_MAP

KNOWN_SOURCE_MAPS: dict[str, dict[str, str]] = {
    "NASA_POWER": {
        "T2M": "Average_Temperature",
        "T2M_MAX": "Temperature_Max",
        "T2M_MIN": "Temperature_Min",
        "T2MDEW": "Temperature_Min",
        "T2MWET": "Temperature_Max",
        "PRECTOTCORR": "Rainfall",
        "PRECTOT": "Rainfall",
        "ALLSKY_SFC_SW_DWN": "Solar_Radiation",
        "ALLSKY_SFC_PAR_TOT": "Solar_Radiation",
        "CLRSKY_SFC_SW_DWN": "Solar_Radiation",
        "RH2M": "Humidity",
        "WS2M": "Wind_Speed",
        "WS2M_MAX": "Wind_Speed",
        "WS2M_MIN": "Wind_Speed",
        "WS50M": "Wind_Speed",
        "WD2M": "Wind_Speed",
        "EVPTRANS": "ET0",
        "ETR": "ET0",
        "GW_ETR": "ET0",
        "QV2M": "Humidity",
        "PS": "Altitude",
    },
    "SoilGrids": {
        "soil_pH": "Soil_pH",
        "phh2o": "Soil_pH",
        "ph": "Soil_pH",
        "cec": "CEC",
        "clay": "Soil_Clay_Pct",
        "silt": "Soil_Silt_Pct",
        "sand": "Soil_Sand_Pct",
        "soc": "Organic_Carbon",
        "nitrogen": "Nitrogen",
        "n": "Nitrogen",
        "bdod": "Soil_Bulk_Density",
        "cfvo": "Soil_Bulk_Density",
        "orcdrc": "Organic_Carbon",
        "ocd": "Organic_Carbon",
    },
    "FAOSTAT": {
        "area": "Country",
        "area_code": "Country_Code",
        "item": "Crop",
        "item_code": "Crop_Code",
        "year": "Year",
        "value": "Yield_per_Hectare",
        "unit": "Unit",
        "production": "Biomass_Yield",
        "yield": "Yield_per_Hectare",
        "harvested_area": "Plot_Size",
        "seed": "Seed_Yield",
    },
    "Zenodo": {
        "lat": "Latitude",
        "latitude": "Latitude",
        "lon": "Longitude",
        "longitude": "Longitude",
        "long": "Longitude",
        "temp": "Average_Temperature",
        "temperature": "Average_Temperature",
        "precip": "Rainfall",
        "precipitation": "Rainfall",
        "rainfall": "Rainfall",
        "rh": "Humidity",
        "humidity": "Humidity",
        "solar_rad": "Solar_Radiation",
        "wind": "Wind_Speed",
        "wind_speed": "Wind_Speed",
        "soil_ph": "Soil_pH",
        "ph": "Soil_pH",
        "organic_carbon": "Organic_Carbon",
        "oc": "Organic_Carbon",
        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "potassium": "Potassium",
        "yield": "Yield_per_Hectare",
    },
    "CGIAR": {
        "lat": "Latitude",
        "latitude": "Latitude",
        "lon": "Longitude",
        "longitude": "Longitude",
        "crop": "Crop",
        "variety": "Variety",
        "season": "Season",
        "year": "Year",
        "yield": "Yield_per_Hectare",
        "plant_height": "Plant_Height_cm",
        "biomass": "Shoot_Biomass_g",
        "rainfall": "Rainfall",
        "temperature": "Average_Temperature",
        "soil_ph": "Soil_pH",
        "nitrogen": "Nitrogen",
        "phosphorus": "Phosphorus",
        "potassium": "Potassium",
    },
}

FUZZY_THRESHOLD = 0.75

UAMS_SET = {c.lower().replace(" ", "_").replace("-", "_") for c in UAMS_COLUMNS}

_normalized_uams: dict[str, str] = {}
for col in UAMS_COLUMNS:
    key = col.lower().replace(" ", "_").replace("-", "_")
    _normalized_uams[key] = col

# Add VARIANT_MAP entries so that common raw column names (Tmax_C, Rainfall_mm, etc.)
# resolve to their canonical UAMS column names.
for raw_variant, canonical_col in VARIANT_MAP.items():
    raw_norm = re.sub(r"[^a-z0-9_]", "", raw_variant.lower().replace(" ", "_").replace("-", "_"))
    raw_norm_alt = raw_variant.lower().replace(" ", "_").replace("-", "_")
    if raw_norm not in _normalized_uams:
        _normalized_uams[raw_norm] = canonical_col
    if raw_norm_alt not in _normalized_uams:
        _normalized_uams[raw_norm_alt] = canonical_col


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def map_column(source: str, col_name: str) -> str | None:
    source_map = KNOWN_SOURCE_MAPS.get(source, {})
    if col_name in source_map:
        return source_map[col_name]
    norm = _normalise(col_name)
    if norm in source_map:
        return source_map[norm]
    if norm in _normalized_uams:
        return _normalized_uams[norm]
    best_match: str | None = None
    best_score = 0.0
    for uams_norm, uams_col in _normalized_uams.items():
        score = SequenceMatcher(None, norm, uams_norm).ratio()
        if score > best_score:
            best_score = score
            best_match = uams_col
    if best_score >= FUZZY_THRESHOLD:
        return best_match
    return None


def map_dataframe(source: str, df: pd.DataFrame, drop_unmapped: bool = True) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    unmapped: list[str] = []
    for col in df.columns:
        mapped = map_column(source, col)
        if mapped:
            renamed[col] = mapped
        else:
            unmapped.append(col)
    result = df.rename(columns=renamed)
    if drop_unmapped and unmapped:
        result = result.drop(columns=[c for c in unmapped if c in result.columns], errors="ignore")
    return result


def discover_new_columns(df: pd.DataFrame, existing_known: set[str]) -> dict[str, str]:
    new_cols: dict[str, str] = {}
    for col in df.columns:
        norm = _normalise(col)
        if norm not in existing_known:
            new_cols[col] = norm
    return new_cols
