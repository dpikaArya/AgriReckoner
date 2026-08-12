"""Phase 20 STEPS 6-8 — External predictor linkage (weather + soil).

Attaches already-acquired, approved Phase 18 external values (NASA POWER
climate, SoilGrids soil, CHIRPS/MapSPAM where available) to observations that
carry sufficient spatial and temporal information. Only project-local,
already-verified records are used — nothing is fabricated and no live API is
called. Every linked value retains full provenance.

Outputs:
    outputs/phase20/external_predictor_linkage.parquet  (long provenance table)
    outputs/phase20/external_predictor_matrix.parquet   (wide per-observation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_spatial import haversine_km, resolve_location

PREDICTOR_UAMS_COLUMNS = [
    "Average_Temperature",
    "Temperature_Max",
    "Temperature_Min",
    "Rainfall",
    "Humidity",
    "Wind_Speed",
    "Solar_Radiation",
    "Organic_Carbon",
    "Soil_pH",
    "Soil_Clay_Pct",
    "Soil_Sand_Pct",
    "Soil_Silt_Pct",
    "Soil_Bulk_Density",
    "Nitrogen",
    "CEC",
    "Soil_Moisture",
]

# staging UAMS_Column -> model matrix column
CANONICAL_PREDICTOR = {
    "Average_Temperature": "Temperature",
    "Temperature_Max": "Temperature_Max",
    "Temperature_Min": "Temperature_Min",
    "Rainfall": "Rainfall",
    "Humidity": "Humidity",
    "Wind_Speed": "Wind_Speed",
    "Solar_Radiation": "Solar_Radiation",
    "Organic_Carbon": "Organic_Carbon",
    "Soil_pH": "Soil_pH",
    "Soil_Clay_Pct": "Soil_Clay_Pct",
    "Soil_Sand_Pct": "Soil_Sand_Pct",
    "Soil_Silt_Pct": "Soil_Silt_Pct",
    "Soil_Bulk_Density": "Soil_Bulk_Density",
    "Nitrogen": "Nitrogen",
    "CEC": "CEC",
    "Soil_Moisture": "Soil_Moisture",
}


def _geo(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def build_predictor_records(cfg: dict) -> pd.DataFrame:
    stg = C.load_input("phase18.staging_external", cfg)
    stg = stg[stg["Is_Duplicate"] != True]  # noqa: E712
    stg = stg[stg["Validation_Status"].astype(str).str.upper() == "APPROVED"]
    recs = []
    for _, r in stg.iterrows():
        col = C.s(r.get("UAMS_Column"))
        if col not in PREDICTOR_UAMS_COLUMNS:
            continue
        lat, lon = _geo(r.get("Location_Lat")), _geo(r.get("Location_Lon"))
        if pd.isna(lat) or pd.isna(lon):
            continue
        recs.append(
            {
                "source": C.s(r.get("Source")),
                "dataset_id": C.s(r.get("Dataset_ID")),
                "source_record_id": C.s(r.get("Provenance_ID")),
                "uams_column": col,
                "predictor": CANONICAL_PREDICTOR[col],
                "value": r.get("Value"),
                "unit": C.s(r.get("Normalized_Unit")) or C.s(r.get("Unit")),
                "lat": lat,
                "lon": lon,
                "year": r.get("Year"),
                "country": C.s(r.get("Country")),
                "provenance": C.s(r.get("Provenance")),
                "license": C.s(r.get("License")),
                "spatial_grade": C.s(r.get("Spatial_Match_Grade")),
                "temporal_grade": C.s(r.get("Temporal_Match_Grade")),
                "aggregation_method": C.s(r.get("Aggregation_Method")),
                "retrieval_timestamp": None,  # not stored per record; see staging provenance
            }
        )
    return pd.DataFrame(recs)


def link_predictors(
    obs: pd.DataFrame, cfg: dict, spatial: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Long provenance table linking predictors to observations."""
    if spatial is None:
        spatial = resolve_location(obs, cfg)
    pred = build_predictor_records(cfg)
    if len(pred) == 0:
        return pd.DataFrame()

    exact_r = cfg["spatial"]["exact_match_radius_km"]
    station_r = cfg["spatial"]["station_match_radius_km"]
    rows = []
    for _, o in spatial.iterrows():
        lat, lon = _geo(o.get("latitude")), _geo(o.get("longitude"))
        if pd.isna(lat) or pd.isna(lon):
            continue
        for _, p in pred.iterrows():
            if pd.isna(p["lat"]) or pd.isna(p["lon"]):
                continue
            d = haversine_km(lat, lon, p["lat"], p["lon"])
            if pd.isna(d):
                continue
            method = "exact_coordinate_match"
            conf = 0.9
            if d > exact_r and d <= station_r:
                method = "station_radius_match"
                conf = 0.7
            elif d > station_r:
                continue
            rows.append(
                {
                    "ObservationID_ML": o.get("ObservationID_ML"),
                    "canonical_observation_id": o.get("canonical_observation_id"),
                    "canonical_location_id": o.get("canonical_location_id"),
                    "Crop": o.get("Crop"),
                    "Year": o.get("Year"),
                    "predictor": p["predictor"],
                    "value": p["value"],
                    "unit": p["unit"],
                    "source": p["source"],
                    "dataset": p["dataset_id"],
                    "source_record_id": p["source_record_id"],
                    "location_lat": p["lat"],
                    "location_lon": p["lon"],
                    "time_period": str(o.get("weather_period")) or str(p["year"]),
                    "retrieval_timestamp": None,
                    "linkage_method": method,
                    "linkage_confidence": conf,
                    "provenance": p["provenance"],
                    "license": p["license"],
                    "spatial_grade": p["spatial_grade"],
                    "temporal_grade": p["temporal_grade"],
                    "aggregation_method": p["aggregation_method"],
                }
            )
    return pd.DataFrame(rows)


def build_predictor_matrix(link: pd.DataFrame) -> pd.DataFrame:
    """Wide per-observation predictor matrix from the linkage table."""
    if len(link) == 0:
        return pd.DataFrame()
    wide = link.pivot_table(
        index=[
            "ObservationID_ML",
            "canonical_observation_id",
            "canonical_location_id",
            "Crop",
            "Year",
        ],
        columns="predictor",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def run(force: bool = False):
    cfg = C.load_config()

    from p20_identity import add_canonical_identity, build_observation_table

    obs = build_observation_table()
    obs = add_canonical_identity(obs)

    spatial = resolve_location(obs, cfg)
    link = link_predictors(obs, cfg, spatial)
    matrix = build_predictor_matrix(link)

    C.to_parquet(link, C.OUT / "external_predictor_linkage.parquet")
    C.to_parquet(matrix, C.OUT / "external_predictor_matrix.parquet")
    if len(link):
        C.to_excel(link, C.REPORTS / "external_predictor_linkage.xlsx", "Linkage")

    result = {
        "generated_at": C.now_full_iso(),
        "observations_with_predictor_link": int(link["ObservationID_ML"].nunique())
        if len(link)
        else 0,
        "linked_values": int(len(link)),
        "predictor_records_used": int(link["source_record_id"].nunique()) if len(link) else 0,
        "sources": sorted(link["source"].dropna().unique().tolist()) if len(link) else [],
        "predictors_linked": sorted(link["predictor"].dropna().unique().tolist())
        if len(link)
        else [],
        "no_fabricated_data": True,
        "no_live_api_calls": True,
    }
    C.write_json(result, C.OUT / "external_linkage_metrics.json")
    C.mark_done("external_linkage", result)
    C.log_msg(f"STEP6-8 external linkage: {result['linked_values']} values linked")
    return result


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
