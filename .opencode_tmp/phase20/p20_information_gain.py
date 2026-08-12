"""Phase 20 STEP 12 + 18 — Information gain and acquisition queue.

For every incomplete yield observation, scores the missing predictors by
expected information gain (how many observations they would complete, crop
and study diversity, source availability, measurement reliability, spatial
and temporal confidence, retrieval feasibility and cost).

Produces:
    outputs/phase20/phase20_information_gain.parquet        (per obs x variable)
    outputs/phase20/phase20_acquisition_priority.parquet    (per variable)

Approved connectors are those already used in Phase 18 (FAOSTAT, NASA_POWER,
SoilGrids, CHIRPS, MapSPAM). No new sources are invented.
"""

from __future__ import annotations

import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_predictors import _present, build_linked_observations

# which approved Phase 18 connector supplies which predictor
SOURCE_AVAILABILITY = {
    "Rainfall": ["CHIRPS", "NASA_POWER"],
    "Temperature": ["NASA_POWER"],
    "Temperature_Max": ["NASA_POWER"],
    "Temperature_Min": ["NASA_POWER"],
    "Soil_pH": ["SoilGrids"],
    "Organic_Carbon": ["SoilGrids"],
    "Nitrogen": ["SoilGrids"],
    "Phosphorus": [],
    "Potassium": [],
    "CEC": ["SoilGrids"],
    "Solar_Radiation": ["NASA_POWER"],
    "Humidity": ["NASA_POWER"],
    "Wind_Speed": ["NASA_POWER"],
    "Soil_Texture": ["SoilGrids"],
    "Soil_Moisture": [],
    "Plant_Population": [],
    "Season": [],
    "Location": [],
    "Year": [],
    "Crop": [],
    "Cultivar": [],
}

# measurement reliability by source (0-1)
SOURCE_RELIABILITY = {
    "CHIRPS": 0.9,
    "NASA_POWER": 0.9,
    "SoilGrids": 0.85,
    "FAOSTAT": 0.9,
    "MapSPAM": 0.7,
}

FEASIBILITY = {
    "Rainfall": 0.9,
    "Temperature": 0.9,
    "Temperature_Max": 0.9,
    "Temperature_Min": 0.9,
    "Soil_pH": 0.8,
    "Organic_Carbon": 0.8,
    "Nitrogen": 0.8,
    "Phosphorus": 0.3,
    "Potassium": 0.3,
    "CEC": 0.8,
    "Solar_Radiation": 0.85,
    "Humidity": 0.85,
    "Wind_Speed": 0.85,
    "Soil_Texture": 0.8,
    "Soil_Moisture": 0.3,
    "Plant_Population": 0.4,
    "Season": 0.5,
    "Location": 0.6,
    "Year": 0.9,
    "Crop": 0.9,
    "Cultivar": 0.5,
}

COST = {
    "Rainfall": 0.2,
    "Temperature": 0.2,
    "Temperature_Max": 0.2,
    "Temperature_Min": 0.2,
    "Soil_pH": 0.3,
    "Organic_Carbon": 0.3,
    "Nitrogen": 0.3,
    "Phosphorus": 0.5,
    "Potassium": 0.5,
    "CEC": 0.3,
    "Solar_Radiation": 0.2,
    "Humidity": 0.2,
    "Wind_Speed": 0.2,
    "Soil_Texture": 0.3,
    "Soil_Moisture": 0.5,
    "Plant_Population": 0.6,
    "Season": 0.6,
    "Location": 0.5,
    "Year": 0.1,
    "Crop": 0.1,
    "Cultivar": 0.5,
}


def _avail(row, col):
    return _present(row.get(col))


def compute_info_gain(obs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    igw = cfg["information_gain"]
    classes = cfg["readiness_classes"]
    mr = classes["MODEL_READY_min"]
    nr = classes["NEAR_READY_min"]
    required = [p for p in cfg["required_predictors"] if p != "Yield"]

    rows = []
    yield_obs = obs[obs["Target"] == "Yield"]
    for _, r in yield_obs.iterrows():
        for var in required:
            if _avail(r, var):
                continue
            spatial_conf = float(r.get("location_confidence") or 0.0)
            temporal_conf = float(r.get("temporal_confidence") or 0.0)
            ratio = float(r.get("CompletenessRatio") or 0.0)
            would_complete = ratio >= nr and ratio < mr  # crosses into MODEL_READY
            rows.append(
                {
                    "ObservationID_ML": r.get("ObservationID_ML"),
                    "canonical_observation_id": r.get("canonical_observation_id"),
                    "canonical_study_id": r.get("canonical_study_id"),
                    "Crop": r.get("Crop"),
                    "MissingPredictor": var,
                    "CurrentCompleteness": round(ratio, 4),
                    "WouldCompleteObservation": would_complete,
                    "SpatialConfidence": round(spatial_conf, 3),
                    "TemporalConfidence": round(temporal_conf, 3),
                }
            )
    ig_df = pd.DataFrame(rows)

    # aggregate per missing predictor
    agg_rows = []
    for var, grp in ig_df.groupby("MissingPredictor"):
        n_complete = int(grp["WouldCompleteObservation"].sum())
        n_missing = int(len(grp))
        crop_coverage = int(grp["Crop"].nunique())
        study_coverage = int(grp["canonical_study_id"].nunique())
        sources = SOURCE_AVAILABILITY.get(var, [])
        source_score = min(1.0, sum(SOURCE_RELIABILITY.get(s, 0.5) for s in sources) / 3.0)
        reliability = max([SOURCE_RELIABILITY.get(s, 0.5) for s in sources], default=0.3)
        feasibility = FEASIBILITY.get(var, 0.5)
        cost = COST.get(var, 0.5)
        near_boost = igw["near_ready_boost"] if n_complete else 0.0

        info_gain = (
            igw["completion_weight"] * min(1.0, n_complete / max(n_missing, 1))
            + igw["crop_coverage_weight"] * min(1.0, crop_coverage / max(10, 1))
            + igw["study_diversity_weight"] * min(1.0, study_coverage / max(10, 1))
            + igw["source_weight"] * source_score
            + igw["reliability_weight"] * reliability
            + igw["spatial_weight"] * float(grp["SpatialConfidence"].mean())
            + igw["temporal_weight"] * float(grp["TemporalConfidence"].mean())
            + igw["feasibility_weight"] * feasibility
            + igw["cost_weight"] * (1.0 - cost)
            + near_boost * 0.05
        )
        priority = (
            "HIGH" if n_complete > 0 else ("MEDIUM" if n_missing > len(yield_obs) / 2 else "LOW")
        )
        agg_rows.append(
            {
                "MissingPredictor": var,
                "ObservationsMissing": n_missing,
                "ObservationsCompletedIfAvailable": n_complete,
                "CropCoverage": crop_coverage,
                "StudyCoverage": study_coverage,
                "SourceAvailability": ";".join(sources) if sources else "none_approved",
                "MeasurementReliability": round(reliability, 3),
                "RetrievalFeasibility": round(feasibility, 3),
                "Cost": round(cost, 3),
                "InfoGain": round(info_gain, 4),
                "AcquisitionPriority": priority,
            }
        )
    agg = pd.DataFrame(agg_rows)
    if len(agg):
        agg = agg.sort_values("InfoGain", ascending=False).reset_index(drop=True)
    return ig_df, agg


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)
    ig_df, agg = compute_info_gain(obs, cfg)

    C.to_parquet(ig_df, C.OUT / "phase20_information_gain.parquet")
    C.to_parquet(agg, C.OUT / "phase20_acquisition_priority.parquet")
    C.to_excel(agg, C.REPORTS / "information_gain_report.xlsx", "InfoGain")
    C.to_excel(agg, C.REPORTS / "acquisition_priority.xlsx", "Acquisition")

    result = {
        "generated_at": C.now_full_iso(),
        "missing_observations_scored": int(len(ig_df)),
        "variables_evaluated": int(len(agg)),
        "observations_completable": int(ig_df["WouldCompleteObservation"].sum()),
        "highest_gain_variables": agg.head(10)["MissingPredictor"].tolist() if len(agg) else [],
        "highest_gain_sources": sorted(
            {
                s
                for v in (agg.head(10)["SourceAvailability"].tolist() if len(agg) else [])
                for s in v.split(";")
                if s != "none_approved"
            }
        ),
        "priority_counts": agg["AcquisitionPriority"].value_counts().to_dict() if len(agg) else {},
        "note": "acquisition prioritizes variables that complete existing yield observations",
    }
    C.write_json(result, C.OUT / "information_gain_metrics.json")
    C.mark_done("information_gain", result)
    C.log_msg(f"STEP12/18 infogain: {result['observations_completable']} observations completable")
    return result


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
