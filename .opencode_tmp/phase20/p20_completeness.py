"""Phase 20 STEP 11 — Observation-level missingness.

Produces outputs/phase20/observation_missingness.parquet, the primary Phase 20
diagnostic. Every row is one observation with its target, predictor counts,
completeness, spatial/temporal completeness, quality grade and model
readiness.

Primary metric driver: independent complete validated yield observations per
crop with sufficient predictor coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_predictors import build_linked_observations, _present

MODEL_READY_COLS = [
    "ObservationID_ML", "canonical_observation_id", "canonical_study_id",
    "canonical_experiment_id", "canonical_location_id", "PaperID",
    "ExperimentID", "TreatmentID", "ObservationID", "Crop", "Crop_Normalized",
    "Canonical_Year", "year", "Season", "season", "Location", "country",
    "latitude", "longitude", "location_precision", "temporal_precision",
    "Yield", "Biomass_Yield", "Plant_Height_cm", "Protein",
]


def build_missingness(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = obs.copy()
    classes = cfg["readiness_classes"]
    mr = classes["MODEL_READY_min"]

    out["SpatialCompleteness"] = out["location_precision"].map({
        "exact_coordinates": 1.0, "experimental_station": 0.9,
        "study_location": 0.8, "district": 0.6, "country_only": 0.4,
        "unknown": 0.0}).fillna(0.0)
    out["TemporalCompleteness"] = out["temporal_precision"].map({
        "growing_season": 1.0, "year_level": 0.6, "unknown": 0.0}).fillna(0.0)

    def _model_readiness(r):
        if r["Target"] != "Yield":
            return "N/A"
        if r["CompletenessRatio"] >= mr and r["SpatialCompleteness"] > 0.4:
            return "MODEL_READY"
        if r["CompletenessRatio"] >= classes["NEAR_READY_min"]:
            return "NEAR_READY"
        return "NOT_READY"

    out["ModelReadiness"] = out.apply(_model_readiness, axis=1)

    keep = MODEL_READY_COLS + [
        "PredictorCount", "AvailablePredictorCount", "MissingPredictorCount",
        "CompletenessRatio", "PredictorCompleteness_TierA",
        "PredictorCompleteness_TierB", "PredictorCompleteness_TierC",
        "MissingCriticalPredictors", "MissingPredictors", "QualityGrade",
        "SpatialCompleteness", "TemporalCompleteness", "Target", "ModelReadiness"]
    return out[[c for c in keep if c in out.columns]].copy()


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)
    miss = build_missingness(obs, cfg)

    C.to_parquet(miss, C.OUT / "observation_missingness.parquet")
    C.to_excel(miss, C.REPORTS / "observation_missingness.xlsx", "Missingness")

    yield_obs = miss[miss["Target"] == "Yield"]
    n_yield = len(yield_obs)
    mr = cfg["readiness_classes"]["MODEL_READY_min"]
    nr = cfg["readiness_classes"]["NEAR_READY_min"]

    result = {
        "generated_at": C.now_full_iso(),
        "observations_total": int(len(miss)),
        "yield_observations": int(n_yield),
        "yield_model_ready": int((yield_obs["ModelReadiness"] == "MODEL_READY").sum()),
        "yield_near_ready": int((yield_obs["ModelReadiness"] == "NEAR_READY").sum()),
        "yield_not_ready": int((yield_obs["ModelReadiness"] == "NOT_READY").sum()),
        "yield_80pc": int((yield_obs["CompletenessRatio"] >= mr).sum()),
        "yield_90pc": int((yield_obs["CompletenessRatio"] >= 0.90).sum()),
        "yield_100pc": int((yield_obs["CompletenessRatio"] >= 1.0).sum()),
        "avg_completeness_yield": round(float(yield_obs["CompletenessRatio"].mean()), 4),
        "missing_critical_predictors": sorted(yield_obs["MissingCriticalPredictors"].map(
            lambda s: s.split(";") if s else []).explode().dropna().unique().tolist()),
        "complete_observations_per_crop": (
            yield_obs[yield_obs["CompletenessRatio"] >= mr]
            .groupby("Crop_Normalized").size().to_dict()),
    }
    C.write_json(result, C.OUT / "missingness_metrics.json")
    C.mark_done("completeness", result)
    C.log_msg(f"STEP11 missingness: {n_yield} yield observations, "
              f"{result['yield_model_ready']} model-ready")
    return result


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
