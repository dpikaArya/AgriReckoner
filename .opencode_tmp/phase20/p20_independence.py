"""Phase 20 STEP 14 — Study independence.

Counts observations per study/experiment/location/paper/treatment and detects
repeated measurements, locations and cultivars. Creates independent grouping
identifiers at observation, treatment, experiment, study and location level.

Outputs:
    outputs/phase20/independence_audit.parquet
"""

from __future__ import annotations

import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_predictors import build_linked_observations


def compute_independence(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()

    def _count(col):
        return out.groupby(col)["ObservationID_ML"].transform("count")

    out["observations_per_study"] = _count("canonical_study_id")
    out["observations_per_experiment"] = _count("canonical_experiment_id")
    out["observations_per_location"] = _count("canonical_location_id")
    out["observations_per_paper"] = _count("PaperID")
    out["observations_per_treatment"] = _count("TreatmentID")
    out["repeated_measurements"] = out["observations_per_treatment"]
    out["repeated_locations"] = out["observations_per_location"]
    out["repeated_cultivars"] = out.groupby("Crop_Normalized")["ObservationID_ML"].transform(
        "count"
    )

    # independent grouping identifiers
    out["IndependentGroup_Observation"] = out["canonical_observation_id"]
    out["IndependentGroup_Treatment"] = (
        out["canonical_experiment_id"].astype(str) + "::" + out["TreatmentID"].astype(str)
    )
    out["IndependentGroup_Experiment"] = out["canonical_experiment_id"]
    out["IndependentGroup_Study"] = out["canonical_study_id"]
    out["IndependentGroup_Location"] = out["canonical_location_id"]
    return out


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)
    ind = compute_independence(obs)

    keep = [
        "ObservationID_ML",
        "canonical_observation_id",
        "canonical_study_id",
        "canonical_experiment_id",
        "canonical_location_id",
        "PaperID",
        "ExperimentID",
        "TreatmentID",
        "Crop",
        "Crop_Normalized",
        "year",
        "observations_per_study",
        "observations_per_experiment",
        "observations_per_location",
        "observations_per_paper",
        "observations_per_treatment",
        "repeated_measurements",
        "repeated_locations",
        "repeated_cultivars",
        "IndependentGroup_Observation",
        "IndependentGroup_Treatment",
        "IndependentGroup_Experiment",
        "IndependentGroup_Study",
        "IndependentGroup_Location",
    ]
    rep = ind[[c for c in keep if c in ind.columns]].copy()

    C.to_parquet(rep, C.OUT / "independence_audit.parquet")
    C.to_excel(rep, C.REPORTS / "independence_report.xlsx", "Independence")

    result = {
        "generated_at": C.now_full_iso(),
        "observations": int(len(ind)),
        "independent_studies": int(ind["canonical_study_id"].nunique()),
        "independent_experiments": int(ind["canonical_experiment_id"].nunique()),
        "independent_locations": int(ind["canonical_location_id"].nunique()),
        "independent_papers": int(ind["PaperID"].nunique()),
        "independent_treatments": int(ind["IndependentGroup_Treatment"].nunique()),
        "max_obs_per_study": int(ind["observations_per_study"].max()),
        "max_obs_per_location": int(ind["observations_per_location"].max()),
        "max_obs_per_paper": int(ind["observations_per_paper"].max()),
        "repeated_location_count": int((ind["observations_per_location"] > 1).sum()),
        "repeated_cultivar_count": int((ind["repeated_cultivars"] > 1).sum()),
    }
    C.write_json(result, C.OUT / "independence_metrics.json")
    C.mark_done("independence", result)
    C.log_msg(
        f"STEP14 independence: {result['independent_studies']} studies, "
        f"{result['independent_locations']} locations"
    )
    return result


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
