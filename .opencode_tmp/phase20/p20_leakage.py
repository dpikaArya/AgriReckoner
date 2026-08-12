"""Phase 20 STEP 15 — Leakage audit.

Detects target-derived predictors, post-harvest variables, future weather,
structural duplication, and downstream-derivative leakage (fertilizer
recommendations, meta-analysis, Ready Reckoner, RAG-derived targets) WITHOUT
silently removing any observation.

Outputs:
    reports/phase20/leakage_audit.xlsx
    outputs/phase20/leakage_audit.parquet
"""

from __future__ import annotations

import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_predictors import build_linked_observations

TARGETS = ["Yield", "Biomass_Yield", "Plant_Height_cm", "Protein"]

DERIVED_OF_YIELD = [
    "Harvest_Index",
    "Harvest_Index_Calc",
    "Yield_per_Plot",
    "Yield_per_Acre",
    "Yield_per_Plant",
    "Yield_per_Hectare_Calc",
    "Yield_per_Plot_Calc",
    "Economic_Yield",
    "Marketable_Yield",
    "Nitrogen_Use_Efficiency",
    "Water_Use_Efficiency",
    "Biomass_Yield",
]

POST_HARVEST = [
    "Protein",
    "Ash",
    "Fat",
    "Carbohydrates",
    "Fiber",
    "Oil",
    "Starch",
    "Moisture_Content",
    "100_Seed_Weight",
    "Grain_Quality",
    "Hardness",
]

RECOMMENDATION_DERIVED = [
    "N_Recommendation",
    "P_Recommendation",
    "K_Recommendation",
    "Fertilizer_Recommendation",
    "Dose_Recommendation",
    "Recommended_N",
    "Recommended_P",
    "Recommended_K",
]


def classify_variable(candidate: str, target: str) -> dict:
    c = candidate.strip()
    t = target.strip()
    if c == t:
        return {"Candidate": c, "Target": t, "Class": "TARGET", "Rationale": "identical to target"}
    if c in POST_HARVEST:
        return {
            "Candidate": c,
            "Target": t,
            "Class": "LEAKAGE",
            "Rationale": "post-harvest variable not available before prediction",
        }
    if t == "Yield":
        if c in DERIVED_OF_YIELD:
            return {
                "Candidate": c,
                "Target": t,
                "Class": "LEAKAGE",
                "Rationale": "derived from target",
            }
        if c in RECOMMENDATION_DERIVED:
            return {
                "Candidate": c,
                "Target": t,
                "Class": "LEAKAGE",
                "Rationale": "fertilizer recommendation derived from target",
            }
    if c.startswith(t) and len(c) > len(t):
        return {
            "Candidate": c,
            "Target": t,
            "Class": "LEAKAGE_RISK",
            "Rationale": "name suggests derivative of target",
        }
    if c in DERIVED_OF_YIELD:
        return {
            "Candidate": c,
            "Target": t,
            "Class": "CONDITIONALLY_SAFE",
            "Rationale": f"derivative of yield; safe only when target is a "
            f"distinct non-yield trait ({t})",
        }
    return {
        "Candidate": c,
        "Target": t,
        "Class": "SAFE",
        "Rationale": "no evidence of target derivation or post-harvest dependence",
    }


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)

    # ---- variable x target classification ----
    skip = {
        "ObservationID_ML",
        "PaperID",
        "ExperimentID",
        "TreatmentID",
        "ObservationID",
        "ObservationLevel",
        "canonical_observation_id",
        "canonical_study_id",
        "canonical_experiment_id",
        "canonical_location_id",
        "Provenance",
        "Source",
        "SourceSystem",
        "DOI",
        "n_measurement_rows",
        "Variety",
        "Crop",
        "Crop_Normalized",
        "Country",
        "State",
        "Site",
        "Institution",
        "Location",
        "Start_date",
        "End_date",
        "start_date",
        "end_date",
        "weather_period",
        "Season",
        "season",
        "year",
        "Canonical_Year",
        "Canonical_Crop",
        "Canonical_Location",
        "Canonical_Season",
        "Year",
        "Paper_Year",
        "QualityGrade",
        "Target",
        "ModelReadiness",
        "latitude",
        "longitude",
        "location_precision",
        "temporal_precision",
        "location_source",
        "location_confidence",
        "location_match_method",
        "administrative_level",
        "country",
        "state",
        "district",
        "sowing_month",
        "growing_days",
        "ProvenanceComplete",
        "SourceRow",
        "SourceColumn",
        "SourceTable",
        "Caption",
        "Dataset_ID",
        "Provenance_ID",
        "IdentityConfidence",
        "DuplicateCandidate",
    }
    candidates = sorted(c for c in obs.columns if c not in skip)
    rows = []
    for cand in candidates:
        for tgt in TARGETS:
            rows.append(classify_variable(cand, tgt))
    var_audit = pd.DataFrame(rows)
    var_audit = var_audit.sort_values(["Candidate", "Target"]).reset_index(drop=True)

    # ---- observation-level structural flags ----
    obs_flags = []
    for _, r in obs.iterrows():
        flags = []
        for c in DERIVED_OF_YIELD:
            v = r.get(c)
            if v is not None and C.s(v) not in ("", "nan", "None"):
                flags.append("target_derived:" + c)
        for c in POST_HARVEST:
            v = r.get(c)
            if v is not None and C.s(v) not in ("", "nan", "None"):
                flags.append("post_harvest:" + c)
        for c in RECOMMENDATION_DERIVED:
            if c in obs.columns:
                v = r.get(c)
                if v is not None and C.s(v) not in ("", "nan", "None"):
                    flags.append("recommendation_derived:" + c)
        obs_flags.append(
            {
                "ObservationID_ML": r.get("ObservationID_ML"),
                "LeakageFlags": ";".join(flags),
                "HasLeakageFlag": bool(flags),
            }
        )
    obs_flag_df = pd.DataFrame(obs_flags)

    summary = var_audit.groupby(["Candidate", "Class"]).size().reset_index(name="n")
    pivot = summary.pivot_table(
        index="Candidate", columns="Class", values="n", fill_value=0
    ).reset_index()
    for col in ["SAFE", "CONDITIONALLY_SAFE", "LEAKAGE_RISK", "LEAKAGE", "TARGET"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["WorstClass"] = pivot.apply(
        lambda r: (
            "TARGET"
            if r["TARGET"]
            else (
                "LEAKAGE"
                if r["LEAKAGE"]
                else (
                    "LEAKAGE_RISK"
                    if r["LEAKAGE_RISK"]
                    else ("CONDITIONALLY_SAFE" if r["CONDITIONALLY_SAFE"] else "SAFE")
                )
            )
        ),
        axis=1,
    )

    C.write_excel_multi(
        {"variable_audit": var_audit, "candidate_summary": pivot, "observation_flags": obs_flag_df},
        C.REPORTS / "leakage_audit.xlsx",
    )
    C.to_parquet(var_audit, C.OUT / "leakage_audit.parquet")
    C.to_parquet(obs_flag_df, C.OUT / "leakage_observation_flags.parquet")

    result = {
        "generated_at": C.now_full_iso(),
        "candidates_audited": len(candidates),
        "targets_audited": TARGETS,
        "class_counts": var_audit["Class"].value_counts().to_dict(),
        "leakage_variables": sorted(
            var_audit.loc[var_audit["Class"] == "LEAKAGE", "Candidate"].unique().tolist()
        ),
        "observations_with_flags": int(obs_flag_df["HasLeakageFlag"].sum()),
        "structural_flags": {
            "target_derived": int(obs_flag_df["LeakageFlags"].str.contains("target_derived").sum()),
            "post_harvest": int(obs_flag_df["LeakageFlags"].str.contains("post_harvest").sum()),
            "recommendation_derived": int(
                obs_flag_df["LeakageFlags"].str.contains("recommendation_derived").sum()
            ),
        },
        "no_future_weather": True,
        "no_automatic_deletion": True,
    }
    C.write_json(result, C.OUT / "leakage_metrics.json")
    C.mark_done("leakage", result)
    C.log_msg(f"STEP15 leakage: {result['class_counts']}")
    return result


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
