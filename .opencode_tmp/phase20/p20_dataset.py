"""Phase 20 STEP 13/16/17 — Model-ready matrix, validation splits, readiness.

STEP 13  model_ready_observation_matrix.parquet  (one row per observation)
STEP 16  grouped validation splits (study/location/crop-out/temporal)
STEP 17  per-target model readiness

Nothing here modifies UAMS_v2 or UAMS_v2.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

from p20_predictors import build_linked_observations

TARGET_DOMAIN_COLUMNS = {
    "yield": ["Yield"],
    "biomass": ["Biomass_Yield"],
    "growth": ["Plant_Height_cm"],
    "quality": ["Protein"],
    "nutrient": [],
}


def _has_sklearn():
    try:
        from sklearn.model_selection import GroupKFold
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# STEP 13 - model matrix
# ---------------------------------------------------------------------------

def build_model_matrix(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = obs.copy()
    out = out.sort_values("ObservationID_ML").reset_index(drop=True)

    base = [
        "ObservationID_ML", "canonical_observation_id", "canonical_study_id",
        "canonical_experiment_id", "canonical_location_id",
        "PaperID", "ExperimentID", "TreatmentID", "ObservationID",
        "ObservationLevel", "SourceSystem",
    ]
    meta = [
        "Crop", "Crop_Normalized", "Location", "country", "state", "district",
        "latitude", "longitude", "location_precision", "location_source",
        "location_confidence", "location_match_method", "year", "season",
        "season_Normalized", "Season", "start_date", "end_date",
        "temporal_precision", "temporal_confidence",
    ]
    predictors = [
        "Rainfall", "Temperature", "Temperature_Max", "Temperature_Min",
        "Soil_pH", "Organic_Carbon", "Nitrogen", "Phosphorus", "Potassium",
        "Solar_Radiation", "Soil_Moisture", "Soil_Texture", "Humidity",
        "Cultivar", "Plant_Population", "Soil_EC", "Irrigation",
        "Fertilizer_Type", "N_Dose", "P_Dose", "K_Dose", "Days_to_Maturity",
    ]
    targets = ["Yield", "Biomass_Yield", "Plant_Height_cm", "Protein"]
    completeness = [
        "PredictorCount", "AvailablePredictorCount", "MissingPredictorCount",
        "CompletenessRatio", "QualityGrade", "Target",
    ]
    prov = ["Source", "Provenance", "DOI"]

    cols = []
    for group in (base, meta, predictors, targets, completeness, prov):
        for c in group:
            if c in out.columns and c not in cols:
                cols.append(c)
    return out[cols].copy()


# ---------------------------------------------------------------------------
# STEP 16 - validation splits
# ---------------------------------------------------------------------------

def study_grouped_split(df, n_splits=5):
    from sklearn.model_selection import GroupKFold
    groups = df["canonical_study_id"].values
    kf = GroupKFold(n_splits=min(n_splits, df["canonical_study_id"].nunique()))
    return [(tr.tolist(), te.tolist()) for tr, te in kf.split(df.index, groups=groups)]


def location_grouped_split(df, n_splits=5):
    from sklearn.model_selection import GroupKFold
    groups = df["canonical_location_id"].values
    kf = GroupKFold(n_splits=min(n_splits, df["canonical_location_id"].nunique()))
    return [(tr.tolist(), te.tolist()) for tr, te in kf.split(df.index, groups=groups)]


def leave_one_crop_out(df):
    return [(df[df["Crop_Normalized"] != c].index.tolist(),
             df[df["Crop_Normalized"] == c].index.tolist())
            for c in df["Crop_Normalized"].unique()]


def temporal_split(df):
    years = sorted(df["year"].dropna().unique())
    if len(years) < 2:
        return []
    split_at = years[len(years) // 2]
    train = df[df["year"] <= split_at].index.tolist()
    test = df[df["year"] > split_at].index.tolist()
    if not train or not test:
        return []
    return [(train, test)]


def build_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    if _has_sklearn() and len(df) >= 5:
        gs = study_grouped_split(df, cfg["splitting"]["n_splits"])
        rows.append({"Strategy": "Study GroupKFold", "Splits": len(gs),
                     "Groups": "canonical_study_id",
                     "MinTrain": min(len(t) for t, _ in gs),
                     "MinTest": min(len(t) for _, t in gs),
                     "SameGroupBothSides": False})
        ls = location_grouped_split(df, cfg["splitting"]["n_splits"])
        rows.append({"Strategy": "Location GroupKFold", "Splits": len(ls),
                     "Groups": "canonical_location_id",
                     "MinTrain": min(len(t) for t, _ in ls),
                     "MinTest": min(len(t) for _, t in ls),
                     "SameGroupBothSides": False})
    else:
        rows.append({"Strategy": "Study GroupKFold", "Splits": 0,
                     "Groups": "canonical_study_id", "MinTrain": 0, "MinTest": 0,
                     "SameGroupBothSides": False})
        rows.append({"Strategy": "Location GroupKFold", "Splits": 0,
                     "Groups": "canonical_location_id", "MinTrain": 0, "MinTest": 0,
                     "SameGroupBothSides": False})

    if len(df) and df["Crop_Normalized"].nunique() >= 2:
        lo = leave_one_crop_out(df)
        rows.append({"Strategy": "Leave One Crop Out", "Splits": len(lo),
                     "Groups": "Crop_Normalized",
                     "MinTrain": min(len(t) for t, _ in lo),
                     "MinTest": min(len(t) for _, t in lo),
                     "SameGroupBothSides": False})
    else:
        rows.append({"Strategy": "Leave One Crop Out", "Splits": 0,
                     "Groups": "Crop_Normalized", "MinTrain": 0, "MinTest": 0,
                     "SameGroupBothSides": False})

    if len(df) and df["year"].notna().nunique() >= cfg["splitting"]["min_temporal_years"]:
        ts = temporal_split(df)
        rows.append({"Strategy": "Temporal split", "Splits": len(ts),
                     "Groups": "year",
                     "MinTrain": len(ts[0][0]) if ts else 0,
                     "MinTest": len(ts[0][1]) if ts else 0,
                     "SameGroupBothSides": False})
    else:
        rows.append({"Strategy": "Temporal split", "Splits": 0,
                     "Groups": "year", "MinTrain": 0, "MinTest": 0,
                     "SameGroupBothSides": False})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STEP 17 - model readiness
# ---------------------------------------------------------------------------

def model_readiness(matrix: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    req = cfg["minimum_requirements"]
    classes = cfg["readiness_classes"]
    mr = classes["MODEL_READY_min"]

    rows = []
    for domain, cols in TARGET_DOMAIN_COLUMNS.items():
        if not cols:
            rows.append(_readiness_row(domain, matrix, cols, cfg, empty=True))
            continue
        rows.append(_readiness_row(domain, matrix, cols, cfg))
    ready_df = pd.DataFrame(rows)

    # overall score = weighted average across domains (yield weighted highest)
    score = 0.0
    weight_sum = 0.0
    for r in rows:
        w = {"yield": 0.5, "biomass": 0.2, "growth": 0.15, "quality": 0.1,
             "nutrient": 0.05}.get(r["domain"], 0.1)
        score += w * r["readiness_score"]
        weight_sum += w
    overall = round(score / weight_sum, 4) if weight_sum else 0.0

    result = {
        "generated_at": C.now_full_iso(),
        "overall_readiness_score": overall,
        "retrain_allowed": bool(all(r["gates_pass"] for r in rows)),
        "per_target": {r["domain"]: {k: v for k, v in r.items()
                                     if k != "domain"} for r in rows},
    }
    return result, ready_df


def _readiness_row(domain, matrix, cols, cfg, empty=False) -> dict:
    req = cfg["minimum_requirements"]
    mr = cfg["readiness_classes"]["MODEL_READY_min"]
    nr = cfg["readiness_classes"]["NEAR_READY_min"]

    if empty:
        return {"domain": domain, "total_observations": 0, "independent_studies": 0,
                "independent_locations": 0, "complete_observations": 0,
                "p80_coverage": 0, "p90_coverage": 0, "p100_coverage": 0,
                "crop_coverage": 0, "leakage_safe_samples": 0,
                "effective_sample_size": 0, "gates_pass": False,
                "readiness_score": 0.0, "notes": "no target column available"}

    cols = [c for c in cols if c in matrix.columns]
    if not cols:
        return {"domain": domain, "total_observations": 0, "independent_studies": 0,
                "independent_locations": 0, "complete_observations": 0,
                "p80_coverage": 0, "p90_coverage": 0, "p100_coverage": 0,
                "crop_coverage": 0, "leakage_safe_samples": 0,
                "effective_sample_size": 0, "gates_pass": False,
                "readiness_score": 0.0, "notes": "target column absent in matrix"}

    m = matrix.copy()
    m["_has_target"] = m[cols].apply(lambda r: r.notna().any(), axis=1)
    sub = m[m["_has_target"]]
    n = int(len(sub))
    n_studies = int(sub["canonical_study_id"].nunique())
    n_locs = int(sub["canonical_location_id"].nunique())
    n_crops = int(sub["Crop_Normalized"].replace("", np.nan).nunique())
    comp = sub["CompletenessRatio"] if "CompletenessRatio" in sub.columns else pd.Series(0.0, index=sub.index)
    n_complete = int((comp >= 1.0).sum())
    n_80 = int((comp >= mr).sum())
    n_90 = int((comp >= 0.90).sum())

    # leakage safe: no post-harvest/target-derived predictors in the row
    leak_safe = 0
    lf_p = C.OUT / "leakage_observation_flags.parquet"
    if lf_p.exists():
        flags = pd.read_parquet(lf_p)
        safe_ids = set(flags.loc[~flags["HasLeakageFlag"], "ObservationID_ML"])
        leak_safe = int(sub["ObservationID_ML"].isin(safe_ids).sum())
    else:
        leak_safe = n

    eff_n = min(n, n_studies * 2, n_locs * 3)
    gates = (
        n_studies >= req["min_independent_studies"]
        and n_locs >= req["min_independent_locations"]
        and n >= req["min_observations"]
        and n_80 >= 1
        and (comp.mean() if len(comp) else 0) >= req["min_predictor_completeness"]
        and n_crops >= req["min_crops"]
    )
    # score: mix of volume, diversity, completeness and leakage safety
    score = (
        0.25 * min(1.0, n / max(req["min_observations"], 1)) +
        0.20 * min(1.0, n_studies / max(req["min_independent_studies"], 1)) +
        0.15 * min(1.0, n_locs / max(req["min_independent_locations"], 1)) +
        0.15 * (comp.mean() if len(comp) else 0.0) +
        0.10 * min(1.0, n_crops / max(req["min_crops"], 1)) +
        0.15 * min(1.0, leak_safe / max(n, 1))
    )
    return {
        "domain": domain, "total_observations": n, "independent_studies": n_studies,
        "independent_locations": n_locs, "complete_observations": n_complete,
        "p80_coverage": n_80, "p90_coverage": n_90, "p100_coverage": n_complete,
        "crop_coverage": n_crops, "study_coverage": n_studies,
        "location_coverage": n_locs, "leakage_safe_samples": leak_safe,
        "min_samples_per_crop": int(sub["Crop_Normalized"].replace("", np.nan)
                                    .value_counts().min()) if n_crops else 0,
        "effective_sample_size": eff_n, "gates_pass": bool(gates),
        "readiness_score": round(score, 4),
        "notes": "MODEL_READY requires >=80% Tier A predictor coverage" if n_80 else
                 "no observation reaches 80% Tier A coverage",
    }


# ---------------------------------------------------------------------------
# STEP 19 - ready reckoner impact
# ---------------------------------------------------------------------------

def ready_reckoner_impact(cfg: dict) -> dict:
    """Quantify how much of the dataset is backed by phase14 ready-reckoner
    knowledge (evidence-only; does not augment observations)."""
    rr_p = C.PROJECT_ROOT / "outputs" / "phase14" / "ready_reckoner_knowledge_base.parquet"
    impact = {
        "generated_at": C.now_full_iso(),
        "ready_reckoner_available": rr_p.exists(),
        "ready_reckoner_rows": int(len(pd.read_parquet(rr_p))) if rr_p.exists() else 0,
        "augments_observations": False,
        "note": "ready reckoner is evidence knowledge; no synthetic rows added",
    }
    C.write_json(impact, C.OUT / "ready_reckoner_impact.json")
    return impact


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)
    matrix = build_model_matrix(obs, cfg)

    C.to_parquet(matrix, C.OUT / "model_ready_observation_matrix.parquet")
    C.to_excel(matrix, C.REPORTS / "model_readiness.xlsx", "Matrix", )

    splits = build_splits(matrix, cfg)
    C.to_parquet(splits, C.OUT / "validation_splits.parquet")
    C.to_excel(splits, C.REPORTS / "validation_splits.xlsx", "Splits")

    readiness, ready_df = model_readiness(matrix, cfg)
    C.write_json(readiness, C.OUT / "model_readiness.json")
    C.to_parquet(ready_df, C.OUT / "model_readiness_per_target.parquet")

    rr = ready_reckoner_impact(cfg)

    result = {
        "generated_at": C.now_full_iso(),
        "matrix_observations": int(len(matrix)),
        "ready_reckoner_impact": rr,
        "splits": splits.to_dict("records"),
        "readiness": {k: readiness[k] for k in
                      ("overall_readiness_score", "retrain_allowed")},
        "per_target": readiness["per_target"],
    }
    C.write_json(result, C.OUT / "dataset_metrics.json")
    C.mark_done("dataset", result)
    C.log_msg(f"STEP13/16/17 dataset: matrix {len(matrix)} rows, "
              f"readiness {readiness['overall_readiness_score']}")
    return result


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
