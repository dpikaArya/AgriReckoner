"""Phase 20 STEPS 9-10 — Crop/season alignment and predictor completeness.

Builds the single linked observation table (identity + spatial + temporal +
external predictors), normalizes crop/season/cultivar through the Phase 17
ontology (preserving original values), and computes per-observation predictor
completeness against the Phase 20 predictor tiers.

Tier definitions (config/phase20.yaml):
    TIER A   Yield, Crop, Location, Year, Season, Rainfall, Temperature,
             Soil_pH, Nitrogen, Phosphorus, Potassium
    TIER B   Solar_Radiation, Organic_Carbon, Soil_Moisture, Texture,
             Humidity, Cultivar, Plant_Population
    TIER C   Soil_EC, Irrigation, Fertilizer_Type, N_Dose, P_Dose, K_Dose,
             Plant_Height_cm, Days_to_Maturity

For a yield observation the target column "Yield" is not counted as a
predictor (it is the target). Tier B/C are never required for basic
readiness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

try:
    from .p20_identity import add_canonical_identity, build_observation_table
    from .p20_spatial import resolve_location
    from .p20_temporal import resolve_temporal
except ImportError:
    from p20_identity import add_canonical_identity, build_observation_table
    from p20_spatial import resolve_location
    from p20_temporal import resolve_temporal

LINKED_CACHE = C.TMP / "phase20_linked_obs.parquet"


def _present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and (np.isnan(v)):
        return False
    if isinstance(v, np.floating) and np.isnan(v):
        return False
    s = C.s(v)
    if s in ("", "nan", "None", "NaN", "none"):
        return False
    try:
        f = float(s)
        if not np.isfinite(f):
            return False
    except (TypeError, ValueError):
        pass
    return True


def build_linked_observations(cfg: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    cfg = cfg or C.load_config()
    if use_cache and LINKED_CACHE.exists():
        return pd.read_parquet(LINKED_CACHE)

    obs = build_observation_table()
    obs = add_canonical_identity(obs)
    obs = resolve_location(obs, cfg)
    obs = resolve_temporal(obs, cfg)

    # merge external predictor matrix
    mat_p = C.OUT / "external_predictor_matrix.parquet"
    if mat_p.exists():
        mat = pd.read_parquet(mat_p)
        if len(mat):
            keys = [k for k in ("ObservationID_ML", "Crop", "Year") if k in mat.columns]
            mat = mat.drop_duplicates(subset="ObservationID_ML")
            obs = obs.merge(
                mat.drop(columns=[k for k in keys if k != "ObservationID_ML"]),
                on="ObservationID_ML",
                how="left",
                suffixes=("", "_ext"),
            )

    obs = _normalize_variables(obs, cfg)
    obs = _canonicalize_targets(obs)
    obs = compute_predictor_completeness(obs, cfg)
    obs.to_parquet(LINKED_CACHE, index=False)
    return obs


# ---------------------------------------------------------------------------
# variable alignment (STEP 9)
# ---------------------------------------------------------------------------


def _normalize_variables(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = obs.copy()
    aliases = _load_crop_aliases(cfg)

    def _norm_crop(v):
        s = C.s(v)
        if not s:
            return ""
        return aliases.get(s.lower(), s)

    if "Crop" in out.columns:
        out["Crop_Normalized"] = out["Crop"].map(_norm_crop)
        out["Crop_Normalized"] = out["Crop_Normalized"].fillna("")

    if "Season" in out.columns:
        out["Season_Normalized"] = out["Season"].map(
            lambda v: C.s(v) if C.s(v).lower() not in ("", "nan", "none") else ""
        )
    else:
        out["Season_Normalized"] = ""

    if "Variety" in out.columns:
        out["Cultivar"] = out["Variety"]
    elif "Cultivar" not in out.columns:
        out["Cultivar"] = np.nan

    if "Average_Temperature" in out.columns:
        if "Temperature" not in out.columns:
            out["Temperature"] = out["Average_Temperature"]
        else:
            out["Temperature"] = out["Temperature"].fillna(out["Average_Temperature"])

    return out


def _load_crop_aliases(cfg: dict) -> dict:
    aliases = {}
    p = C.safe_resolve(cfg["inputs"]["phase17"]["ontology_aliases"])
    if not p.exists():
        return aliases
    onto = pd.read_parquet(p)
    crop = onto[onto["UAMS_Column"].astype(str).str.upper() == "CROP"]
    for _, r in crop.iterrows():
        a = C.s(r.get("Alias"))
        if a:
            aliases[a.lower()] = a
    # FAOSTAT item canonicalization only via exact alias; never fuzzy
    return aliases


def _canonicalize_targets(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    yield_cols = [
        c for c in ("Yield_per_Hectare", "Yield_per_Plot", "Yield_per_Acre") if c in out.columns
    ]
    if yield_cols:
        out["Yield"] = out[yield_cols].apply(
            lambda r: next((v for v in r if _present(v)), np.nan), axis=1
        )
    elif "Yield" not in out.columns:
        out["Yield"] = np.nan
    else:
        out["Yield"] = out.get("Yield", np.nan)

    for t in ("Biomass_Yield", "Plant_Height_cm", "Protein"):
        if t not in out.columns:
            out[t] = np.nan
    return out


# ---------------------------------------------------------------------------
# predictor completeness (STEP 10)
# ---------------------------------------------------------------------------


def compute_predictor_completeness(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = obs.copy()
    tiers = cfg["predictor_tiers"]
    required = list(cfg["required_predictors"])
    critical = list(cfg["critical_predictors"])
    classes = cfg["readiness_classes"]
    target = "Yield"

    out["PredictorCount"] = 0
    out["AvailablePredictorCount"] = 0
    out["MissingPredictorCount"] = 0
    out["CompletenessRatio"] = 0.0
    out["MissingCriticalPredictors"] = ""
    out["MissingPredictors"] = ""
    out["PredictorCompleteness_TierA"] = 0.0
    out["PredictorCompleteness_TierB"] = 0.0
    out["PredictorCompleteness_TierC"] = 0.0
    out["QualityGrade"] = "INSUFFICIENT"

    # effective required predictors = required list minus the target column
    eff_required = [p for p in required if p != target]

    def _avail(row, col):
        return _present(row.get(col))

    for idx, row in out.iterrows():
        missing = [c for c in eff_required if not _avail(row, c)]
        miss_crit = [c for c in critical if not _avail(row, c)]
        n_avail = len(eff_required) - len(missing)
        ratio = n_avail / len(eff_required) if eff_required else 1.0

        tier_b = list(tiers.get("TIER_B", []))
        tier_c = list(tiers.get("TIER_C", []))
        tB = sum(1 for c in tier_b if _avail(row, c)) / len(tier_b) if tier_b else 1.0
        tC = sum(1 for c in tier_c if _avail(row, c)) / len(tier_c) if tier_c else 1.0

        out.at[idx, "PredictorCount"] = len(eff_required)
        out.at[idx, "AvailablePredictorCount"] = n_avail
        out.at[idx, "MissingPredictorCount"] = len(missing)
        out.at[idx, "CompletenessRatio"] = round(ratio, 4)
        out.at[idx, "PredictorCompleteness_TierA"] = round(ratio, 4)
        out.at[idx, "PredictorCompleteness_TierB"] = round(tB, 4)
        out.at[idx, "PredictorCompleteness_TierC"] = round(tC, 4)
        out.at[idx, "MissingPredictors"] = ";".join(missing)
        out.at[idx, "MissingCriticalPredictors"] = ";".join(miss_crit)
        out.at[idx, "QualityGrade"] = _grade(ratio, classes)

    # tier coverage columns
    for tier in tiers:
        cols = list(tiers[tier])
        out[f"Tier_{tier}_available"] = out.apply(
            lambda r, cols=cols: int(sum(1 for c in cols if _present(r.get(c)))), axis=1
        )
        out[f"Tier_{tier}_required"] = len(cols)

    has_yield = (
        out["Yield"].apply(_present)
        if "Yield" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["Target"] = np.where(has_yield, "Yield", "")
    return out


def _grade(ratio: float, classes: dict) -> str:
    if ratio >= classes["MODEL_READY_min"]:
        return "MODEL_READY"
    if ratio >= classes["NEAR_READY_min"]:
        return "NEAR_READY"
    if ratio >= classes["PARTIAL_min"]:
        return "PARTIAL"
    return "INSUFFICIENT"


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_linked_observations(cfg)

    keep = [
        "ObservationID_ML",
        "canonical_observation_id",
        "canonical_study_id",
        "canonical_experiment_id",
        "canonical_location_id",
        "PaperID",
        "ExperimentID",
        "TreatmentID",
        "ObservationID",
        "Crop",
        "Crop_Normalized",
        "Canonical_Year",
        "year",
        "season",
        "Season",
        "Location",
        "country",
        "latitude",
        "longitude",
        "location_precision",
        "Yield",
        "Biomass_Yield",
        "Plant_Height_cm",
        "Protein",
        "Rainfall",
        "Temperature",
        "Temperature_Max",
        "Temperature_Min",
        "Soil_pH",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Solar_Radiation",
        "Organic_Carbon",
        "Soil_Moisture",
        "Soil_Texture",
        "Texture",
        "Humidity",
        "Cultivar",
        "Plant_Population",
        "Soil_EC",
        "Irrigation",
        "Fertilizer_Type",
        "N_Dose",
        "P_Dose",
        "K_Dose",
        "Days_to_Maturity",
        "PredictorCount",
        "AvailablePredictorCount",
        "MissingPredictorCount",
        "CompletenessRatio",
        "PredictorCompleteness_TierA",
        "PredictorCompleteness_TierB",
        "PredictorCompleteness_TierC",
        "MissingPredictors",
        "MissingCriticalPredictors",
        "QualityGrade",
        "Target",
    ]
    rep = obs[[c for c in keep if c in obs.columns]].copy()

    C.to_parquet(rep, C.OUT / "predictor_completeness.parquet")
    C.to_excel(rep, C.REPORTS / "predictor_completeness.xlsx", "Completeness")

    mean = float(obs["CompletenessRatio"].mean())
    result = {
        "generated_at": C.now_full_iso(),
        "observations": int(len(obs)),
        "avg_predictor_completeness": round(mean, 4),
        "grade_counts": obs["QualityGrade"].value_counts().to_dict(),
        "target_counts": obs["Target"].value_counts().to_dict(),
        "tierA_avg": round(float(obs["PredictorCompleteness_TierA"].mean()), 4),
        "tierB_avg": round(float(obs["PredictorCompleteness_TierB"].mean()), 4),
        "tierC_avg": round(float(obs["PredictorCompleteness_TierC"].mean()), 4),
    }
    C.write_json(result, C.OUT / "predictor_completeness_metrics.json")
    C.mark_done("predictors", result)
    C.log_msg(f"STEP10 predictors: avg completeness {mean:.4f}")
    return result


if __name__ == "__main__":
    import sys

    run(force="--force" in sys.argv)
