"""Phase 20 STEP 2 — Canonical observation identity.

Builds the wide observation table from UAMS_v2.1 (literature rows grouped by
Paper/Experiment/Treatment, external rows one row per ObservationID) and
derives conservative canonical identity keys:

    canonical_observation_id
    canonical_study_id
    canonical_experiment_id
    canonical_location_id

Duplicate records are never silently merged — ambiguous identities are routed
to a review queue.

Protected inputs (UAMS_v2, UAMS_v2.1, ...) are read-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C


SPURIOUS_COLUMNS = {"100", "86.95652173913044", "Value", "Unit", "Treatment",
                    "PARAMETER", "Statistic", "Property", "Treatment1"}

IDENTITY_CARRY = {
    "SourceRow", "SourceColumn", "SourceSystem", "SourceTable", "Caption",
    "DOI", "OriginalPaperID", "Provenance_ID", "Dataset_ID",
}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _first(x):
    vals = [v for v in x if v is not None and C.s(v) not in ("", "nan", "None")]
    return vals[0] if vals else np.nan


def build_observation_table(uams: pd.DataFrame | None = None) -> pd.DataFrame:
    """One wide row per scientific observation.

    Literature rows (PaperID present) are grouped by (PaperID, ExperimentID,
    TreatmentID); replication rows are aggregated by mean for numeric
    variables. External rows (no PaperID) yield one row per ObservationID.
    """
    if uams is None:
        uams = pd.read_parquet(C.PROJECT_ROOT / "outputs" / "UAMS_v2.1.parquet")
    u = uams.copy()
    u["_num"] = u["NormalizedValue"].map(_num)
    u["_num_ok"] = u["_num"].notna()
    u["PaperID_s"] = u["PaperID"].astype(str).str.strip()

    lit = u[u["PaperID_s"].ne("")].copy()
    ext = u[u["PaperID_s"].eq("")].copy()
    wide_parts = []

    if len(lit):
        grp_keys = ["PaperID", "ExperimentID", "TreatmentID"]
        base = lit.groupby(grp_keys).agg(
            Crop=("Crop", lambda s: _first(list(s.dropna()))),
            Year=("Year", lambda s: _first(list(s.dropna()))),
            Variety=("TreatmentLabel", lambda s: _first(list(s.dropna()))),
            n_measurement_rows=("MeasurementID", "count"),
            SourceSystem=("SourceSystem", lambda s: _first(list(s.dropna()))),
            DOI=("DOI_paper", lambda s: _first(list(s.dropna()))),
        ).reset_index()

        num_agg = lit[lit["_num_ok"]].groupby(grp_keys + ["Variable"])["_num"] \
            .mean().reset_index()
        num_wide = num_agg.pivot_table(index=grp_keys, columns="Variable",
                                       values="_num", aggfunc="mean").reset_index()
        num_wide.columns = [str(c) for c in num_wide.columns]

        cat_agg = lit[~lit["_num_ok"]].groupby(grp_keys + ["Variable"])[
            "NormalizedValue"].agg(lambda s: _first(list(s))).reset_index()
        cat_wide = cat_agg.pivot_table(index=grp_keys, columns="Variable",
                                       values="NormalizedValue",
                                       aggfunc=_first).reset_index()
        cat_wide.columns = [str(c) for c in cat_wide.columns]

        protected = {"Crop", "Year", "Variety", "n_measurement_rows", "SourceSystem",
                     "DOI"}
        for _c in protected:
            if _c in _keep(num_wide.columns) and _c not in grp_keys:
                num_wide = num_wide.rename(columns={_c: _c + "_var"})
            if _c in _keep(cat_wide.columns) and _c not in grp_keys:
                cat_wide = cat_wide.rename(columns={_c: _c + "_var"})

        lit_wide = base.merge(num_wide, on=grp_keys, how="left") \
            .merge(cat_wide, on=grp_keys, how="left")
        lit_wide["ObservationLevel"] = "literature"
        wide_parts.append(lit_wide)

    if len(ext):
        recs = []
        for _, r in ext.iterrows():
            rec = {
                "PaperID": "", "ExperimentID": "", "TreatmentID": "",
                "Crop": r.get("Crop"), "Year": r.get("Year"),
                "Variety": r.get("TreatmentLabel"), "n_measurement_rows": 1,
                "ObservationLevel": "external",
                "SourceSystem": r.get("SourceSystem"),
                "ObservationID": r.get("ObservationID"),
                "DOI": r.get("DOI"),
                r.get("Variable"): r.get("NormalizedValue"),
            }
            for k in IDENTITY_CARRY:
                if k in r.index:
                    rec[k] = r.get(k)
            recs.append(rec)
        ext_wide = pd.DataFrame(recs)
        wide_parts.append(ext_wide)

    obs = pd.concat(wide_parts, ignore_index=True, sort=False)
    obs = obs.reset_index(drop=True)
    obs.insert(0, "ObservationID_ML", obs.index.astype(str))
    obs["PaperID"] = obs["PaperID"].fillna("").astype(str)
    obs["ExperimentID"] = obs["ExperimentID"].fillna("").astype(str)
    obs["TreatmentID"] = obs["TreatmentID"].fillna("").astype(str)
    obs["ObservationID"] = obs.get("ObservationID", pd.Series("", index=obs.index))
    obs["ObservationID"] = obs["ObservationID"].fillna("").astype(str)

    drop_cols = [c for c in obs.columns if c.strip() in SPURIOUS_COLUMNS]
    obs = obs.drop(columns=[c for c in drop_cols if c in obs.columns])
    return obs


def _keep(cols) -> set:
    return {str(c) for c in cols}


def _norm_id(*parts) -> str:
    pieces = []
    for p in parts:
        s = C.s(p)
        if s and s.lower() not in ("nan", "none"):
            pieces.append(s)
    return "::".join(pieces) if pieces else "UNKNOWN"


def add_canonical_identity(obs: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical identity columns for every observation row."""
    out = obs.copy()
    n = len(out)

    out["canonical_observation_id"] = [f"OBS-{i:06d}" for i in range(n)]
    out["canonical_study_id"] = out.apply(
        lambda r: _norm_id(r["PaperID"]) if C.s(r["PaperID"]) else _norm_id("EXT", r["ObservationID"]),
        axis=1)
    out["canonical_experiment_id"] = out.apply(
        lambda r: _norm_id(r["PaperID"], r["ExperimentID"]) if C.s(r["PaperID"])
        else _norm_id("EXT", r["ObservationID"]), axis=1)
    out["canonical_location_id"] = out.apply(
        lambda r: _norm_id(r["PaperID"], r["ExperimentID"], "LOC") if C.s(r["PaperID"])
        else _norm_id("EXT", r["ObservationID"], "LOC"), axis=1)

    out["Canonical_Crop"] = out["Crop"]
    out["Canonical_Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Canonical_Location"] = out.get("Location", np.nan)
    out["Canonical_Season"] = out.get("Season", np.nan)

    out["Source"] = out["SourceSystem"]
    out["Provenance"] = np.where(out["ObservationLevel"] == "literature",
                                 "literature:" + out["PaperID"].astype(str),
                                 "external:" + out["ObservationID"].astype(str))
    return out


def _duplicate_fingerprint(r) -> str:
    return C.record_fingerprint(
        r.get("PaperID"), r.get("ExperimentID"), r.get("TreatmentID"),
        r.get("Crop"), r.get("Year"))


def detect_duplicates(obs: pd.DataFrame) -> pd.DataFrame:
    """Flag potential duplicate observations without merging.

    Conservative: rows sharing paper/experiment/treatment/crop/year but with
    differing values go to a review queue.
    """
    out = obs.copy()
    fp = out.apply(_duplicate_fingerprint, axis=1)
    out["_fp"] = fp
    counts = out.groupby("_fp")["ObservationID_ML"].transform("count")
    out["IdentityConfidence"] = np.where(counts > 1, 0.5, 1.0)
    out["DuplicateCandidate"] = counts > 1
    return out.drop(columns=["_fp"])


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_observation_table()
    obs = add_canonical_identity(obs)
    obs = detect_duplicates(obs)

    dup = obs[obs["DuplicateCandidate"]].copy()

    C.to_parquet(obs, C.OUT / "canonical_identity.parquet")
    C.to_parquet(dup, C.OUT / "duplicate_review_queue.parquet")
    C.to_excel(dup, C.REPORTS / "canonical_identity_report.xlsx", "ReviewQueue")

    result = {
        "generated_at": C.now_full_iso(),
        "observations_total": int(len(obs)),
        "literature_observations": int((obs["ObservationLevel"] == "literature").sum()),
        "external_observations": int((obs["ObservationLevel"] == "external").sum()),
        "independent_studies": int(obs["canonical_study_id"].nunique()),
        "independent_experiments": int(obs["canonical_experiment_id"].nunique()),
        "independent_locations": int(obs["canonical_location_id"].nunique()),
        "duplicate_candidates": int(len(dup)),
        "duplicates_merged": 0,
        "note": "duplicates flagged for review; none silently merged",
    }
    C.write_json(result, C.OUT / "canonical_identity_metrics.json")
    C.mark_done("identity", result)
    C.log_msg(f"STEP2 identity: {len(obs)} observations, "
              f"{result['independent_studies']} studies, "
              f"{result['duplicate_candidates']} duplicate candidates")
    return result


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
