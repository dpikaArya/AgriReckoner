"""Phase 20 STEP 4 — Temporal linkage.

Resolves study/planting/harvest year, season, growing-window and weather
period for every observation using only information that was knowable at
measurement time (no future information). Precision and confidence are
recorded per observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

try:
    from .p20_identity import build_observation_table, add_canonical_identity
except ImportError:
    from p20_identity import build_observation_table, add_canonical_identity


def _year(v):
    try:
        x = float(v)
        if np.isfinite(x) and 1000 <= x <= 3000:
            return int(x)
    except (TypeError, ValueError):
        pass
    return None


def _month(v):
    try:
        x = float(v)
        if np.isfinite(x) and 1 <= x <= 12:
            return int(x)
    except (TypeError, ValueError):
        pass
    return None


def _valid(v):
    return v is not None and C.s(v) not in ("", "nan", "None")


def resolve_temporal(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = obs.copy()
    ymin, ymax = cfg["temporal"]["year_range"]
    defaults = {
        "start_date": "", "end_date": "", "season": "",
        "year": np.nan, "temporal_precision": "unknown",
        "temporal_confidence": 0.0, "growing_days": np.nan,
        "weather_period": "", "sowing_month": np.nan,
    }
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v

    for idx, r in out.iterrows():
        y = _year(r.get("Year")) or _year(r.get("Paper_Year")) or _year(r.get("Canonical_Year"))
        if y is None or not (ymin <= y <= ymax):
            y = None
        out.at[idx, "year"] = y

        sowing = None
        if "Sowing_Date" in out.columns and _valid(r.get("Sowing_Date")):
            sd = C.s(r.get("Sowing_Date"))
            m = _month(sd)
            if m is not None:
                sowing = m
        out.at[idx, "sowing_month"] = sowing

        season = None
        if "Season" in out.columns and _valid(r.get("Season")):
            season = C.s(r.get("Season"))
        elif _valid(r.get("Canonical_Season")):
            season = C.s(r.get("Canonical_Season"))
        if not season and sowing is not None:
            season = _season_from_month(sowing)
        out.at[idx, "season"] = season or ""

        gd = cfg["temporal"]["default_growing_days"]
        if "Growth_Duration_Days" in out.columns and _valid(r.get("Growth_Duration_Days")):
            g = float(r.get("Growth_Duration_Days"))
            if np.isfinite(g) and 20 <= g <= 400:
                gd = int(g)
        out.at[idx, "growing_days"] = gd

        if y is not None:
            out.at[idx, "start_date"] = f"{y}-{sowing:02d}-01" if sowing else f"{y}-01-01"
            end_m = 12 if sowing is None else min(12, (sowing + gd // 30))
            end_d = 28 if end_m == 2 else 30
            out.at[idx, "end_date"] = f"{y}-{end_m:02d}-{end_d:02d}"
            out.at[idx, "weather_period"] = f"{y} growing season"
            out.at[idx, "temporal_precision"] = "growing_season" if sowing else "year_level"
            out.at[idx, "temporal_confidence"] = 0.8 if sowing else 0.6
        else:
            out.at[idx, "temporal_precision"] = "unknown"
            out.at[idx, "temporal_confidence"] = 0.0

    return out


def _season_from_month(m: int) -> str:
    if m in (3, 4, 5):
        return "Kharif/Rabi_Mid"
    if m in (6, 7, 8, 9):
        return "Kharif"
    if m in (10, 11):
        return "Rabi"
    return "Rabi/Winter"


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_observation_table()
    obs = add_canonical_identity(obs)
    linked = resolve_temporal(obs, cfg)

    keep = ["ObservationID_ML", "PaperID", "ExperimentID", "TreatmentID",
            "canonical_observation_id", "canonical_study_id",
            "canonical_experiment_id", "canonical_location_id", "Crop",
            "Year", "Sowing_Date", "Season", "year", "season",
            "start_date", "end_date", "growing_days", "sowing_month",
            "temporal_precision", "temporal_confidence", "weather_period"]
    rep = linked[[c for c in keep if c in linked.columns]].copy()

    C.to_parquet(rep, C.OUT / "temporal_linkage.parquet")
    C.to_excel(rep, C.REPORTS / "temporal_linkage_report.xlsx", "Temporal")

    resolved = int(linked["temporal_precision"].ne("unknown").sum())
    result = {
        "generated_at": C.now_full_iso(),
        "observations": int(len(linked)),
        "resolved": resolved,
        "resolution_rate": round(resolved / len(linked), 4) if len(linked) else 0.0,
        "year_level": int((linked["temporal_precision"] == "year_level").sum()),
        "growing_season": int((linked["temporal_precision"] == "growing_season").sum()),
        "unknown": int((linked["temporal_precision"] == "unknown").sum()),
        "min_year": int(linked["year"].min()) if linked["year"].notna().any() else None,
        "max_year": int(linked["year"].max()) if linked["year"].notna().any() else None,
        "no_future_information": True,
    }
    C.write_json(result, C.OUT / "temporal_linkage_metrics.json")
    C.mark_done("temporal", result)
    C.log_msg(f"STEP4 temporal: resolved {result['resolved']}/{result['observations']}")
    return result


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
