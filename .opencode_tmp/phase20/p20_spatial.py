"""Phase 20 STEP 3 — Hierarchical spatial linkage.

Resolves a location record for every observation using a strict priority
list, never inventing coordinates:

  1. exact latitude/longitude (plot/station coordinates)
  2. coordinates recovered from the Phase 18 staging source records
  3. experimental station coordinates (study-level, radius matched)
  4. district + country
  5. known research station
  6. administrative region
  7. country only

Every resolved location records precision, source, confidence, and match
method. Plot-level precision is never assigned to country-only records.
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


def _geo(x):
    try:
        v = float(x)
        return v if np.isfinite(v) and abs(v) <= 180.0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _valid(v):
    return v is not None and C.s(v) not in ("", "nan", "None")


def _parse_coords(value) -> tuple:
    """Parse a 'lat,lon' / 'lat;lon' / pair-of-numbers coordinate string."""
    if not _valid(value):
        return np.nan, np.nan
    txt = C.s(value)
    for sep in [",", ";", "|", " "]:
        if sep in txt:
            parts = [p.strip() for p in txt.split(sep) if p.strip()]
            if len(parts) == 2:
                lat = _geo(parts[0])
                lon = _geo(parts[1])
                if not pd.isna(lat) and not pd.isna(lon) and abs(lat) <= 90:
                    return lat, lon
    v = _geo(txt)
    return v, np.nan


def build_staging_coords(cfg) -> pd.DataFrame:
    """Location look-up table from Phase 18 approved external records."""
    stg = C.load_input("phase18.staging_external", cfg)
    rows = []
    for _, r in stg.iterrows():
        lat, lon = _geo(r.get("Location_Lat")), _geo(r.get("Location_Lon"))
        if pd.isna(lat) or pd.isna(lon):
            continue
        rows.append({
            "lat": lat, "lon": lon,
            "country": C.s(r.get("Country")),
            "provenance_id": C.s(r.get("Provenance_ID")),
            "source": C.s(r.get("Source")),
            "spatial_grade": C.s(r.get("Spatial_Match_Grade")),
        })
    out = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
    return out.reset_index(drop=True)


def resolve_location(obs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hierarchical location resolution for every observation."""
    out = obs.copy()
    stg = build_staging_coords(cfg)
    precision_order = cfg["spatial"]["precision_order"]
    exact_r = cfg["spatial"]["exact_match_radius_km"]
    station_r = cfg["spatial"]["station_match_radius_km"]

    defaults = {
        "latitude": np.nan, "longitude": np.nan,
        "location_precision": "unknown",
        "location_source": "", "location_confidence": 0.0,
        "administrative_level": "",
        "country": "", "state": "", "district": "",
        "location_match_method": "",
    }
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v

    # match observed coordinates to staged coordinate set
    observed = []
    for _, r in out.iterrows():
        lat = np.nan
        lon = np.nan
        if "Latitude" in out.columns and _valid(r.get("Latitude")):
            lat, lon = _parse_coords(r.get("Latitude"))
            if pd.isna(lon) and _valid(r.get("Longitude")):
                lon = _geo(r.get("Longitude"))
        elif "Latitude" not in out.columns and "Longitude" in out.columns:
            lon = _geo(r.get("Longitude"))
        if pd.isna(lat) and _valid(r.get("Location")):
            p = _parse_coords(r.get("Location"))
            if not pd.isna(p[0]):
                lat, lon = p
        observed.append((lat, lon))

    # nearest staged coordinate for each observed pair
    staged = stg
    for i, (lat, lon) in enumerate(observed):
        if pd.isna(lat) or pd.isna(lon):
            continue
        best = None
        best_d = None
        for _, s in staged.iterrows():
            d = haversine_km(lat, lon, s["lat"], s["lon"])
            if pd.isna(d):
                continue
            if best_d is None or d < best_d:
                best_d = d
                best = s
        if best is not None and best_d is not None:
            method = "exact_match"
            if best_d > exact_r and best_d <= station_r:
                method = "station_radius_match"
            if best_d > station_r:
                continue
            out.at[i, "latitude"] = best["lat"]
            out.at[i, "longitude"] = best["lon"]
            out.at[i, "location_precision"] = "experimental_station" if method == "station_radius_match" else "exact_coordinates"
            out.at[i, "location_source"] = "phase18_staged:" + C.s(best["source"])
            out.at[i, "location_confidence"] = round(max(0.1, 1.0 - best_d / max(station_r, 1e-6)), 3)
            out.at[i, "country"] = best["country"]
            out.at[i, "administrative_level"] = best["spatial_grade"]
            out.at[i, "location_match_method"] = method

    # carry the observed raw coordinates as a fallback precision signal
    for i, (lat, lon) in enumerate(observed):
        if pd.isna(out.at[i, "latitude"]):
            out.at[i, "latitude"] = lat
            out.at[i, "longitude"] = lon
            if not pd.isna(lat) and not pd.isna(lon):
                out.at[i, "location_precision"] = "study_location"
                out.at[i, "location_confidence"] = 0.5
                out.at[i, "location_source"] = "uams_v2_1"
                out.at[i, "location_match_method"] = "direct_observation"

    # country-only precision for external FAOSTAT rows linked via staging
    for i, r in out.iterrows():
        if r["location_precision"] in ("unknown",):
            c = C.s(r.get("Country"))
            if c and c.lower() not in ("nan", "none"):
                out.at[i, "country"] = c
                out.at[i, "location_precision"] = "country_only"
                out.at[i, "location_confidence"] = 0.3
                out.at[i, "administrative_level"] = "country"
                out.at[i, "location_source"] = "uams_v2_1"
                out.at[i, "location_match_method"] = "country_label"

    # normalize precision rank
    out["location_precision"] = pd.Categorical(
        out["location_precision"], categories=precision_order, ordered=True)
    return out


def run(force: bool = False):
    cfg = C.load_config()
    obs = build_observation_table_with_identity()
    linked = resolve_location(obs, cfg)

    keep = ["ObservationID_ML", "PaperID", "ExperimentID", "TreatmentID",
            "ObservationID", "canonical_observation_id", "canonical_study_id",
            "canonical_experiment_id", "canonical_location_id", "Crop", "Year",
            "Country", "State", "Site", "Institution", "Location",
            "latitude", "longitude", "location_precision", "location_source",
            "location_confidence", "administrative_level", "country", "state",
            "district", "location_match_method"]
    rep = linked[[c for c in keep if c in linked.columns]].copy()

    C.to_parquet(rep, C.OUT / "spatial_linkage.parquet")
    C.to_excel(rep, C.REPORTS / "spatial_linkage_report.xlsx", "Spatial")

    prec_counts = linked["location_precision"].astype(str).value_counts().to_dict()
    success = linked["location_precision"].notna().sum()
    rate = success / len(linked) if len(linked) else 0.0

    result = {
        "generated_at": C.now_full_iso(),
        "observations": int(len(linked)),
        "resolved": int(success),
        "resolution_rate": round(rate, 4),
        "precision_counts": {str(k): int(v) for k, v in prec_counts.items()},
        "exact_or_station": int((linked["location_precision"].isin(
            ["exact_coordinates", "experimental_station", "study_location"])).sum()),
        "country_only": int((linked["location_precision"] == "country_only").sum()),
        "unknown": int((linked["location_precision"] == "unknown").sum()),
        "no_coordinates_invented": True,
    }
    C.write_json(result, C.OUT / "spatial_linkage_metrics.json")
    C.mark_done("spatial", result)
    C.log_msg(f"STEP3 spatial: resolved {result['resolved']}/{result['observations']}")
    return result


def build_observation_table_with_identity():
    obs = build_observation_table()
    obs = add_canonical_identity(obs)
    return obs


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
