from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.config.schema import UAMS_COLUMNS
from agri_ai_agent.external_data.column_mapper import (
    KNOWN_SOURCE_MAPS,
    _normalized_uams,
    discover_new_columns,
    map_dataframe,
    map_column,
)
from agri_ai_agent.external_data.dataset_package import DatasetPackage

logger = logging.getLogger(__name__)

EXTERNAL_SCHEMA_REGISTRY: dict[str, str] = dict(_normalized_uams)


def register_external_columns(columns: dict[str, str]) -> list[str]:
    registered: list[str] = []
    for raw_name, norm_name in columns.items():
        if norm_name not in EXTERNAL_SCHEMA_REGISTRY:
            uams_name = norm_name.title().replace("_", " ")
            uams_name = uams_name.replace(" ", "_")
            EXTERNAL_SCHEMA_REGISTRY[norm_name] = uams_name
            registered.append(uams_name)
    return registered


def _detect_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lat_col = None
    lon_col = None
    for col in df.columns:
        low = col.lower().replace(" ", "_").replace("-", "_")
        if low in ("lat", "latitude", "y") and lat_col is None:
            lat_col = col
        elif low in ("lon", "long", "longitude", "x") and lon_col is None:
            lon_col = col
    return lat_col, lon_col


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for col in df.columns:
        low = col.lower().replace(" ", "_").replace("-", "_")
        if any(kw in low for kw in ("year", "date", "doy", "month", "day")):
            date_cols.append(col)
    return date_cols


def _detect_crop_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        low = col.lower().replace(" ", "_").replace("-", "_")
        if low in ("crop", "item", "crop_name", "crop_type", "commodity"):
            return col
    return None


def _spatial_join(
    master: pd.DataFrame,
    external: pd.DataFrame,
    master_lat: str,
    master_lon: str,
    ext_lat: str,
    ext_lon: str,
    lat_tol: float = 1.0,
    lon_tol: float = 1.0,
) -> pd.DataFrame:
    if master.empty or external.empty:
        return master
    m_lat = pd.to_numeric(master[master_lat], errors="coerce")
    m_lon = pd.to_numeric(master[master_lon], errors="coerce")
    e_lat = pd.to_numeric(external[ext_lat], errors="coerce")
    e_lon = pd.to_numeric(external[ext_lon], errors="coerce")
    valid_m = m_lat.notna() & m_lon.notna()
    valid_e = e_lat.notna() & e_lon.notna()
    if not valid_m.any() or not valid_e.any():
        return master
    master_valid = master[valid_m].copy()
    ext_valid = external[valid_e].copy()
    if master_valid.empty or ext_valid.empty:
        return master
    ext_to_join = ext_valid.drop(columns=[ext_lat, ext_lon], errors="ignore")
    master_valid["_lat_key"] = (m_lat[valid_m] / lat_tol).round().astype(int)
    master_valid["_lon_key"] = (m_lon[valid_m] / lon_tol).round().astype(int)
    ext_join = ext_valid.copy()
    ext_join["_lat_key"] = (e_lat[valid_e] / lat_tol).round().astype(int)
    ext_join["_lon_key"] = (e_lon[valid_e] / lon_tol).round().astype(int)
    ext_join = ext_join.drop(columns=[ext_lat, ext_lon], errors="ignore")
    ext_merge_cols = [c for c in ext_join.columns if c not in master_valid.columns or c in ("_lat_key", "_lon_key")]
    joined = master_valid.merge(
        ext_join[ext_merge_cols],
        on=["_lat_key", "_lon_key"],
        how="left",
        suffixes=("", "_ext"),
    )
    dupes = [c for c in joined.columns if c.endswith("_ext")]
    joined = joined.drop(columns=dupes, errors="ignore")
    joined = joined.drop(columns=["_lat_key", "_lon_key"], errors="ignore")
    result = master.copy()
    for col in joined.columns:
        if col in result.columns and col not in (master_lat, master_lon):
            result[col] = result[col].fillna(joined[col])
    for col in joined.columns:
        if col not in result.columns:
            result[col] = joined[col]
    return result


def _categorical_join(
    master: pd.DataFrame,
    external: pd.DataFrame,
    master_key: str,
    ext_key: str,
) -> pd.DataFrame:
    if master_key not in master.columns or ext_key not in external.columns:
        return master
    to_join = external.drop(columns=[ext_key], errors="ignore")
    ext_cols = [c for c in to_join.columns if c not in master.columns]
    if not ext_cols:
        return master
    join_map = external[[ext_key]].drop_duplicates()
    if join_map.empty:
        return master
    result = master.merge(
        to_join[ext_cols + [ext_key]],
        left_on=master_key,
        right_on=ext_key,
        how="left",
        suffixes=("", "_ext"),
    )
    result = result.drop(columns=[ext_key], errors="ignore")
    return result


def enrich_master(
    master_df: pd.DataFrame,
    packages_by_source: dict[str, list[DatasetPackage]],
) -> pd.DataFrame:
    result = master_df.copy()
    if result.empty:
        return result

    for source, packages in packages_by_source.items():
        for pkg in packages:
            ext_df = pkg.to_dataframe()
            if ext_df is None or ext_df.empty:
                continue
            mapped = map_dataframe(source, ext_df, drop_unmapped=False)
            unmapped_cols = [c for c in mapped.columns if c not in UAMS_COLUMNS and not c.startswith("_")]
            if unmapped_cols:
                new_registered = register_external_columns(
                    {c: c for c in unmapped_cols}
                )
                if new_registered:
                    logger.info(
                        "[%s] Registered %d new schema columns from package %s: %s",
                        source, len(new_registered), pkg.resource_id,
                        new_registered[:5],
                    )

            # 1. Try spatial join (lat/lon)
            m_lat, m_lon = _detect_lat_lon_columns(result)
            e_lat, e_lon = _detect_lat_lon_columns(mapped)
            if m_lat and m_lon and e_lat and e_lon:
                before = len(result.columns)
                result = _spatial_join(
                    result, mapped,
                    m_lat, m_lon, e_lat, e_lon,
                )
                added = len(result.columns) - before
                if added > 0:
                    logger.info(
                        "[%s] Spatial join added %d columns from %s",
                        source, added, pkg.resource_id,
                    )
                continue

            # 2. Try categorical join (crop + year)
            crop_col = _detect_crop_column(mapped)
            if crop_col and "Crop" in result.columns:
                before = len(result.columns)
                result = _categorical_join(result, mapped, "Crop", crop_col)
                added = len(result.columns) - before
                if added > 0:
                    logger.info(
                        "[%s] Crop join added %d columns from %s",
                        source, added, pkg.resource_id,
                    )
                continue

            # 3. Fallback: merge by matching column names
            match_cols = [c for c in mapped.columns if c in result.columns]
            if match_cols:
                new_cols = [c for c in mapped.columns if c not in result.columns and c not in UAMS_COLUMNS]
                if new_cols:
                    for nc in new_cols:
                        result[nc] = np.nan
                    logger.info(
                        "[%s] Added %d new columns via column-match from %s: %s",
                        source, len(new_cols), pkg.resource_id, new_cols[:5],
                    )

            # 4. Row append only for structured numeric data with matching columns
            mapped_uams = map_dataframe(source, ext_df, drop_unmapped=True)
            uams_in_mapped = [c for c in mapped_uams.columns if c in UAMS_COLUMNS]
            uams_in_result = [c for c in UAMS_COLUMNS if c in result.columns]
            shared = set(uams_in_mapped) & set(uams_in_result)
            if len(shared) >= 3:
                for col in mapped_uams.columns:
                    if col in result.columns:
                        result[col] = result[col].fillna(mapped_uams[col])
                    else:
                        result[col] = mapped_uams[col]

    return result
