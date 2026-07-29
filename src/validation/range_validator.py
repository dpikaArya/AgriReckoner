from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import logging


AGRICULTURAL_RANGES: Dict[str, dict] = {
    "Temperature": {"min": -50, "max": 60, "unit": "°C"},
    "Rainfall": {"min": 0, "max": 10000, "unit": "mm/year"},
    "Humidity": {"min": 0, "max": 100, "unit": "%"},
    "Soil_pH": {"min": 0, "max": 14, "unit": "pH"},
    "Organic_Carbon": {"min": 0, "max": 100, "unit": "%"},
    "Nitrogen": {"min": 0, "max": 1000, "unit": "kg/ha"},
    "Phosphorus": {"min": 0, "max": 1000, "unit": "kg/ha"},
    "Potassium": {"min": 0, "max": 1000, "unit": "kg/ha"},
    "Crop_Yield": {"min": 0, "max": 100, "unit": "t/ha"},
    "Latitude": {"min": -90, "max": 90, "unit": "degrees"},
    "Longitude": {"min": -180, "max": 180, "unit": "degrees"},
    "Elevation": {"min": -500, "max": 9000, "unit": "m"},
    "Wind_Speed": {"min": 0, "max": 200, "unit": "km/h"},
    "Solar_Radiation": {"min": 0, "max": 1500, "unit": "W/m²"},
    "CO2_Concentration": {"min": 300, "max": 1000, "unit": "ppm"},
}


class RangeValidator:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.ranges = AGRICULTURAL_RANGES.copy()
        self.logger = logger or logging.getLogger(__name__)

    def validate(
        self,
        df: pd.DataFrame,
        column_map: Optional[Dict[str, str]] = None,
    ) -> dict:
        column_map = column_map or {}
        results = {}
        auto_map = self._auto_detect_columns(df)

        resolved = {}
        for range_key in self.ranges:
            mapped_col = column_map.get(range_key)
            if mapped_col and mapped_col in df.columns:
                resolved[range_key] = mapped_col
            elif range_key in auto_map:
                resolved[range_key] = auto_map[range_key]

        for range_key, col_name in resolved.items():
            r = self.ranges[range_key]
            series = df[col_name].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                results[range_key] = {
                    "column": col_name,
                    "error": f"Column '{col_name}' is not numeric",
                }
                continue

            violations: List[Tuple[int, float]] = []
            for idx, val in series.items():
                if val < r["min"] or val > r["max"]:
                    violations.append((int(idx), float(val)))
                    if len(violations) >= 10:
                        break

            valid_count = int(len(series) - len(violations))
            out_of_range_count = len(violations)
            results[range_key] = {
                "column": col_name,
                "valid_count": valid_count,
                "out_of_range_count": out_of_range_count,
                "min_val": float(series.min()) if len(series) > 0 else None,
                "max_val": float(series.max()) if len(series) > 0 else None,
                "range_min": r["min"],
                "range_max": r["max"],
                "unit": r["unit"],
                "violations": violations,
            }

        temperature_checks = self._check_temperature_consistency(df)
        if temperature_checks:
            results["_temperature_consistency"] = temperature_checks

        return results

    def validate_coordinates(self, lat: pd.Series, lon: pd.Series) -> dict:
        valid_pairs = 0
        invalid_pairs = 0
        violations = []
        lat_range = self.ranges["Latitude"]
        lon_range = self.ranges["Longitude"]

        for idx in range(len(lat)):
            lv = lat.iloc[idx]
            ln = lon.iloc[idx]
            if pd.isna(lv) or pd.isna(ln):
                invalid_pairs += 1
                violations.append((int(lat.index[idx]), float(lv) if not pd.isna(lv) else None, float(ln) if not pd.isna(ln) else None, "null value"))
                continue
            lat_ok = lat_range["min"] <= lv <= lat_range["max"]
            lon_ok = lon_range["min"] <= ln <= lon_range["max"]
            if lat_ok and lon_ok:
                valid_pairs += 1
            else:
                invalid_pairs += 1
                reason = []
                if not lat_ok:
                    reason.append(f"lat={lv} not in [{lat_range['min']},{lat_range['max']}]")
                if not lon_ok:
                    reason.append(f"lon={ln} not in [{lon_range['min']},{lon_range['max']}]")
                violations.append((int(lat.index[idx]), float(lv), float(ln), "; ".join(reason)))
                if len(violations) >= 10:
                    break

        return {
            "valid_pairs": valid_pairs,
            "invalid_pairs": invalid_pairs,
            "total_pairs": len(lat),
            "violations": violations,
        }

    def _auto_detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        col_lower = {c: c.lower().replace(" ", "_").replace("-", "_") for c in df.columns}
        rev_map = {v: k for k, v in col_lower.items()}

        for range_key in self.ranges:
            norm_key = range_key.lower()
            if norm_key in rev_map:
                mapping[range_key] = rev_map[norm_key]
                continue
            maybe = norm_key.replace("_", "")
            for c, cnorm in col_lower.items():
                if cnorm.replace("_", "") == maybe:
                    mapping[range_key] = c
                    break
        return mapping

    def _check_temperature_consistency(self, df: pd.DataFrame) -> Optional[dict]:
        tmin_col = None
        tmax_col = None
        for c in df.columns:
            cl = c.lower().replace(" ", "_").replace("-", "_")
            if cl in ("tmin", "temp_min", "temperature_min"):
                tmin_col = c
            elif cl in ("tmax", "temp_max", "temperature_max"):
                tmax_col = c

        if tmin_col and tmax_col:
            tmin = df[tmin_col]
            tmax = df[tmax_col]
            mask = ~(tmin.isna() | tmax.isna())
            violations = []
            for idx in df[mask].index:
                if tmin[idx] > tmax[idx]:
                    violations.append((int(idx), float(tmin[idx]), float(tmax[idx])))
                    if len(violations) >= 10:
                        break
            return {
                "tmin_column": tmin_col,
                "tmax_column": tmax_col,
                "consistent_count": int(mask.sum() - len(violations)),
                "inconsistent_count": len(violations),
                "violations": violations,
            }
        return None
