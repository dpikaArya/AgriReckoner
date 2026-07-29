"""
Validation Agent — consolidated pipeline:
  Evidence validation → Source traceability → Unit harmonization → Quality assurance.
Extends BaseAgent and uses AgriAISettings.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings


UNIT_CONVERSIONS = {
    "acre_to_hectare": 0.404686,
    "hectare_to_acre": 2.47105,
    "kg_per_acre_to_kg_per_ha": 2.47105,
    "kg_per_ha_to_kg_per_acre": 0.404686,
    "g_per_plant_to_kg_per_ha": 10.0,
    "mg_per_kg_to_ppm": 1.0,
    "inch_to_cm": 2.54,
    "cm_to_inch": 0.393701,
    "mm_to_cm": 0.1,
    "cm_to_mm": 10.0,
    "ppm_to_mg_per_kg": 1.0,
    "tonne_per_ha_to_kg_per_ha": 1000.0,
    "lb_per_acre_to_kg_per_ha": 1.12085,
    "quintal_per_ha_to_kg_per_ha": 100.0,
    "kg_per_m2_to_tonne_per_ha": 10.0,
    "g_per_m2_to_kg_per_ha": 10.0,
    "mg_per_kg_to_g_per_tonne": 1.0,
    "ds_per_m_to_mmho_per_cm": 1.0,
    "percent_to_decimal": 0.01,
    "decimal_to_percent": 100.0,
}

RANGE_CONSTRAINTS = {
    "Soil_pH": (0, 14),
    "EC": (0, 100),
    "Humidity": (0, 100),
    "Temperature_Max": (-20, 60),
    "Temperature_Min": (-30, 50),
    "Average_Temperature": (-25, 55),
    "Rainfall": (0, 10000),
    "Protein": (0, 100),
    "Ash": (0, 100),
    "Fat": (0, 100),
    "Fiber": (0, 100),
    "Carbohydrates": (0, 100),
    "Moisture_Content": (0, 100),
    "Dry_Matter": (0, 100),
    "SPAD": (0, 100),
    "Yield_per_Hectare": (0, 100000),
    "Yield_per_Acre": (0, 50000),
    "Yield_per_Plot": (0, 100000),
    "100_Seed_Weight": (0, 1000),
    "Organic_Carbon": (0, 100),
    "Organic_Matter": (0, 100),
    "Nitrogen": (0, 1000),
    "Phosphorus": (0, 1000),
    "Potassium": (0, 1000),
    "Zinc": (0, 500),
    "Iron": (0, 1000),
    "Manganese": (0, 500),
    "Copper": (0, 200),
    "Boron": (0, 100),
}

BIOLOGICAL_RULES: dict[str, dict] = {
    "Plant_Height_cm": {"min": 1, "max": 800, "unit": "cm", "crop_ranges": {
        "wheat": (20, 120), "rice": (50, 150), "maize": (100, 400),
        "bell pepper": (30, 150), "carrot": (15, 40), "spinach": (15, 50),
    }},
    "Yield_per_Hectare": {"min": 100, "max": 20000, "unit": "kg/ha", "crop_ranges": {
        "wheat": (1000, 8000), "rice": (1500, 10000), "maize": (2000, 12000),
        "bell pepper": (5000, 80000), "carrot": (10000, 60000),
        "spinach": (5000, 30000), "chickpea": (800, 3000),
    }},
    "Soil_pH": {"min": 3.0, "max": 10.0, "unit": "pH"},
    "Nitrogen": {"min": 10, "max": 500, "unit": "kg/ha"},
    "Phosphorus": {"min": 5, "max": 200, "unit": "kg/ha"},
    "Potassium": {"min": 10, "max": 400, "unit": "kg/ha"},
    "Harvest_Index": {"min": 0.1, "max": 0.7, "unit": "ratio"},
    "SPAD": {"min": 10, "max": 80, "unit": "SPAD units"},
    "Leaf_Number": {"min": 2, "max": 100, "unit": "count"},
    "Tillers": {"min": 1, "max": 50, "unit": "count"},
    "Fruit_Weight": {"min": 0.5, "max": 2000, "unit": "g"},
    "100_Seed_Weight": {"min": 1, "max": 500, "unit": "g"},
}

AGRONOMIC_RULES: list[dict] = [
    {
        "name": "yield_biomass_ratio",
        "condition": lambda r: (
            r.get("Yield_per_Hectare") is not None and r.get("Biomass_Yield") is not None
            and r["Biomass_Yield"] > 0
        ),
        "check": lambda r: 0.1 <= (r["Yield_per_Hectare"] / r["Biomass_Yield"]) <= 0.8,
        "message": "Yield/Biomass ratio outside 0.1-0.8",
        "severity": "warning",
    },
    {
        "name": "temperature_consistency",
        "condition": lambda r: (
            r.get("Temperature_Max") is not None and r.get("Temperature_Min") is not None
        ),
        "check": lambda r: r["Temperature_Max"] >= r["Temperature_Min"],
        "message": "Tmax < Tmin",
        "severity": "error",
    },
    {
        "name": "height_leaf_area_correlation",
        "condition": lambda r: (
            r.get("Plant_Height_cm") is not None and r.get("Leaf_Area_cm2") is not None
        ),
        "check": lambda r: not (r["Plant_Height_cm"] < 5 and r["Leaf_Area_cm2"] > 100),
        "message": "Suspicious height-leaf area combination",
        "severity": "warning",
    },
    {
        "name": "organic_matter_carbon_ratio",
        "condition": lambda r: (
            r.get("Organic_Matter") is not None and r.get("Organic_Carbon") is not None
            and r["Organic_Carbon"] > 0
        ),
        "check": lambda r: 1.0 <= (r["Organic_Matter"] / r["Organic_Carbon"]) <= 2.5,
        "message": "OM/OC ratio outside 1.0-2.5 (expected ~1.724)",
        "severity": "warning",
    },
    {
        "name": "fertilizer_rate_sanity",
        "condition": lambda r: r.get("Nitrogen") is not None,
        "check": lambda r: r["Nitrogen"] <= 300,
        "message": "N rate > 300 kg/ha is unusually high",
        "severity": "warning",
    },
    {
        "name": "harvest_index_plausibility",
        "condition": lambda r: r.get("Harvest_Index") is not None,
        "check": lambda r: 0.15 <= r["Harvest_Index"] <= 0.65,
        "message": "Harvest index outside 0.15-0.65 range",
        "severity": "warning",
    },
]


class ValidationAgent(BaseAgent):
    """Validates evidence, traces provenance, harmonises units, and runs quality checks."""

    def __init__(
        self,
        settings: Optional[AgriAISettings] = None,
        retry_max: Optional[int] = None,
        retry_delay: Optional[float] = None,
        extractions_path: Optional[Path] = None,
        doc_struct_path: Optional[Path] = None,
    ):
        super().__init__(settings, retry_max, retry_delay)
        self.extractions_path = extractions_path or self.settings.OUTPUT_DIR / "Scientific_Extractions.json"
        self.doc_struct_path = doc_struct_path or self.settings.OUTPUT_DIR / "Document_Structures.json"

    @property
    def agent_name(self) -> str:
        return "ValidationAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        extraction_records = self._load_json(self.extractions_path, [])
        doc_structures = self._load_json(self.doc_struct_path, [])

        validated = self._validate_evidence(extraction_records, doc_structures)
        self._save_validated(validated)

        provenance = self._build_provenance(validated, doc_structures, df)
        self._save_provenance(provenance)

        conversions = self._harmonize_units(df)
        self._save_unit_report(conversions)

        issues = self._quality_checks(df)
        self._save_quality_reports(issues, df)

        if self.contract is not None:
            self.contract.metadata = {
                "facts_validated": len(extraction_records),
                "provenance_records": len(provenance),
                "conversions": len(conversions),
                "quality_issues": {k: len(v) if isinstance(v, list) else v for k, v in issues.items()},
            }
        return df

    # ── Evidence validation ──────────────────────────────────────────────

    def _validate_evidence(self, extractions: list, doc_structures: list) -> list:
        validated = []
        for ext in extractions:
            evidence = self._cross_validate(ext, doc_structures)
            if evidence.get("contradicted"):
                validated.append({**ext, "validation": "REJECTED", "reason": evidence.get("reason", "")})
            else:
                new_conf = self._compute_confidence(ext, evidence)
                validated.append({**ext, "validation": "ACCEPTED", "adjusted_confidence": new_conf})
        return validated

    def _cross_validate(self, ext: dict, doc_structures: list) -> dict:
        var = ext.get("variable", "")
        value = str(ext.get("value", ""))
        paper = ext.get("paper", "")
        if not value:
            return {"contradicted": True, "reason": "Empty value"}
        for struct in doc_structures:
            if struct.get("filename") != paper:
                continue
            tables = struct.get("tables", [])
            match_count = 0
            for table in tables:
                caption = table.get("caption", "")
                if var.lower() in caption.lower() and value.lower() in caption.lower():
                    match_count += 1
            if match_count > 0:
                return {"contradicted": False, "supporting_mentions": match_count}
        return {"contradicted": False, "supporting_mentions": 0}

    def _compute_confidence(self, ext: dict, evidence: dict) -> str:
        mentions = evidence.get("supporting_mentions", 0)
        source_text = ext.get("source_text", "")
        if mentions > 1 or (mentions > 0 and len(source_text) > 20):
            return "high"
        if mentions > 0 or len(source_text) > 10:
            return "medium"
        return ext.get("confidence", "low")

    # ── Provenance / traceability ────────────────────────────────────────

    def _build_provenance(self, validated: list, doc_structures: list, df: pd.DataFrame) -> list:
        doi_map = {s.get("filename", ""): s.get("doi", "") for s in doc_structures}
        records = []
        for v in validated:
            paper = v.get("paper", "")
            records.append({
                "paper": paper,
                "doi": doi_map.get(paper, ""),
                "variable": v.get("variable", ""),
                "value": v.get("value", ""),
                "unit": v.get("unit", ""),
                "page": v.get("page", 1),
                "table": v.get("table", ""),
                "row": v.get("row", ""),
                "column": v.get("column", ""),
                "figure": v.get("figure", ""),
                "caption": v.get("source_text", "")[:200],
                "extraction_timestamp": datetime.now().isoformat(),
                "agent_version": "1.0.0",
                "extraction_method": "regex_pattern",
                "confidence": v.get("adjusted_confidence", v.get("confidence", "low")),
                "validation": v.get("validation", "ACCEPTED"),
                "source_text": v.get("source_text", "")[:200],
                "char_start": v.get("char_start", 0),
                "char_end": v.get("char_end", 0),
            })
        skip_cols = {
            "Crop_Code", "Season_Code", "Variety_Code", "Fertilizer_Code",
            "Soil_Texture_Code", "Country_Code", "Feature_Available_Before_Prediction",
        }
        if df is not None:
            known_vars = {r["variable"] for r in records}
            for col in df.columns:
                if col not in known_vars and col not in skip_cols:
                    records.append({
                        "paper": "pipeline_input", "doi": "", "variable": col,
                        "value": "", "unit": "", "page": 0, "table": "", "row": "",
                        "column": "", "figure": "",
                        "caption": "Column from pipeline input dataset",
                        "extraction_timestamp": datetime.now().isoformat(),
                        "agent_version": "1.0.0",
                        "extraction_method": "dataset_import",
                        "confidence": "high", "validation": "ACCEPTED",
                        "source_text": "", "char_start": 0, "char_end": 0,
                    })
        return records

    # ── Unit harmonisation ───────────────────────────────────────────────

    def _harmonize_units(self, df: pd.DataFrame) -> list:
        conversions = []
        conversions.extend(self._convert_yield(df))
        conversions.extend(self._convert_temperature(df))
        conversions.extend(self._convert_column_units(df))
        conversions.extend(self._convert_fertilizer_rates(df))
        return conversions

    def _convert_yield(self, df: pd.DataFrame) -> list:
        conv = []
        if "Yield_per_Acre" in df.columns and "Yield_per_Hectare" not in df.columns:
            df["Yield_per_Hectare"] = pd.to_numeric(df["Yield_per_Acre"], errors="coerce") * 2.47105
            conv.append({"from": "Yield_per_Acre", "to": "Yield_per_Hectare",
                         "factor": 2.47105, "type": "yield",
                         "rows_affected": int(df["Yield_per_Acre"].notna().sum())})
        if "Yield_per_Plot" in df.columns:
            plot_col = next((c for c in ["Plot_Size", "plot_size", "plot_size_m2"] if c in df.columns), None)
            if plot_col is not None and "Yield_per_Hectare" not in df.columns:
                plot_size = pd.to_numeric(df[plot_col], errors="coerce")
                yield_plot = pd.to_numeric(df["Yield_per_Plot"], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    df["Yield_per_Hectare_Calc"] = np.where(plot_size > 0, yield_plot / plot_size * 10000, np.nan)
                conv.append({"from": f"Yield_per_Plot + {plot_col}", "to": "Yield_per_Hectare_Calc",
                             "factor": "10000 / plot_size_m2", "type": "yield",
                             "rows_affected": int(yield_plot.notna().sum())})
        return conv

    def _convert_temperature(self, df: pd.DataFrame) -> list:
        conv = []
        for col in ["Temperature_Max", "Temperature_Min", "Average_Temperature"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().any() and vals.max() > 100:
                    df[col] = (vals - 32) * 5 / 9
                    conv.append({"from": f"{col} (°F)", "to": f"{col} (°C)",
                                 "factor": "(°F - 32) × 5/9", "type": "temperature",
                                 "rows_affected": int(vals.notna().sum())})
        return conv

    def _convert_column_units(self, df: pd.DataFrame) -> list:
        conv = []
        for col in df.columns:
            col_lower = col.lower()
            numeric = df[col].dtype in (np.float64, np.int64, float, int)
            if "inch" in col_lower and numeric:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 2.54
                conv.append({"from": col, "to": col.replace("inch", "cm").replace("inches", "cm"),
                             "factor": 2.54, "type": "length",
                             "rows_affected": int(df[col].notna().sum())})
            if "lb" in col_lower and numeric:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 0.453592
                conv.append({"from": col, "to": col + "_kg", "factor": 0.453592,
                             "type": "mass", "rows_affected": int(df[col].notna().sum())})
        return conv

    def _convert_fertilizer_rates(self, df: pd.DataFrame) -> list:
        conv = []
        for col in ["Nitrogen", "Phosphorus", "Potassium"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                if vals.notna().any() and vals.max() < 1:
                    df[col] = vals * 1000
                    conv.append({"from": f"{col} (low kg/ha)", "to": f"{col} (g/ha)",
                                 "factor": 1000, "type": "mass",
                                 "rows_affected": int(vals.notna().sum())})
        return conv

    # ── Quality assurance ────────────────────────────────────────────────

    def _quality_checks(self, df: pd.DataFrame) -> dict:
        issues: dict = {
            "duplicate_rows": 0, "duplicate_columns": [], "impossible_values": [],
            "outliers": [], "missing_identifiers": [], "negative_values": [],
            "range_violations": [], "unit_inconsistencies": [],
            "ontology_inconsistencies": [],
            "biological_violations": [],
            "agronomic_violations": [],
            "outliers_mad": [],
        }
        issues["duplicate_rows"] = int(df.duplicated().sum())
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        issues["duplicate_columns"] = list(set(dup_cols))

        for col in ["Paper_ID", "DOI", "Experiment_ID", "Plot_ID", "Sample_ID"]:
            if col in df.columns:
                missing = int(df[col].isna().sum())
                if missing > 0:
                    issues["missing_identifiers"].append(f"{col}: {missing} missing")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            neg = int((df[col] < 0).sum())
            if neg > 0:
                issues["negative_values"].append(f"{col}: {neg} negative(s)")

        for col, (lo, hi) in RANGE_CONSTRAINTS.items():
            if col in df.columns:
                vals = df[col].dropna()
                bad = int((vals < lo).sum() + (vals > hi).sum())
                if bad:
                    issues["impossible_values"].append(f"{col}: {bad} outside [{lo}, {hi}]")
                    issues["range_violations"].append({"column": col, "min": lo, "max": hi, "violations": bad})

        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) > 3:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = int(((vals < lower) | (vals > upper)).sum())
                    if outliers > 0:
                        issues["outliers"].append(f"{col}: {outliers} outlier(s)")

        issues["biological_violations"] = self._validate_biological_rules(df)
        issues["agronomic_violations"] = self._validate_agronomic_rules(df)
        issues["outliers_mad"] = self._detect_outliers_mad(df)
        issues["unit_inconsistencies"] = self._check_unit_consistency(df)
        issues["ontology_inconsistencies"] = self._check_ontology_consistency(df)
        return issues

    def _validate_biological_rules(self, df: pd.DataFrame) -> list:
        violations = []
        for col, rules in BIOLOGICAL_RULES.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            lo, hi = rules["min"], rules["max"]
            n_outside = int(((vals < lo) | (vals > hi)).sum())
            if n_outside > 0:
                violations.append(
                    f"{col}: {n_outside} values outside biological range "
                    f"[{lo}, {hi}] {rules['unit']}"
                )
        return violations

    def _validate_agronomic_rules(self, df: pd.DataFrame) -> list:
        violations = []
        for _, row in df.iterrows():
            for rule in AGRONOMIC_RULES:
                try:
                    if rule["condition"](row) and not rule["check"](row):
                        violations.append(
                            f"Row {row.name}: {rule['message']} [{rule['severity']}]"
                        )
                except Exception as e:
                    self.log.debug("Agronomic rule %s failed: %s", rule.get("name", rule.get("message", "?")), e)
        if len(violations) > 50:
            violations = violations[:50] + [f"... and {len(violations) - 50} more"]
        return violations

    def _detect_outliers_mad(self, df: pd.DataFrame, threshold: float = 3.5) -> list:
        outliers = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            vals = df[col].dropna()
            if len(vals) < 5:
                continue
            median = vals.median()
            mad = np.median(np.abs(vals - median))
            if mad == 0:
                continue
            modified_z = 0.6745 * (vals - median) / mad
            n_outliers = int((np.abs(modified_z) > threshold).sum())
            if n_outliers > 0:
                outliers.append(f"{col}: {n_outliers} MAD outliers (z>{threshold})")
        return outliers

    def _check_unit_consistency(self, df: pd.DataFrame) -> list:
        issues = []
        if "Yield_per_Hectare" in df.columns and "Yield_per_Acre" in df.columns:
            yh = pd.to_numeric(df["Yield_per_Hectare"], errors="coerce")
            ya = pd.to_numeric(df["Yield_per_Acre"], errors="coerce")
            ratio = yh / ya
            expected = 2.47105
            bad = int((ratio.notna() & (np.abs(ratio - expected) > 0.1 * expected)).sum())
            if bad:
                issues.append(f"Yield_per_Hectare/Yield_per_Acre ratio off: {bad} rows")
        return issues

    def _check_ontology_consistency(self, df: pd.DataFrame) -> list:
        issues = []
        if "Temperature_Max" in df.columns and "Temperature_Min" in df.columns:
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
            bad = int((tmax < tmin).sum())
            if bad:
                issues.append(f"Tmax < Tmin in {bad} rows")
        if all(c in df.columns for c in ["Average_Temperature", "Temperature_Max", "Temperature_Min"]):
            tavg = pd.to_numeric(df["Average_Temperature"], errors="coerce")
            tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
            tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
            expected = (tmax + tmin) / 2
            bad = int(((tavg.notna()) & (expected.notna()) & (np.abs(tavg - expected) > 5)).sum())
            if bad:
                issues.append(f"Average_Temperature ≠ (Tmax+Tmin)/2 in {bad} rows")
        return issues

    # ── I/O helpers & reports ────────────────────────────────────────────

    def _load_json(self, path: Path, default):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.log.warning("Could not load %s", path)
        return default

    def _save_validated(self, validated: list):
        path = self.settings.OUTPUT_DIR / "Validated_Extractions.json"
        path.write_text(json.dumps(validated, indent=2, default=str), encoding="utf-8")
        if self.contract is not None:
            self.contract.artifacts.append(str(path))
        counts = {"accepted": 0, "rejected": 0}
        for v in validated:
            counts["rejected" if v.get("validation") == "REJECTED" else "accepted"] += 1
        self.save_text_artifact(self._validation_report(validated, counts), "Evidence_Validation_Report.md")

    def _validation_report(self, validated: list, counts: dict) -> str:
        lines = [
            "# Evidence Validation Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total: {len(validated)} | Accepted: {counts['accepted']} | Rejected: {counts['rejected']}",
            "",
        ]
        for v in validated[:50]:
            lines.append(
                f"- {v.get('paper')}: {v.get('variable')} = {v.get('value')} "
                f"[{v.get('validation')}] conf: {v.get('adjusted_confidence', v.get('confidence'))}"
            )
        if len(validated) > 50:
            lines.append(f"- ... and {len(validated) - 50} more")
        return "\n".join(lines)

    def _save_provenance(self, records: list):
        path = self.settings.OUTPUT_DIR / "Provenance_Registry.json"
        path.write_text(
            json.dumps({"pipeline": "AgriAI v1.0", "generated_at": datetime.now().isoformat(),
                        "provenance_records": records}, indent=2, default=str),
            encoding="utf-8",
        )
        if self.contract is not None:
            self.contract.artifacts.append(str(path))

    def _save_unit_report(self, conversions: list):
        if not conversions:
            return
        lines = ["# Unit Conversion Report", f"Generated: {datetime.now().isoformat()}", ""]
        for c in conversions:
            lines.append(f"- {c['from']} → {c['to']} (×{c['factor']}): {c['rows_affected']} rows [{c['type']}]")
        self.save_text_artifact("\n".join(lines), "Unit_Conversion_Report.md")

    def _save_quality_reports(self, issues: dict, df: pd.DataFrame):
        lines = [
            "# Quality Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Rows: {len(df)} | Cols: {len(df.columns)}",
            f"Duplicates: {issues['duplicate_rows']} rows, {len(issues['duplicate_columns'])} cols",
        ]
        for key in ("impossible_values", "outliers", "outliers_mad", "missing_identifiers",
                     "negative_values", "unit_inconsistencies", "ontology_inconsistencies",
                     "biological_violations", "agronomic_violations"):
            items = issues.get(key, [])
            if items:
                lines.append(f"")
                lines.append(f"## {key.replace('_', ' ').title()}")
                for item in items[:20]:
                    lines.append(f"- {item}")
                if len(items) > 20:
                    lines.append(f"- ... and {len(items) - 20} more")
        self.save_text_artifact("\n".join(lines), "Quality_Report.md")

        rows = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            issues_text = []
            for v in issues["impossible_values"]:
                if col in v:
                    issues_text.append(v)
            for v in issues["outliers"]:
                if col in v.split(":")[0]:
                    issues_text.append(v)
            rows.append({
                "Column": col,
                "Data_Type": str(df[col].dtype),
                "Non_Null_Count": len(df) - missing,
                "Null_Count": missing,
                "Null_Pct": round(missing / len(df) * 100, 2) if len(df) else 0,
                "Unique_Values": int(df[col].nunique()),
                "Issues": "; ".join(issues_text),
            })
        self.save_artifact(pd.DataFrame(rows), "Validation_Report.csv")
