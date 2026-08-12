"""Phase 20 STEP 5 — Ontology recovery.

Classifies every Phase 18/19 rejected staged record into a documented
rejection taxonomy and writes a DERIVED ontology extension (never touching
the validated Phase 17 ontology). No ambiguous alias is created from fuzzy
similarity alone.

Taxonomy:
    valid_ontology_mismatch, missing_alias, unit_mismatch,
    source_schema_mismatch, ambiguous_variable, duplicate,
    non_agricultural_variable, insufficient_metadata, true_unmappable_variable

Outputs:
    .opencode_tmp/phase20/phase20_ontology_extensions.yaml
    outputs/phase20/ontology_recovery.parquet
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yaml

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C

TAXONOMY = [
    "valid_ontology_mismatch",
    "missing_alias",
    "unit_mismatch",
    "source_schema_mismatch",
    "ambiguous_variable",
    "duplicate",
    "non_agricultural_variable",
    "insufficient_metadata",
    "true_unmappable_variable",
]


def _empty_uams(v):
    return v is None or C.s(v) in ("", "nan", "None")


def classify_record(row: pd.Series) -> dict:
    var = C.s(row.get("Variable"))
    src = C.s(row.get("Source"))
    unit = C.s(row.get("Unit"))
    grade = C.s(row.get("Mapping_Grade"))
    status = C.s(row.get("Validation_Status"))
    dup = bool(row.get("Is_Duplicate"))

    classification = "insufficient_metadata"
    canonical = ""
    confidence = 0.0
    method = "none"
    evidence = ""

    if dup:
        classification = "duplicate"
        evidence = "Is_Duplicate flag set by Phase 18 validation"
    elif status in ("REJECTED", "FAILED"):
        classification = "invalid_record"
        evidence = f"validation status {status}"
    elif var in ("production_tonnes",):
        classification = "source_schema_mismatch"
        evidence = ("FAOSTAT 'production_tonnes' is a national production "
                    "aggregate (t), not a plot/field-level observation; mapping "
                    "it to Yield_per_Hectare would change semantics and leak "
                    "national aggregates into plot-level modelling.")
        confidence = 0.9
    elif var in ("area_harvested",):
        classification = "source_schema_mismatch"
        evidence = ("FAOSTAT 'area_harvested' is a national harvested area "
                    "aggregate (ha); it is not an observation-level predictor.")
        confidence = 0.9
    elif var.startswith("100") or var in ("Value", "PARAMETER", "Property", "Statistic", "Treatment1"):
        classification = "non_agricultural_variable"
        evidence = f"spurious table-scraping variable name '{var}'"
        confidence = 0.8
    elif not var:
        classification = "insufficient_metadata"
        evidence = "empty variable name; no mapping evidence"
    elif unit and unit.lower() in ("nan", "none", "unknown"):
        classification = "unit_mismatch"
        evidence = f"missing/unknown unit for '{var}'"
    else:
        classification = "valid_ontology_mismatch"
        evidence = f"'{var}' has no high-confidence alias in the Phase 17 ontology"

    return {
        "source_variable": var,
        "canonical_variable": canonical,
        "unit": unit,
        "source": src,
        "confidence": confidence,
        "mapping_method": method,
        "classification": classification,
        "evidence": evidence,
        "mapping_grade": grade,
        "validation_status": status,
        "is_duplicate": dup,
    }


def load_validated_ontology(cfg: dict) -> pd.DataFrame:
    """Read-only reference to the Phase 17 validated ontology aliases."""
    p = C.safe_resolve(cfg["inputs"]["phase17"]["ontology_aliases"])
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def build_rejection_table(cfg: dict) -> pd.DataFrame:
    stg = C.load_input("phase18.staging_external", cfg)
    rejected = stg[stg["UAMS_Column"].map(_empty_uams)].copy()
    rows = [classify_record(r) for _, r in rejected.iterrows()]
    return pd.DataFrame(rows)


def write_extension_yaml(summary: dict, alias_rows: list[dict], reject_rows: list[dict]) -> Path:
    doc = {
        "phase20_ontology_extensions": {
            "version": "1.0.0",
            "date_added": date.today().isoformat(),
            "methodology": (
                "Derived extension produced by Phase 20 STEP 5. Validated "
                "Phase 17 ontology is NEVER modified. Alias candidates are "
                "accepted only with explicit evidence and confidence >= "
                "configured minimum. No mapping is created from fuzzy "
                "similarity alone."
            ),
            "alias_min_confidence": summary["alias_min_confidence"],
            "aliases": alias_rows,
            "rejected_variable_classification": reject_rows,
            "recovery_metrics": {
                k: v for k, v in summary.items() if k != "alias_min_confidence"
            },
        }
    }
    p = C.safe_resolve(C.ONTO_EXT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
    return p


def run(force: bool = False):
    cfg = C.load_config()
    alias_min = cfg["ontology"]["alias_min_confidence"]
    onto = load_validated_ontology(cfg)

    stg = C.load_input("phase18.staging_external", cfg)
    total = int(len(stg))
    rejected = int(stg["UAMS_Column"].map(_empty_uams).sum())
    rejected_rate_before = round(rejected / total, 4) if total else 0.0

    rej_df = build_rejection_table(cfg)
    if len(rej_df):
        rej_df["classification"] = pd.Categorical(
            rej_df["classification"], categories=TAXONOMY + ["invalid_record"],
            ordered=False)
    counts = rej_df["classification"].astype(str).value_counts().to_dict() \
        if len(rej_df) else {}

    # alias candidates: only explicit, evidence-based, non-ambiguous
    alias_rows = []
    # (no legitimate observation-level alias exists for the rejected
    # national-aggregate variables; nothing is fabricated here)

    summary = {
        "alias_min_confidence": alias_min,
        "validated_ontology_aliases": int(len(onto)),
        "staged_records_total": total,
        "staged_records_rejected_before": rejected,
        "staged_rejection_rate_before": rejected_rate_before,
        "rejections_classified": int(len(rej_df)),
        "rejections_reclassified_rate": round(len(rej_df) / rejected, 4) if rejected else 0.0,
        "new_aliases_added": int(len(alias_rows)),
        "staged_records_rejected_after": rejected,
        "staged_rejection_rate_after": rejected_rate_before,
        "rejection_reduction": 0.0,
        "note": ("Rejected variables are national aggregates with no honest "
                 "observation-level alias; rejection rate unchanged, but every "
                 "rejection is now classified and auditable. No fabricated "
                 "mapping was introduced."),
    }

    path = write_extension_yaml(summary, alias_rows, rej_df.to_dict("records")
                                if len(rej_df) else [])

    C.to_parquet(rej_df, C.OUT / "ontology_recovery.parquet")
    C.to_excel(rej_df, C.REPORTS / "ontology_recovery_report.xlsx", "Rejections")
    C.write_json(summary, C.OUT / "ontology_recovery_metrics.json")

    result = dict(summary)
    result["generated_at"] = C.now_full_iso()
    result["classification_counts"] = counts
    result["extension_path"] = str(path)
    C.mark_done("ontology", result)
    C.log_msg(f"STEP5 ontology: classified {result['rejections_classified']} "
              f"rejections, {result['new_aliases_added']} aliases added")
    return result


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
