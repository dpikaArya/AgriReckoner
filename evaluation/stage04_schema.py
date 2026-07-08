"""
Stage 04: Schema Mapping Evaluation
Evaluates variable mapping correctness, duplicate detection, synonym resolution, column normalization.
"""

import time
from pathlib import Path

from evaluation.utils import OUTPUT_DIR, write_report, load_dataframe, get_master_df, get_uams_cols


def evaluate_schema():
    uams_cols = get_uams_cols()
    uams_set = set(uams_cols)
    uams_count = len(uams_cols)

    variable_map_path = OUTPUT_DIR / "Variable_Mapping.csv"
    unmapped_path = OUTPUT_DIR / "Unmapped_Columns.csv"
    master_df = get_master_df()

    mapped_vars = {}
    if variable_map_path.exists():
        vm_df = load_dataframe(str(variable_map_path))
        if vm_df is not None:
            for _, row in vm_df.iterrows():
                mapped_vars[str(row.iloc[0])] = str(row.iloc[1]) if len(row) > 1 else ""

    unmapped_vars = []
    if unmapped_path.exists():
        um_df = load_dataframe(str(unmapped_path))
        if um_df is not None:
            unmapped_vars = [str(c) for c in um_df.iloc[:, 0].tolist() if str(c) != "nan"]

    actual_cols = set(master_df.columns) if master_df is not None else set()
    schema_coverage = len(actual_cols & uams_set) / uams_count if uams_count > 0 else 0
    schema_completeness = len(actual_cols & uams_set) / len(actual_cols) if actual_cols else 0

    duplicates_in_cols = []
    if master_df is not None:
        dup_cols = master_df.columns[master_df.columns.duplicated()].tolist()
        duplicates_in_cols = list(set(dup_cols))

    mapping_accuracy = len(mapped_vars) / (len(mapped_vars) + len(unmapped_vars)) if (len(mapped_vars) + len(unmapped_vars)) > 0 else 0
    duplicate_removal = 1.0 - (len(duplicates_in_cols) / len(actual_cols)) if actual_cols else 1.0

    report = f"""# Stage 04: Schema Mapping Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
UAMS schema version: 1.0
UAMS columns: {uams_count}

## Summary
- Variable mapping accuracy: {mapping_accuracy:.1%}
- Schema coverage: {schema_coverage:.1%}
- Schema completeness: {schema_completeness:.1%}
- Duplicate removal accuracy: {duplicate_removal:.1%}
- Total columns in output: {len(actual_cols)}
- UAMS columns present: {len(actual_cols & uams_set)}
- Unmapped columns: {len(unmapped_vars)}
- Duplicate columns: {len(duplicates_in_cols)}

## Unmapped Columns ({len(unmapped_vars)})
"""
    for v in unmapped_vars[:30]:
        report += f"- {v}\n"
    if len(unmapped_vars) > 30:
        report += f"- ... and {len(unmapped_vars) - 30} more\n"

    report += f"""
## Duplicate Columns ({len(duplicates_in_cols)})
"""
    for d in duplicates_in_cols[:20]:
        report += f"- {d}\n"

    report += """
## Bottlenecks & Recommendations
1. **Unmapped columns** indicate missing entries in VARIANT_MAP or the UAMS schema
2. **Duplicate columns** suggest the ingestion step needs better column deduplication
3. Expand VARIANT_MAP to cover remaining unmapped column name variants
4. Consider automated synonym discovery using WordNet or domain thesauri
"""
    path = write_report("04_Schema_Report.md", report)
    return {
        "mapping_accuracy": mapping_accuracy,
        "schema_coverage": schema_coverage,
        "schema_completeness": schema_completeness,
        "duplicate_removal": duplicate_removal,
        "unmapped_count": len(unmapped_vars),
        "duplicate_count": len(duplicates_in_cols),
    }, str(path)
