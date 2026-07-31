"""
Stage 04: Schema Mapping Evaluation
Evaluates variable mapping correctness, duplicate detection, synonym resolution, column normalization.
Distinguishes core UAMS (A-N) from Data ADES extensions (O-Z).
"""

import time
from pathlib import Path

from evaluation.utils import OUTPUT_DIR, get_master_df, get_uams_cols, load_dataframe, write_report

CORE_GROUPS = [
    "A. Paper Metadata",
    "B. Crop Information",
    "C. Experimental Design",
    "D. Environment",
    "E. Soil Properties",
    "F. Fertilizer Information",
    "G. Crop Growth Parameters",
    "H. Yield Parameters",
    "I. Grain Quality",
    "J. ML Target Variables",
    "K. Engineered Features",
    "L. Leakage Labels",
    "M. Encoded Variables",
    "N. ML Predictions",
]


def _get_core_extended_cols():
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agri_ai_agent.config.schema import SCHEMA_GROUPS

        core = []
        extended = []
        for g, cols in SCHEMA_GROUPS.items():
            if any(
                g.startswith(prefix)
                for prefix in [
                    "A.",
                    "B.",
                    "C.",
                    "D.",
                    "E.",
                    "F.",
                    "G.",
                    "H.",
                    "I.",
                    "J.",
                    "K.",
                    "L.",
                    "M.",
                    "N.",
                ]
            ):
                core.extend(cols)
            else:
                extended.extend(cols)
        return core, extended
    except ImportError:
        return [], []


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

    # Core vs Extended breakdown
    core_cols, extended_cols = _get_core_extended_cols()
    core_set = set(core_cols)
    ext_set = set(extended_cols)
    core_present = len(actual_cols & core_set)
    ext_present = len(actual_cols & ext_set)
    core_coverage = core_present / len(core_cols) if core_cols else 0
    ext_coverage = ext_present / len(extended_cols) if extended_cols else 0

    duplicates_in_cols = []
    if master_df is not None:
        dup_cols = master_df.columns[master_df.columns.duplicated()].tolist()
        duplicates_in_cols = list(set(dup_cols))

    mapping_accuracy = (
        len(mapped_vars) / (len(mapped_vars) + len(unmapped_vars))
        if (len(mapped_vars) + len(unmapped_vars)) > 0
        else 0
    )
    duplicate_removal = 1.0 - (len(duplicates_in_cols) / len(actual_cols)) if actual_cols else 1.0

    report = f"""# Stage 04: Schema Mapping Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
UAMS schema version: 2.0
UAMS columns: {uams_count} (Core A-N: {len(core_cols)}, Extended O-Z: {len(extended_cols)})

## Summary
- Variable mapping accuracy: {mapping_accuracy:.1%}
- Overall schema coverage: {schema_coverage:.1%}
- Core schema coverage (A-N): {core_coverage:.1%} ({core_present}/{len(core_cols)})
- Extended schema coverage (O-Z): {ext_coverage:.1%} ({ext_present}/{len(extended_cols)})
- Schema completeness: {schema_completeness:.1%}
- Duplicate removal accuracy: {duplicate_removal:.1%}
- Total columns in output: {len(actual_cols)}
- UAMS columns present: {len(actual_cols & uams_set)}
- Unmapped columns: {len(unmapped_vars)}
- Duplicate columns: {len(duplicates_in_cols)}

## Core Schema Groups (A-N)
"""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agri_ai_agent.config.schema import SCHEMA_GROUPS

        for g_name in CORE_GROUPS:
            g_cols = SCHEMA_GROUPS.get(g_name, [])
            present = sum(1 for c in g_cols if c in actual_cols)
            pct = present / len(g_cols) * 100 if g_cols else 0
            report += f"- {g_name}: {present}/{len(g_cols)} ({pct:.0f}%)\n"
    except ImportError:
        report += "- (schema groups not available)\n"

    report += f"""
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
1. **Extended schema (O-Z)**: Data ADES extensions require specialized source data (soil enzymes, microbial communities, remote sensing, nematodes)
2. **Core schema**: Focus on populating remaining core columns from available paper data
3. Expand VARIANT_MAP to cover remaining unmapped column name variants
4. Consider automated synonym discovery using WordNet or domain thesauri
"""
    path = write_report("04_Schema_Report.md", report)
    return {
        "mapping_accuracy": mapping_accuracy,
        "schema_coverage": schema_coverage,
        "core_coverage": core_coverage,
        "ext_coverage": ext_coverage,
        "schema_completeness": schema_completeness,
        "duplicate_removal": duplicate_removal,
        "unmapped_count": len(unmapped_vars),
        "duplicate_count": len(duplicates_in_cols),
    }, str(path)
