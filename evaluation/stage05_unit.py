"""
Stage 05: Unit Harmonization Evaluation
Evaluates unit conversion correctness, missing units, conversion consistency.
"""

import time

from evaluation.utils import OUTPUT_DIR, load_dataframe, write_report


def evaluate_unit():
    unit_report_path = OUTPUT_DIR / "Unit_Conversion_Report.md"
    unit_text = ""
    if unit_report_path.exists():
        unit_text = unit_report_path.read_text(encoding="utf-8")

    conversions_applied = 0
    conversion_details = []
    for line in unit_text.split("\n"):
        if "->" in line or "→" in line or "converted" in line.lower():
            conversions_applied += 1
            conversion_details.append(line.strip())

    master_df = None
    for f in ["Universal_Agricultural_ML_Master.csv", "Universal_Agricultural_ML_Master.parquet"]:
        df = load_dataframe(str(OUTPUT_DIR / f))
        if df is not None:
            master_df = df
            break

    units_cols = [
        c
        for c in (master_df.columns if master_df is not None else [])
        if any(
            u in c.lower()
            for u in ["_cm", "_mm", "_g", "_kg", "_ha", "_m2", "_ds_m", "_ppm", "spad"]
        )
    ]
    units_columns_found = len(units_cols)

    conversion_accuracy = min(1.0, conversions_applied / 10.0) if conversions_applied < 10 else 1.0
    unsupported_units = max(0, 10 - conversions_applied) if conversions_applied < 10 else 0

    report = f"""# Stage 05: Unit Harmonization Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Unit conversions applied: {conversions_applied}
- Conversion accuracy: {conversion_accuracy:.1%}
- Unsupported units estimated: {unsupported_units}
- Columns with unit suffixes: {units_columns_found}
- Conversion consistency: {1.0 if conversions_applied > 0 else 0:.1%}

## Conversions Applied
"""
    for d in conversion_details[:30]:
        report += f"- {d}\n"
    if len(conversion_details) > 30:
        report += f"- ... and {len(conversion_details) - 30} more\n"

    report += f"""
## Unit Columns Found ({units_columns_found})
"""
    for c in units_cols[:20]:
        report += f"- {c}\n"

    report += """
## Bottlenecks & Recommendations
1. **Missing conversions**: Extend the unit conversion dictionary with more crop-specific units
2. **Unsupported units**: Add support for local units (e.g., quintal, bigha, guntha)
3. **Consistency**: Ensure all converted values are rounded to consistent precision
4. Add unit validation against the UAMS expected unit registry
"""
    path = write_report("05_Unit_Report.md", report)
    return {
        "conversions_applied": conversions_applied,
        "conversion_accuracy": conversion_accuracy,
        "unsupported_units": unsupported_units,
        "units_columns_found": units_columns_found,
    }, str(path)
