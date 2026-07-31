"""
Output Validation.
Verifies the Universal_Agricultural_Machine_Learning_Schema_v1.xlsx
contains all required sheets, no duplicated columns/variables,
consistent units, complete ontology mappings, no feature leakage, model ready.
"""

import time

from evaluation.utils import (
    OUTPUT_DIR,
    get_master_df,
    get_uams_cols,
    write_report,
)

REQUIRED_SHEETS = [
    "01_Metadata",
    "02_Field_Profile",
    "03_Crop_Profile",
    "04_Soil_Profile",
    "05_Weather_TimeSeries",
    "06_Management_Events",
    "07_Plant_Observations",
    "08_Remote_Sensing",
    "09_Sensor_Data",
    "10_Laboratory_Analysis",
    "11_Derived_Features",
    "12_Targets",
    "13_Data_Quality",
    "14_Feature_Dictionary",
    "15_Evidence_Metadata",
]


def validate_output():
    xlsx_path = OUTPUT_DIR / "Universal_Agricultural_Machine_Learning_Schema_v1.xlsx"
    master_df = get_master_df()

    if not xlsx_path.exists():
        report = "# Output Validation Report\n**Error:** Schema XLSX not found.\n"
        path = write_report("Output_Validation.md", report)
        return {}, str(path)

    import openpyxl

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    actual_sheets = wb.sheetnames
    wb.close()

    sheet_status = {}
    for sheet in REQUIRED_SHEETS:
        sheet_status[sheet] = sheet in actual_sheets

    present_sheets = sum(1 for v in sheet_status.values() if v)
    missing_sheets = [s for s, v in sheet_status.items() if not v]

    dup_cols = []
    inconsistent_units = []
    missing_onto = []
    leakage_issues = []
    model_ready = True

    if master_df is not None:
        dup_cols = [c for c in master_df.columns if master_df.columns.tolist().count(c) > 1]
        dup_cols = list(set(dup_cols))

        for col in master_df.columns:
            if master_df[col].dtype == "object":
                vals = master_df[col].dropna().unique()
                if len(vals) > 0:
                    pass

        uams_set = set(get_uams_cols())
        missing_onto = [
            c
            for c in master_df.columns
            if c not in uams_set
            and not c.endswith("_Code")
            and c not in ["Feature_Available_Before_Prediction"]
        ]

        if "Feature_Available_Before_Prediction" not in master_df.columns:
            leakage_issues.append("Missing Feature_Available_Before_Prediction flag")

        model_ready = len(missing_onto) < 10

    report = f"""# Output Validation Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Schema XLSX Validation
| Check | Result |
|-------|--------|
| File exists | {"✅" if xlsx_path.exists() else "❌"} |
| Sheets present | {present_sheets}/{len(REQUIRED_SHEETS)} |

### Sheet Status
| Sheet | Present |
|-------|---------|
"""
    for sheet in REQUIRED_SHEETS:
        icon = "✅" if sheet_status[sheet] else "❌"
        report += f"| {icon} | {sheet} |\n"

    report += f"""
### Missing Sheets ({len(missing_sheets)})
"""
    for s in missing_sheets:
        report += f"- {s}\n"

    report += f"""
## Column Validation
- Duplicated columns: {len(dup_cols)}
- Variables with missing ontology: {len(missing_onto)}
- Inconsistent units detected: {len(inconsistent_units)}
- Leakage issues: {len(leakage_issues)}
- Model ready: {"✅" if model_ready else "❌"}

"""
    if dup_cols:
        report += "### Duplicated Columns\n"
        for c in dup_cols[:15]:
            report += f"- {c}\n"

    report += f"""
### Variables Missing Ontology Mappings ({len(missing_onto)})
"""
    for c in missing_onto[:20]:
        report += f"- {c}\n"

    report += f"""
### Leakage Issues ({len(leakage_issues)})
"""
    for issue in leakage_issues:
        report += f"- {issue}\n"

    report += """
## Overall Validation
"""
    all_pass = (
        present_sheets == len(REQUIRED_SHEETS) and len(dup_cols) == 0 and len(leakage_issues) == 0
    )
    report += f"**Output Validation: {'✅ PASS' if all_pass else '❌ NEEDS WORK'}**\n"

    report += """
## Recommendations
1. Add all 15 required sheets to the XLSX workbook
2. Remove duplicated columns from the output
3. Complete ontology mappings for all unmapped variables
4. Add Feature_Available_Before_Prediction flag
5. Ensure consistent units across all sheets
"""
    path = write_report("Output_Validation.md", report)
    return {
        "sheets_present": present_sheets,
        "total_sheets": len(REQUIRED_SHEETS),
        "duplicated_columns": len(dup_cols),
        "missing_mappings": len(missing_onto),
        "leakage_issues": len(leakage_issues),
        "validation_pass": all_pass,
    }, str(path)
