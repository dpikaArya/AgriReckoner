"""
Stage 08: Leakage Detection Evaluation
Evaluates harvest variable detection, future variable detection, target leakage.
"""

import time

from evaluation.utils import OUTPUT_DIR, get_master_df, load_dataframe, write_report

LEAKED_CANDIDATES = [
    "Yield_per_Plot",
    "Yield_per_Acre",
    "Yield_per_Hectare",
    "Fruit_Number",
    "Fruit_Weight",
    "Fruit_Diameter_mm",
    "Spike_Length",
    "Seeds_per_Spike",
    "100_Seed_Weight",
    "Root_Weight",
    "Pod_Weight",
    "Harvest_Index",
    "Biomass_Yield",
    "Protein",
    "Ash",
    "Gluten",
    "Fiber",
]


def _flagged_features(leakage_df):
    """Read the flagged-feature names out of whichever report layout is present.

    The producer writes a single ``leaked_feature`` column; this stage expected a two-column
    ``Feature`` / ``Available_Before_Prediction`` layout that has never existed. The result
    was an empty list on every run, so the stage reported no leakage detected while the guard
    was in fact flagging 39 features — a certificate that could not go red.
    """
    if leakage_df is None or leakage_df.empty:
        return []
    columns = {c.strip().lstrip("﻿"): c for c in leakage_df.columns}
    if "leaked_feature" in columns:
        return leakage_df[columns["leaked_feature"]].dropna().astype(str).tolist()
    if "Feature" in columns and "Available_Before_Prediction" in columns:
        available = leakage_df[columns["Available_Before_Prediction"]].astype(bool)
        return leakage_df.loc[~available, columns["Feature"]].astype(str).tolist()
    raise ValueError(
        f"Leakage report has an unrecognised layout: columns={list(leakage_df.columns)}. "
        "Expected 'leaked_feature', or 'Feature' plus 'Available_Before_Prediction'."
    )


def evaluate_leakage():
    master_df = get_master_df()
    leakage_report_path = OUTPUT_DIR / "Leakage_Report.csv"

    leakage_df = None
    if leakage_report_path.exists():
        leakage_df = load_dataframe(str(leakage_report_path))

    flagged_leaked = _flagged_features(leakage_df)

    actual_cols = set(master_df.columns) if master_df is not None else set()
    harvest_in_data = [c for c in LEAKED_CANDIDATES if c in actual_cols]

    total_leakage_possible = len(harvest_in_data)
    total_leakage_detected = len([c for c in harvest_in_data if c in flagged_leaked])

    leakage_detected_rate = (
        total_leakage_detected / total_leakage_possible if total_leakage_possible > 0 else 0
    )
    leakage_prevented = 0
    if master_df is not None and "Feature_Available_Before_Prediction" in master_df.columns:
        # Cast first: the column arrives as float64, and inverting a float raises
        # "ufunc 'invert' not supported", which the runner then swallowed as a printed line.
        available = master_df["Feature_Available_Before_Prediction"].fillna(False).astype(bool)
        flagged_available = master_df[~available]
        leakage_prevented = flagged_available.shape[0] if flagged_available is not None else 0

    false_positives = max(0, len(flagged_leaked) - total_leakage_possible)

    report = f"""# Stage 08: Leakage Detection Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Harvest/post-harvest variables in data: {total_leakage_possible}
- Leakage detected: {total_leakage_detected}
- Leakage prevented (flagged): {leakage_prevented or 0}
- False positives: {false_positives}
- Leakage detection rate: {leakage_detected_rate:.1%}

## Variables Flagged as Leakage ({len(flagged_leaked)})
"""
    for v in flagged_leaked[:30]:
        report += f"- {v}\n"

    report += f"""
## Harvest Variables in Dataset (Potential Leakage: {total_leakage_possible})
"""
    for v in harvest_in_data[:30]:
        report += f"- {v}\n"

    report += """
## Bottlenecks & Recommendations
1. **Undetected leakage**: Ensure all post-harvest measurement variables are flagged
2. **Target leakage**: Verify that target variables are not used as features
3. **Temporal leakage**: Add date-based leakage checks for time-series splits
4. Implement proper train/test separation with leakage awareness
5. Add cross-validation strategy that respects the leakage flag
"""
    path = write_report("08_Leakage_Report.md", report)
    return {
        "leakage_detected": total_leakage_detected,
        "leakage_prevented": leakage_prevented or 0,
        "false_positives": false_positives,
        "leakage_detection_rate": leakage_detected_rate,
    }, str(path)
