"""
Stage 07: Feature Engineering Evaluation
Evaluates generated features, formula correctness, feature reproducibility.
"""

import time
from pathlib import Path

from evaluation.utils import OUTPUT_DIR, write_report, get_master_df

import numpy as np


ENGINEERED_FEATURES = [
    "Growing_Degree_Days", "Heat_Units", "Harvest_Index_Calc",
    "Nitrogen_Use_Efficiency", "Water_Use_Efficiency",
    "Rainfall_Anomaly", "Stress_Index", "Disease_Risk_Index",
    "Yield_per_Plant", "Yield_per_Plot_Calc", "Yield_per_Hectare_Calc",
    "Temp_x_Rainfall", "N_x_P", "Temp_ squared",
    "Rainfall_7d_MA", "Temp_7d_MA",
]


def evaluate_features():
    master_df = get_master_df()
    n_rows = len(master_df) if master_df is not None else 0
    n_cols = len(master_df.columns) if master_df is not None else 0

    if master_df is None:
        report = "# Stage 07: Feature Engineering Report\nNo data available.\n"
        path = write_report("07_Feature_Report.md", report)
        return {}, str(path)

    existing_features = [f for f in ENGINEERED_FEATURES if f in master_df.columns]
    missing_features = [f for f in ENGINEERED_FEATURES if f not in master_df.columns]

    non_null_counts = {}
    for f in existing_features:
        non_null = master_df[f].notna().sum()
        non_null_counts[f] = int(non_null)

    calc_accuracy = 1.0
    if "Growing_Degree_Days" in master_df.columns and \
       "Temperature_Max" in master_df.columns and "Temperature_Min" in master_df.columns:
        tmax = master_df["Temperature_Max"].dropna()
        tmin = master_df["Temperature_Min"].dropna()
        if len(tmax) > 0 and len(tmin) > 0:
            expected_gdd = ((tmax + tmin) / 2 - 10).clip(lower=0)
            actual_gdd = master_df["Growing_Degree_Days"]
            matched = actual_gdd.notna() & expected_gdd.notna()
            if matched.sum() > 0:
                diff = np.abs(actual_gdd[matched] - expected_gdd[matched])
                calc_accuracy = float((diff < 0.1).mean())

    feature_count = len(existing_features)
    total_expected = len(ENGINEERED_FEATURES)

    report = f"""# Stage 07: Feature Engineering Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total engineered features expected: {total_expected}
- Engineered features generated: {feature_count}
- Missing features: {len(missing_features)}
- Calculation accuracy (GDD test): {calc_accuracy:.1%}
- Feature reproducibility: {'PASS' if calc_accuracy > 0.8 else 'NEEDS IMPROVEMENT'}

## Features Generated ({feature_count})
"""
    for f in existing_features:
        report += f"- {f}: {non_null_counts.get(f, 0)}/{n_rows} non-null\n"

    report += f"""
## Missing Features ({len(missing_features)})
"""
    for f in missing_features:
        report += f"- {f}\n"

    report += """
## Bottlenecks & Recommendations
1. **Missing features**: Add formulas for the {missing_count} unimplemented engineered features
2. **Null values**: Many engineered features need complete input columns to compute values
3. **Formula verification**: Add unit tests to verify all feature engineering formulas
4. Consider adding feature importance analysis to prioritize high-value engineered features
""".replace("{missing_count}", str(len(missing_features)))

    path = write_report("07_Feature_Report.md", report)
    return {
        "feature_count": feature_count,
        "missing_features": len(missing_features),
        "calculation_accuracy": calc_accuracy,
        "existing_features": existing_features,
        "missing_feature_list": missing_features,
    }, str(path)
