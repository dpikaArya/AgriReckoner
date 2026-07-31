"""
Stage 09: Statistical Diagnostics Evaluation
Computes summary statistics, missing values, correlation matrix, VIF, skewness, kurtosis, normality.
"""

import time

import numpy as np
import pandas as pd

from evaluation.utils import OUTPUT_DIR, get_master_df, load_dataframe, write_report


def evaluate_statistics():
    master_df = get_master_df()

    if master_df is None:
        report = "# Stage 09: Statistics Report\nNo data available.\n"
        path = write_report("09_Statistics_Report.md", report)
        return {}, str(path)

    n_rows = len(master_df)
    n_cols = len(master_df.columns)
    numeric_df = master_df.select_dtypes(include=["number"])
    numeric_cols = numeric_df.columns.tolist()

    numeric_df.describe().to_dict() if len(numeric_cols) > 0 else {}

    missing_counts = master_df.isna().sum().to_dict()
    total_cells = n_rows * n_cols
    total_missing = master_df.isna().sum().sum()
    missing_pct = total_missing / total_cells if total_cells > 0 else 0

    missing_by_col = {
        k: int(v) for k, v in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    }

    corr_matrix = None
    if len(numeric_cols) > 1:
        corr_matrix = numeric_df.corr(method="pearson")

    high_corr_pairs = []
    if corr_matrix is not None:
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = abs(corr_matrix.iloc[i, j])
                if val > 0.8 and not np.isnan(val):
                    high_corr_pairs.append(
                        {
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": round(val, 3),
                        }
                    )

    vif_data = None
    vif_path = OUTPUT_DIR / "VIF_Report.csv"
    if vif_path.exists():
        vif_df = load_dataframe(str(vif_path))
        if vif_df is not None:
            high_vif = vif_df[vif_df.iloc[:, 1] > 10] if len(vif_df.columns) > 1 else pd.DataFrame()
            vif_data = {
                "high_vif_count": len(high_vif),
                "max_vif": float(vif_df.iloc[:, 1].max()) if len(vif_df.columns) > 1 else 0,
            }

    skewness = {}
    kurtosis = {}
    normality = {}
    for col in numeric_cols[:30]:
        col_data = numeric_df[col].dropna()
        if len(col_data) > 3:
            skewness[col] = round(float(col_data.skew()), 3)
            kurtosis[col] = round(float(col_data.kurtosis()), 3)

            from scipy import stats

            _, p_value = stats.shapiro(col_data[:5000] if len(col_data) > 5000 else col_data)
            normality[col] = {
                "statistic": round(float(p_value), 4),
                "is_normal": bool(p_value > 0.05),
            }

    high_skew = {k: v for k, v in skewness.items() if abs(v) > 1}
    high_kurtosis = {k: v for k, v in kurtosis.items() if abs(v) > 3}
    non_normal = {k: v for k, v in normality.items() if not v["is_normal"]}

    report = f"""# Stage 09: Statistical Diagnostics Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Dataset shape: {n_rows} rows × {n_cols} columns
- Numeric columns: {len(numeric_cols)}
- Missing values: {total_missing}/{total_cells} ({missing_pct:.1%})
- High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}
- High VIF features: {vif_data["high_vif_count"] if vif_data else "N/A"}
- Highly skewed features: {len(high_skew)}
- High kurtosis features: {len(high_kurtosis)}
- Non-normal features: {len(non_normal)}

## Missing Values (Top 20)
| Column | Missing | % |
|--------|---------|---|
"""
    for col, count in missing_by_col.items():
        pct = count / n_rows * 100
        report += f"| {col} | {count} | {pct:.1f}% |\n"

    report += f"""
## High Correlation Pairs ({len(high_corr_pairs)})
"""
    for pair in high_corr_pairs[:20]:
        report += f"- {pair['var1']} ↔ {pair['var2']}: r={pair['correlation']:.3f}\n"

    report += f"""
## Skewness (|skew| > 1: {len(high_skew)})
"""
    for col, val in list(high_skew.items())[:15]:
        report += f"- {col}: {val:.3f}\n"

    report += f"""
## Kurtosis (|kurt| > 3: {len(high_kurtosis)})
"""
    for col, val in list(high_kurtosis.items())[:15]:
        report += f"- {col}: {val:.3f}\n"

    report += f"""
## Non-Normal Features (Shapiro-Wilk p<0.05: {len(non_normal)})
"""
    for col in list(non_normal.keys())[:15]:
        report += f"- {col}\n"

    report += """
## Bottlenecks & Recommendations
1. **Missing values**: Impute or drop columns with >10% missing before modeling
2. **High correlation**: Remove or combine highly correlated features
3. **High VIF**: Apply dimensionality reduction or regularization
4. **Skewness/Kurtosis**: Apply power transforms (Box-Cox, Yeo-Johnson)
5. **Normality**: Consider non-parametric tests or robust methods
"""
    path = write_report("09_Statistics_Report.md", report)
    return {
        "missing_pct": missing_pct,
        "high_corr_pairs": len(high_corr_pairs),
        "high_skew_features": len(high_skew),
        "non_normal_features": len(non_normal),
        "vif_high": vif_data["high_vif_count"] if vif_data else 0,
    }, str(path)
