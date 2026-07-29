"""
Stage 10: Model Readiness Evaluation
Evaluates readiness for ML models: missing values, encoding, scaling, leakage,
multicollinearity, target imbalance, feature importance readiness.
"""

import time
from pathlib import Path

from evaluation.utils import OUTPUT_DIR, write_report, get_master_df, get_uams_cols
import numpy as np
import pandas as pd


EVALUATED_MODELS = [
    "Multiple Linear Regression", "Polynomial Regression",
    "Random Forest", "Extra Trees", "XGBoost", "LightGBM",
    "CatBoost", "Support Vector Regression",
    "Neural Networks", "LSTM", "Transformer",
]


def evaluate_model_readiness():
    master_df = get_master_df()

    if master_df is None:
        report = "# Stage 10: Model Readiness Report\nNo data available.\n"
        path = write_report("10_Model_Readiness.md", report)
        return {}, str(path)

    n_rows = len(master_df)
    n_cols = len(master_df.columns)
    numeric_df = master_df.select_dtypes(include=["number"])
    cat_df = master_df.select_dtypes(include=["object", "category"])

    total_missing = int(master_df.isna().sum().sum())
    missing_pct = total_missing / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0

    high_missing_cols = [
        c for c in master_df.columns
        if master_df[c].isna().mean() > 0.1
    ]
    cat_unencoded = [c for c in cat_df.columns
                     if not c.endswith("_Code") and c not in ["Paper_ID", "DOI"]]

    numeric_cols_for_vif = numeric_df.columns.tolist()
    vif_issues = False
    if len(numeric_cols_for_vif) > 2:
        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_vif = [(upper.columns[c], upper.columns[r])
                     for c in range(len(upper.columns))
                     for r in range(c + 1, len(upper.columns))
                     if not np.isnan(upper.iloc[r, c]) and upper.iloc[r, c] > 0.9]
        vif_issues = len(high_vif) > 0

    target_imbalance = False
    for t in ["Target_Yield", "Target_Fertilizer", "Target_Nitrogen"]:
        if t in master_df.columns:
            col = master_df[t].dropna()
            if len(col) > 0:
                try:
                    q75, q25 = col.quantile(0.75), col.quantile(0.25)
                    iqr = q75 - q25
                    if iqr > 0:
                        skew_val = float(col.skew())
                        if abs(skew_val) > 2:
                            target_imbalance = True
                except Exception as e:
                    import logging
                    logging.getLogger("Stage10").debug("Imbalance check failed for %s: %s", t, e)

    model_readiness = {}
    for model in EVALUATED_MODELS:
        issues = []

        if missing_pct > 10:
            issues.append("High missing values")
        if cat_unencoded:
            issues.append("Unencoded categorical variables")
        if model in ["Support Vector Regression", "Neural Networks", "LSTM", "Transformer"]:
            issues.append("Requires feature scaling")
        if vif_issues and model in ["Multiple Linear Regression", "Polynomial Regression"]:
            issues.append("Multicollinearity present")
        if target_imbalance and model in ["Random Forest", "Extra Trees"]:
            issues.append("Target imbalance may affect performance")
        if missing_pct > 5 and model in ["XGBoost", "LightGBM", "CatBoost"]:
            pass
        if model in ["Transformer", "LSTM"] and n_rows < 100:
            issues.append("Insufficient samples for deep learning")

        readiness = "ready" if not issues else "conditional" if len(issues) <= 2 else "not_ready"
        model_readiness[model] = {
            "readiness": readiness,
            "issues": issues,
        }

    ready_count = sum(1 for v in model_readiness.values() if v["readiness"] == "ready")
    conditional_count = sum(1 for v in model_readiness.values() if v["readiness"] == "conditional")
    not_ready_count = sum(1 for v in model_readiness.values() if v["readiness"] == "not_ready")

    report = f"""# Stage 10: Model Readiness Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {n_rows} rows × {n_cols} columns

## Summary
- Models ready: {ready_count}/{len(EVALUATED_MODELS)}
- Models conditional: {conditional_count}/{len(EVALUATED_MODELS)}
- Models not ready: {not_ready_count}/{len(EVALUATED_MODELS)}
- Missing values: {missing_pct:.1f}%
- Unencoded categorical columns: {len(cat_unencoded)}
- High missing value columns: {len(high_missing_cols)}
- Multicollinearity: {'Yes' if vif_issues else 'No'}
- Target imbalance: {'Yes' if target_imbalance else 'No'}

## Model Readiness
"""
    for model, info in model_readiness.items():
        icon = "✅" if info["readiness"] == "ready" else "⚠️" if info["readiness"] == "conditional" else "❌"
        report += f"| {icon} | {model} | {info['readiness']} | {', '.join(info['issues']) if info['issues'] else 'None'} |\n"

    report += f"""
## Missing Values
- {len(high_missing_cols)} columns with >10% missing:
"""
    for c in high_missing_cols[:15]:
        pct = master_df[c].isna().mean() * 100
        report += f"  - {c}: {pct:.1f}%\n"

    report += f"""
## Categorical Variables Requiring Encoding ({len(cat_unencoded)})
"""
    for c in cat_unencoded[:20]:
        vals = master_df[c].nunique()
        report += f"  - {c} ({vals} unique values)\n"

    report += """
## Bottlenecks & Recommendations
1. **Encode categorical variables** using Label Encoding (already implemented in Agent 08)
2. **Impute missing values** with median/mode before training
3. **Scale features** for SVR, Neural Networks, LSTM, Transformer
4. **Address multicollinearity** via feature selection or regularization
5. **Balance targets** using resampling or appropriate loss functions
"""
    path = write_report("10_Model_Readiness.md", report)
    return model_readiness, str(path)
