"""
Post-Pipeline Comprehensive Analysis
- Model Performance Assessment (all ML models with full metrics)
- Multi-Regression Results table
- Updated Ready Reckoner for all 24 PDFs
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# =========================================================================
# STEP 1: Re-extract all table-level data from all PDFs
# =========================================================================
def extract_all_table_data():
    """Extract table-level data from all PDFs using pdfplumber + enhanced."""
    from enhanced_extraction import process_one_pdf
    from pdfplumber_extraction import process_pdf

    papers_dir = BASE_DIR / "Data ADES"
    all_rows = []

    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        pdf_name = pdf_path.name
        rows = []

        # Try pdfplumber first
        try:
            plumber_rows, meta = process_pdf(str(pdf_path))
            if plumber_rows:
                for r in plumber_rows:
                    r["Source_File"] = pdf_name
                rows = plumber_rows
        except Exception as e:
            log(f"  {pdf_name}: pdfplumber extraction failed: {e}")

        # Try enhanced extraction as fallback
        if not rows:
            try:
                enh_rows = process_one_pdf(str(pdf_path))
                if enh_rows:
                    for r in enh_rows:
                        r["Source_File"] = pdf_name
                    rows = enh_rows
            except Exception as e:
                log(f"  {pdf_name}: enhanced extraction failed: {e}")

        if rows:
            all_rows.extend(rows)
            log(f"  {pdf_name}: {len(rows)} rows")
        else:
            log(f"  {pdf_name}: 0 rows")

    log(f"Total table-level rows: {len(all_rows)}")
    return all_rows


# =========================================================================
# STEP 2: Build comprehensive training dataset
# =========================================================================
def build_training_dataset(table_rows):
    """Convert raw table rows into a clean ML-ready dataset."""
    if not table_rows:
        return pd.DataFrame()

    df = pd.DataFrame(table_rows)

    # Ensure numeric columns are numeric
    num_cols = [
        "Yield_per_Hectare",
        "Yield_per_Plot",
        "Yield_per_Acre",
        "Biomass_Yield",
        "Harvest_Index",
        "Plant_Height_cm",
        "SPAD",
        "Tillers",
        "Spike_Length",
        "Seeds_per_Spike",
        "100_Seed_Weight",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Organic_Carbon",
        "Soil_pH",
        "Protein",
        "Iron_ppm",
        "Zinc_ppm",
        "Copper",
        "Manganese_ppm",
        "Sodium",
        "Magnesium",
        "Calcium",
        "Sulfur",
        "Boron",
        "Fruit_Weight",
        "Fruit_Diameter_mm",
        "Fruit_Length",
        "Leaf_Area_cm2",
        "Rainfall",
        "Temperature_C",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================================
# STEP 3: Train all models with full evaluation
# =========================================================================
def train_all_models(df):
    """Train 10+ ML models with comprehensive evaluation metrics."""
    import xgboost as xgb
    from sklearn.ensemble import (
        AdaBoostRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    # Identify target and features
    target = "Yield_per_Hectare"

    # The shared guard, not a hand-written list: it excludes every post-harvest outcome
    # (protein, seed weight, spike length...) which is measured at harvest alongside
    # yield and so is unavailable when a recommendation is actually made.
    from agri_ai_agent.ml.leakage import select_feature_columns

    exclude = {"Source_File", "Treatment", "Paper_ID", "Design"}
    safe_cols = set(select_feature_columns(df, target, base_exclude=exclude))

    feature_cols = []
    for col in df.columns:
        if col not in safe_cols:
            continue
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            if df[col].notna().sum() >= 3:  # at least 3 non-null values
                feature_cols.append(col)

    log(f"  Target: {target}")
    log(f"  Features: {len(feature_cols)} — {feature_cols}")

    # Prepare X, y — only rows where target is non-null
    mask = df[target].notna()
    X_full = df.loc[mask, feature_cols].copy()
    y_full = df.loc[mask, target].copy()

    # Fill remaining NaN with column median
    for col in X_full.columns:
        med = X_full[col].median()
        if pd.isna(med):
            X_full[col] = 0  # all-NaN column -> fill with 0
        else:
            X_full[col] = X_full[col].fillna(med)

    # Drop columns that are still all zero/NaN or have zero variance
    X_full = X_full.loc[:, X_full.notna().any()]
    # Also drop columns with zero or near-zero variance
    for col in list(X_full.columns):
        if X_full[col].std() < 1e-10:
            X_full.drop(columns=[col], inplace=True, errors="ignore")
    feature_cols = list(X_full.columns)

    log(f"  Training samples: {len(X_full)}")

    # Define all models
    models = {
        "Multiple Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso Regression": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1))]),
        "Elastic Net": Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.1, l1_ratio=0.5))]
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0))]),
        "KNN Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]
        ),
    }

    # Cross-validation strategy
    n_samples = len(X_full)
    if n_samples >= 10:
        cv = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)
    elif n_samples >= 3:
        cv = KFold(n_splits=n_samples, shuffle=True, random_state=42)
    else:
        cv = None

    results = []

    for name, model in models.items():
        try:
            # Scored on the rows it was fitted on: this measures how well the model
            # memorises the training set, never how it would do on an unseen trial.
            # Reported under *_InSample names so it cannot be quoted as skill.
            model.fit(X_full, y_full)
            y_pred_full = model.predict(X_full)
            mae = mean_absolute_error(y_full, y_pred_full)
            mse = mean_squared_error(y_full, y_pred_full)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_full, y_pred_full)
            medae = median_absolute_error(y_full, y_pred_full)
            n = len(y_full)
            p = len(feature_cols)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan

            try:
                mape = mean_absolute_percentage_error(y_full, y_pred_full) * 100
            except Exception as e:
                log(f"  MAPE calculation failed: {e}")
                mape = np.nan

            # Cross-validation metrics
            cv_r2_mean = np.nan
            cv_r2_std = np.nan
            cv_mae_mean = np.nan
            cv_mae_std = np.nan
            if cv is not None and n_samples >= 3:
                try:
                    cv_r2_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2")
                    cv_mae_scores = cross_val_score(
                        model, X_full, y_full, cv=cv, scoring="neg_mean_absolute_error"
                    )
                    cv_r2_mean = cv_r2_scores.mean()
                    cv_r2_std = cv_r2_scores.std()
                    cv_mae_mean = -cv_mae_scores.mean()
                    cv_mae_std = cv_mae_scores.std()
                except Exception as e:
                    log(f"  {name}: cross-validation failed: {e}")

            # AIC and BIC (for linear models only)
            aic = np.nan
            bic = np.nan
            if hasattr(model, "named_steps"):
                m = model.named_steps.get("model", None)
            else:
                m = model
            if hasattr(m, "coef_") or hasattr(m, "intercept_"):
                try:
                    rss = np.sum((y_full - y_pred_full) ** 2)
                    k = p + 1
                    aic = n * np.log(rss / n) + 2 * k
                    bic = n * np.log(rss / n) + k * np.log(n)
                except Exception as e:
                    log(f"  {name}: AIC/BIC computation failed: {e}")

            # Feature importance
            fi = {}
            if hasattr(model, "feature_importances_"):
                for feat, imp in zip(feature_cols, model.feature_importances_, strict=True):
                    fi[feat] = round(imp, 4)
            elif hasattr(model, "named_steps"):
                m = model.named_steps.get("model", None)
                if hasattr(m, "coef_"):
                    for feat, coef in zip(feature_cols, m.coef_, strict=True):
                        fi[feat] = round(coef, 4)

            results.append(
                {
                    "Model": name,
                    "N_Samples": n,
                    "N_Features": p,
                    "MAE_InSample": round(mae, 4),
                    "MSE_InSample": round(mse, 4),
                    "RMSE_InSample": round(rmse, 4),
                    "R2_InSample": round(r2, 4),
                    "Adj_R2_InSample": round(adj_r2, 4) if not np.isnan(adj_r2) else "",
                    "MAPE (%)": round(mape, 2) if not np.isnan(mape) else "",
                    "Median_AE_InSample": round(medae, 4),
                    "CV_R2_Mean": round(cv_r2_mean, 4) if not np.isnan(cv_r2_mean) else "",
                    "CV_R2_Std": round(cv_r2_std, 4) if not np.isnan(cv_r2_std) else "",
                    "CV_MAE_Mean": round(cv_mae_mean, 4) if not np.isnan(cv_mae_mean) else "",
                    "CV_MAE_Std": round(cv_mae_std, 4) if not np.isnan(cv_mae_std) else "",
                    "AIC": round(aic, 2) if not np.isnan(aic) else "",
                    "BIC": round(bic, 2) if not np.isnan(bic) else "",
                }
            )

            log(f"  {name:30s}  R2(in-sample)={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

        except Exception as e:
            log(f"  {name:30s}  FAILED: {e}")
            results.append(
                {
                    "Model": name,
                    "N_Samples": len(X_full),
                    "N_Features": len(feature_cols),
                    "MAE_InSample": "",
                    "MSE_InSample": "",
                    "RMSE_InSample": "",
                    "R2_InSample": "",
                    "Adj_R2_InSample": "",
                    "MAPE (%)": "",
                    "Median_AE_InSample": "",
                    "CV_R2_Mean": "",
                    "CV_R2_Std": "",
                    "CV_MAE_Mean": "",
                    "CV_MAE_Std": "",
                    "AIC": "",
                    "BIC": "",
                }
            )

    return pd.DataFrame(results), models, feature_cols, X_full, y_full


# =========================================================================
# STEP 4: Multi-Regression Analysis
# =========================================================================
def multi_regression_analysis(X, y, feature_cols=None):
    """Detailed multiple regression with coefficient analysis."""
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    # Use actual columns from X
    if feature_cols is None:
        feature_cols = list(X.columns)
    else:
        feature_cols = list(X.columns)  # always use X's actual columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols).reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    # Statsmodels OLS for detailed statistics
    X_const = sm.add_constant(X_scaled_df)
    try:
        model = sm.OLS(y_reset, X_const).fit()
    except Exception as e:
        log(f"  Statsmodels OLS failed: {e}")
        # Try with reset index
        try:
            X_const2 = sm.add_constant(X_scaled_df)
            model = sm.OLS(y_reset.values, X_const2).fit()
        except Exception as e2:
            log(f"  Statsmodels OLS retry also failed: {e2}")
            return pd.DataFrame(), {}

    # Build results table
    rows = []
    var_names = ["const"] + feature_cols
    for i, feat in enumerate(var_names):
        coef = model.params.iloc[i]
        se = model.bse.iloc[i]
        t_val = model.tvalues.iloc[i]
        p_val = model.pvalues.iloc[i]
        ci_low = model.conf_int().iloc[i, 0]
        ci_high = model.conf_int().iloc[i, 1]
        sig = ""
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        elif p_val < 0.1:
            sig = "."

        rows.append(
            {
                "Variable": feat,
                "Coefficient": round(coef, 6),
                "Std_Error": round(se, 6),
                "t_value": round(t_val, 4),
                "p_value": round(p_val, 6),
                "Significance": sig,
                "CI_Lower": round(ci_low, 6),
                "CI_Upper": round(ci_high, 6),
                "Std_Coefficient": round(coef, 6),  # already standardized
            }
        )

    reg_df = pd.DataFrame(rows)

    # Model summary stats
    summary = {
        "R_squared": round(model.rsquared, 4),
        "Adj_R_squared": round(model.rsquared_adj, 4),
        "F_statistic": round(model.fvalue, 4),
        "F_p_value": round(model.f_pvalue, 6),
        "Log_Likelihood": round(model.llf, 2),
        "AIC": round(model.aic, 2),
        "BIC": round(model.bic, 2),
        "Durbin_Watson": round(sm.stats.stattools.durbin_watson(model.resid), 4),
        "N_Observations": int(model.nobs),
        "N_Features": len(feature_cols),
        "Residual_Std_Error": round(np.sqrt(model.mse_resid), 4),
        "Cond_Number": round(model.condition_number, 2),
    }

    log(f"  OLS R2: {summary['R_squared']}, Adj R2: {summary['Adj_R_squared']}")
    log(f"  F-stat: {summary['F_statistic']}, p={summary['F_p_value']}")

    return reg_df, summary


# =========================================================================
# STEP 5: Ready Reckoner from all 24 PDFs
# =========================================================================
def build_ready_reckoner(table_rows):
    """Build Ready Reckoner table from extracted table-level data."""
    if not table_rows:
        return pd.DataFrame()

    df = pd.DataFrame(table_rows)

    # Convert numeric columns
    for col in df.columns:
        if col not in ("Treatment", "Source_File", "Reader"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate per source file (paper)
    agg_funcs = {}
    for col in df.columns:
        if col in ("Treatment", "Source_File", "Reader"):
            continue
        agg_funcs[col] = ["mean", "min", "max", "std", "count"]

    # Simple aggregation: mean per source file
    agg_df = (
        df.groupby("Source_File")
        .agg(
            {col: "mean" for col in df.columns if col not in ("Treatment", "Source_File", "Reader")}
        )
        .reset_index()
    )

    # Add treatment count
    treat_counts = df.groupby("Source_File")["Treatment"].nunique().reset_index()
    treat_counts.columns = ["Source_File", "N_Treatments"]
    agg_df = agg_df.merge(treat_counts, on="Source_File", how="left")

    return agg_df


# =========================================================================
# MAIN
# =========================================================================
def main():
    log("=" * 70)
    log("  POST-PIPELINE COMPREHENSIVE ANALYSIS")
    log("=" * 70)

    # Step 1: Extract all table data
    log("\n--- Step 1: Extracting table data from all PDFs ---")
    table_rows = extract_all_table_data()

    # Step 2: Build training dataset
    log("\n--- Step 2: Building training dataset ---")
    df = build_training_dataset(table_rows)
    log(f"  Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    log(f"  Yield_per_Hectare non-null: {df['Yield_per_Hectare'].notna().sum()}")

    # Step 3: Train all models
    log("\n--- Step 3: Training all ML models ---")
    model_results, trained_models, feature_cols, X, y = train_all_models(df)

    # feature_cols may have been trimmed by train_all_models
    feature_cols = list(X.columns)

    # Save model performance table
    perf_path = TABLE_DIR / "Model_Performance_Assessment.xlsx"
    model_results.to_excel(perf_path, index=False, engine="openpyxl")
    log(f"\n  Model Performance table saved: {perf_path}")

    # Also save as CSV
    model_results.to_csv(
        TABLE_DIR / "Model_Performance_Assessment.csv", index=False, encoding="utf-8-sig"
    )
    log(f"  Model Performance CSV saved: {TABLE_DIR / 'Model_Performance_Assessment.csv'}")

    # Step 4: Multi-Regression
    log("\n--- Step 4: Multiple Regression Analysis ---")
    reg_results, reg_summary = multi_regression_analysis(X, y, feature_cols)

    if not reg_results.empty:
        # Save regression coefficients
        reg_path = TABLE_DIR / "Multi_Regression_Results.xlsx"
        with pd.ExcelWriter(reg_path, engine="openpyxl") as writer:
            reg_results.to_excel(writer, sheet_name="Coefficients", index=False)
            summary_df = pd.DataFrame([reg_summary]).T.reset_index()
            summary_df.columns = ["Metric", "Value"]
            summary_df.to_excel(writer, sheet_name="Model_Summary", index=False)
        log(f"  Multi-Regression table saved: {reg_path}")

        reg_results.to_csv(
            TABLE_DIR / "Multi_Regression_Results.csv", index=False, encoding="utf-8-sig"
        )
        log(f"  Multi-Regression CSV saved: {TABLE_DIR / 'Multi_Regression_Results.csv'}")
    else:
        log("  Multi-Regression analysis returned empty results")

    # Step 5: Ready Reckoner
    log("\n--- Step 5: Building Ready Reckoner ---")
    rr_df = build_ready_reckoner(table_rows)
    if not rr_df.empty:
        rr_path = OUTPUTS_DIR / "Ready_Reckoner_24Papers.xlsx"
        rr_df.to_excel(rr_path, index=False, engine="openpyxl")
        log(f"  Ready Reckoner saved: {rr_path}")

    # Print summary
    log("\n" + "=" * 70)
    log("  ANALYSIS COMPLETE")
    log("=" * 70)
    log("\n  Tables produced:")
    log(f"    1. {TABLE_DIR / 'Model_Performance_Assessment.xlsx'}")
    log(f"    2. {TABLE_DIR / 'Multi_Regression_Results.xlsx'}")
    log(f"    3. {OUTPUTS_DIR / 'Ready_Reckoner_24Papers.xlsx'}")
    log(
        f"\n  Best model by R2: {model_results.loc[model_results['R2'].astype(str).ne('').idxmax(), 'Model'] if len(model_results) > 0 else 'N/A'}"
    )

    return model_results, reg_results, rr_df


if __name__ == "__main__":
    main()
