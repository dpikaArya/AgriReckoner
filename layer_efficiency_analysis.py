"""
Layer Efficiency Analysis, Model Benchmark & Updated Ready Reckoner
===================================================================
Generates comprehensive efficiency report for all pipeline layers,
model training results, and an updated ready reckoner table.
"""
import sys, json, time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATAADES_DIR = BASE_DIR / "DataADES"

print("=" * 80)
print("AAIF — LAYER EFFICIENCY, MODEL BENCHMARK & READY RECKONER UPDATE")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# =====================================================================
# 1. LOAD ALL PIPELINE OUTPUTS
# =====================================================================
print("\n[1/5] Loading pipeline outputs...")

schema_csv = OUTPUTS_DIR / "Universal_Agricultural_Schema.csv"
ingestion_csv = OUTPUTS_DIR / "ingestion_report.csv"
validation_xlsx = OUTPUTS_DIR / "validation_report.xlsx"
features_csv = OUTPUTS_DIR / "features_dataset.csv"
model_metrics_xlsx = OUTPUTS_DIR / "model_metrics.xlsx"
recommendations_csv = OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.csv"
ready_reckoner_xlsx = OUTPUTS_DIR / "Ready_Reckoner.xlsx"

schema_df = pd.read_csv(schema_csv) if schema_csv.exists() else pd.DataFrame()
ingestion_df = pd.read_csv(ingestion_csv) if ingestion_csv.exists() else pd.DataFrame()
features_df = pd.read_csv(features_csv) if features_csv.exists() else pd.DataFrame()
model_df = pd.read_excel(model_metrics_xlsx, engine="openpyxl") if model_metrics_xlsx.exists() else pd.DataFrame()
rec_df = pd.read_csv(recommendations_csv) if recommendations_csv.exists() else pd.DataFrame()
rr_df = pd.read_excel(ready_reckoner_xlsx, engine="openpyxl") if ready_reckoner_xlsx.exists() else pd.DataFrame()

# Count DataADES PDFs
dataades_pdfs = list(DATAADES_DIR.glob("*.pdf")) if DATAADES_DIR.exists() else []
n_dataades = len(dataades_pdfs)

# Count master datasets
master_dir = BASE_DIR / "data" / "master_datasets"
master_xlsx = list(master_dir.glob("*.xlsx")) if master_dir.exists() else []

# =====================================================================
# 2. LAYER EFFICIENCY ANALYSIS
# =====================================================================
print("[2/5] Computing layer-wise efficiency...")

from agri_ai_agent.config.schema import UAMS_COLUMNS, SCHEMA_GROUPS

total_uams = len(UAMS_COLUMNS)

# --- Layer 1: Ingestion ---
n_total_pdfs = len(ingestion_df) if not ingestion_df.empty else 0
n_new = int((ingestion_df["Status"] == "New").sum()) if not ingestion_df.empty and "Status" in ingestion_df.columns else 0
n_prev = int((ingestion_df["Status"] == "Previously Processed").sum()) if not ingestion_df.empty and "Status" in ingestion_df.columns else 0
n_dups = int((ingestion_df["Duplicate_Flag"] == "Yes").sum()) if not ingestion_df.empty and "Duplicate_Flag" in ingestion_df.columns else 0
ingestion_eff = round((n_new / max(1, n_total_pdfs)) * 100, 1) if n_total_pdfs > 0 else 0
ingestion_dup_detect = round((n_dups / max(1, n_total_pdfs)) * 100, 1) if n_total_pdfs > 0 else 0
ingestion_incremental = round((n_prev / max(1, n_total_pdfs)) * 100, 1) if n_total_pdfs > 0 else 0

# --- Layer 2-3: AI Extraction + Schema Mapping ---
n_schema_rows = len(schema_df) if not schema_df.empty else 0
n_schema_cols = len(schema_df.columns) if not schema_df.empty else 0
# Columns with data
cols_with_data = sum(1 for c in UAMS_COLUMNS if c in schema_df.columns and schema_df[c].notna().any()) if not schema_df.empty else 0
schema_coverage = round((cols_with_data / total_uams) * 100, 1)
# PDF extraction success rate
pdfs_with_yield = 0
pdfs_with_ph = 0
pdfs_with_crop = 0
if not schema_df.empty:
    if "Yield_per_Hectare" in schema_df.columns:
        pdfs_with_yield = int(schema_df["Yield_per_Hectare"].notna().sum())
    if "Soil_pH" in schema_df.columns:
        pdfs_with_ph = int(schema_df["Soil_pH"].notna().sum())
    if "Crop" in schema_df.columns:
        pdfs_with_crop = int(schema_df["Crop"].notna().sum())

extraction_yield_eff = round((pdfs_with_yield / max(1, n_schema_rows)) * 100, 1)
extraction_crop_eff = round((pdfs_with_crop / max(1, n_schema_rows)) * 100, 1)

# --- Layer 4: Validation ---
val_issues = 0
if validation_xlsx.exists():
    try:
        val_df = pd.read_excel(validation_xlsx, engine="openpyxl")
        val_issues = len(val_df[val_df["Missing_Pct"] > 0]) if "Missing_Pct" in val_df.columns else 0
    except Exception:
        pass
total_cols_in_schema = n_schema_cols
cols_clean = total_cols_in_schema - val_issues
validation_eff = round((cols_clean / max(1, total_cols_in_schema)) * 100, 1)

# --- Layer 5: Feature Engineering ---
n_features_added = 0
if not features_df.empty and not schema_df.empty:
    n_features_added = len(features_df.columns) - len(schema_df.columns)
feature_eng_eff = round((n_features_added / max(1, total_uams)) * 100, 1)

# --- Layer 6: Model Training ---
n_models_trained = len(model_df) if not model_df.empty else 0
n_targets = model_df["Target"].nunique() if not model_df.empty and "Target" in model_df.columns else 0
n_best_r2 = 0
if not model_df.empty and "R2" in model_df.columns:
    best_r2 = model_df["R2"].max()
    n_best_r2 = round(best_r2 * 100, 1)
training_success = n_models_trained
best_r2_str = f"{model_df['R2'].max():.4f}" if (not model_df.empty and "R2" in model_df.columns) else "N/A"

# --- Layer 7: Fuzzy Logic ---
fuzzy_rules_path = BASE_DIR / "fuzzy_logic" / "fertilizer_rules.yaml"
n_fuzzy_rules = 0
if fuzzy_rules_path.exists():
    import yaml
    with open(fuzzy_rules_path) as f:
        data = yaml.safe_load(f)
        n_fuzzy_rules = len(data.get("rules", []))
fuzzy_eff = 100.0 if n_fuzzy_rules > 0 else 0.0

# --- Layer 8: Recommendations ---
n_recs = len(rec_df) if not rec_df.empty else 0
unique_crops_rec = rec_df["Crop"].nunique() if not rec_df.empty and "Crop" in rec_df.columns else 0
rec_eff = round((unique_crops_rec / max(1, n_total_pdfs)) * 100, 1) if n_total_pdfs > 0 else 0

# --- Layer 9: Ready Reckoner ---
n_reckoner = len(rr_df) if not rr_df.empty else 0
reckoner_eff = round((n_reckoner / max(1, n_recs)) * 100, 1) if n_recs > 0 else 0

# --- Layer 10: Continuous Learning ---
db_path = BASE_DIR / "database" / "paper_registry.sqlite"
n_registered = 0
if db_path.exists():
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    n_registered = conn.execute("SELECT COUNT(*) FROM paper_registry").fetchone()[0]
    conn.close()
cl_eff = round((n_registered / max(1, n_registered + n_dataades)) * 100, 1)

# =====================================================================
# 3. PRINT LAYER EFFICIENCY TABLE
# =====================================================================
print("\n" + "=" * 80)
print("LAYER-WISE WORKING EFFICIENCY (%)")
print("=" * 80)

layers = [
    ("Phase 0", "Master Dataset Loading", f"{len(master_xlsx)} xlsx files loaded", "100.0%"),
    ("Phase 1", "Ingestion (PDF to Registry)",
     f"Total PDFs={n_total_pdfs}, New={n_new}, Dups={n_dups}, Prev={n_prev}",
     f"{ingestion_eff}%"),
    ("Phase 2-3", "AI Extraction + Schema Mapping",
     f"Rows={n_schema_rows}, Cols={n_schema_cols}, Schema Coverage={schema_coverage}%, Yield Extracted={extraction_yield_eff}%, Crop Detected={extraction_crop_eff}%",
     f"{schema_coverage}%"),
    ("Phase 4", "Data Validation",
     f"Total Cols={total_cols_in_schema}, Issues={val_issues}, Clean={cols_clean}",
     f"{validation_eff}%"),
    ("Phase 5", "Feature Engineering",
     f"Features Added={n_features_added}, Total Features={len(features_df.columns) if not features_df.empty else 0}",
     f"{feature_eng_eff}%"),
    ("Phase 6", "Model Training",
     f"Models={n_models_trained}, Targets={n_targets}, Best R2={best_r2_str}",
     f"See Model Table"),
    ("Phase 7", "Fuzzy Logic",
     f"Rules={n_fuzzy_rules}, Input Vars=10",
     f"{fuzzy_eff}%"),
    ("Phase 8", "Recommendation Agent",
     f"Recs={n_recs}, Unique Crops={unique_crops_rec}",
     f"{rec_eff}%"),
    ("Phase 9", "Ready Reckoner",
     f"Entries={n_reckoner}",
     f"{reckoner_eff}%"),
    ("Phase 10", "Continuous Learning",
     f"Registered={n_registered}",
     f"{cl_eff}%"),
]

for phase, name, detail, eff in layers:
    print(f"  {phase:12s} | {name:30s} | {detail:70s} | Eff: {eff}")

# =====================================================================
# 4. MODEL EFFICIENCY TABLE
# =====================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING EFFICIENCY (All Models)")
print("=" * 80)

if not model_df.empty:
    for _, row in model_df.iterrows():
        model_name = row["Model"]
        target = row["Target"]
        r2 = row["R2"]
        rmse = row["RMSE"]
        mae = row["MAE"]
        mape = row["MAPE"]
        cv_r2 = row["CV_Mean_R2"]
        cv_std = row["CV_Std_R2"]
        
        # Efficiency = R² converted to %, clamped to 0-100
        r2_pct = max(0, round(r2 * 100, 1)) if not np.isnan(r2) else 0
        cv_pct = max(0, round(cv_r2 * 100, 1)) if not np.isnan(cv_r2) else 0
        rmse_eff = max(0, round((1 - rmse / (rmse + 1)) * 100, 1))
        
        print(f"  {model_name:20s} | Target: {target:20s} | R²={r2:.4f} ({r2_pct}%) | RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}% | CV_R²={cv_r2:.4f} ({cv_pct}%)")

# Best model summary
if not model_df.empty and "CV_Mean_R2" in model_df.columns:
    best_idx = model_df["CV_Mean_R2"].idxmax()
    best = model_df.loc[best_idx]
    print(f"\n  BEST MODEL: {best['Model']} on {best['Target']}")
    print(f"    R² = {best['R2']:.4f} | CV R² = {best['CV_Mean_R2']:.4f} ± {best['CV_Std_R2']:.4f}")
    print(f"    RMSE = {best['RMSE']:.4f} | MAE = {best['MAE']:.4f} | MAPE = {best['MAPE']:.2f}%")

# =====================================================================
# 5. GENERATE UPDATED READY RECKONER TABLE
# =====================================================================
print("\n" + "=" * 80)
print("UPDATED READY RECKONER TABLE")
print("=" * 80)

# Merge recommendations with schema data for richer reckoner
reckoner_data = []
if not rec_df.empty:
    for _, rec in rec_df.iterrows():
        crop = rec.get("Crop", "")
        paper_id = rec.get("Paper_ID", "")
        
        # Find matching schema row
        soil_cond = "N/A"
        climate = ""
        yield_ha = None
        expected_yield = rec.get("Expected_Yield_kg_ha")
        
        if not schema_df.empty and "Paper_ID" in schema_df.columns:
            match = schema_df[schema_df["Paper_ID"] == paper_id]
            if len(match) > 0:
                m = match.iloc[0]
                ph = m.get("Soil_pH")
                n_val = m.get("Nitrogen")
                p_val = m.get("Phosphorus")
                k_val = m.get("Potassium")
                oc = m.get("Organic_Carbon")
                tmax = m.get("Temperature_Max")
                tmin = m.get("Temperature_Min")
                rain = m.get("Rainfall")
                yield_ha = m.get("Yield_per_Hectare")
                
                parts = []
                if pd.notna(ph): parts.append(f"pH {ph}")
                if pd.notna(n_val): parts.append(f"N {n_val}")
                if pd.notna(p_val): parts.append(f"P {p_val}")
                if pd.notna(k_val): parts.append(f"K {k_val}")
                if pd.notna(oc): parts.append(f"OC {oc}%")
                soil_cond = ", ".join(parts) if parts else "N/A"
                
                clim_parts = []
                if pd.notna(tmax) and pd.notna(tmin):
                    clim_parts.append(f"T {tmin}-{tmax}°C")
                if pd.notna(rain):
                    clim_parts.append(f"Rain {rain}mm")
                climate = ", ".join(clim_parts)
        
        # Application method
        fert_name = rec.get("Best_Treatment", "")
        method_map = {
            "Urea": "Soil application (band placement)", "DAP": "Soil application at sowing",
            "MOP": "Soil application", "NPK": "Broadcast + incorporation",
            "Compost": "Broadcast + incorporation", "Zinc": "Foliar spray",
            "SSP": "Soil application at sowing", "Ammonium": "Soil application",
            "Dolomite": "Soil application (broadcast)",
        }
        app_method = "Soil application"
        for kw, method in method_map.items():
            if kw.lower() in str(fert_name).lower():
                app_method = method
                break
        
        # Model efficiency for this crop (if available)
        model_eff_str = "N/A"
        if not model_df.empty and "Target" in model_df.columns:
            best_model_row = model_df.loc[model_df["CV_Mean_R2"].idxmax()]
            model_eff_str = f"{best_model_row['Model']} (R²={best_model_row['R2']:.4f})"
        
        reckoner_data.append({
            "Crop": crop,
            "Recommended_Treatment": fert_name,
            "Soil_Condition": soil_cond,
            "Climate_Condition": climate,
            "Yield_per_Hectare": yield_ha if pd.notna(yield_ha) else "N/A",
            "Expected_Yield_kg_ha": expected_yield if pd.notna(expected_yield) else "N/A",
            "Application_Method": app_method,
            "Application_Interval": rec.get("Application_Interval", ""),
            "Confidence_Score": rec.get("Confidence_Score", ""),
            "Recommendation": rec.get("Recommendation", ""),
            "Model_Efficiency": model_eff_str,
        })

reckoner_final = pd.DataFrame(reckoner_data)

# Save updated ready reckoner
reckoner_csv_path = OUTPUTS_DIR / "Ready_Reckoner_Table_Updated.csv"
reckoner_final.to_csv(reckoner_csv_path, index=False)
print(f"Updated Ready Reckoner CSV saved: {reckoner_csv_path}")

reckoner_xlsx_path = OUTPUTS_DIR / "Ready_Reckoner_Updated.xlsx"
reckoner_final.to_excel(reckoner_xlsx_path, index=False, engine="openpyxl")
print(f"Updated Ready Reckoner XLSX saved: {reckoner_xlsx_path}")

# Print the ready reckoner
print("\n" + reckoner_final.to_string(index=False))

# =====================================================================
# 6. SCHEMA UPDATE SUMMARY (DataADES PDFs integrated)
# =====================================================================
print("\n" + "=" * 80)
print("SCHEMA UPDATE SUMMARY (DataADES PDFs to Universal Schema)")
print("=" * 80)
print(f"  DataADES PDFs processed:       {n_dataades}")
print(f"  Master dataset files:          {len(master_xlsx)}")
print(f"  Total schema rows:             {n_schema_rows}")
print(f"  Total schema columns:          {n_schema_cols}")
print(f"  UAMS columns mapped:           {cols_with_data}/{total_uams} ({schema_coverage}%)")
print(f"  Schema groups covered:         {len([g for g, cols in SCHEMA_GROUPS.items() if any(c in schema_df.columns for c in cols)])}/{len(SCHEMA_GROUPS)}")
print(f"  Papers registered in DB:       {n_registered}")
print(f"  Features engineered:           {n_features_added}")

# =====================================================================
# 7. SAVE COMPREHENSIVE REPORT
# =====================================================================
print("\n[5/5] Saving comprehensive report...")

report_lines = [
    "=" * 80,
    "AAIF COMPREHENSIVE EFFICIENCY & MODEL BENCHMARK REPORT",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 80,
    "",
    "SECTION 1: LAYER-WISE EFFICIENCY",
    "-" * 40,
]
for phase, name, detail, eff in layers:
    report_lines.append(f"  {phase:12s} | {name:30s} | Efficiency: {eff}")

report_lines.extend([
    "",
    "SECTION 2: MODEL BENCHMARK",
    "-" * 40,
])

if not model_df.empty:
    for _, row in model_df.iterrows():
        r2_pct = max(0, round(row["R2"] * 100, 1)) if not np.isnan(row["R2"]) else 0
        cv_pct = max(0, round(row["CV_Mean_R2"] * 100, 1)) if not np.isnan(row["CV_Mean_R2"]) else 0
        report_lines.append(
            f"  {row['Model']:20s} | {row['Target']:20s} | R2={row['R2']:.4f} ({r2_pct}%) | CV_R2={row['CV_Mean_R2']:.4f} ({cv_pct}%) | RMSE={row['RMSE']:.4f} | MAE={row['MAE']:.4f} | MAPE={row['MAPE']:.2f}%"
        )

report_lines.extend([
    "",
    "SECTION 3: SCHEMA UPDATE (DataADES Integration)",
    "-" * 40,
    f"  DataADES PDFs:            {n_dataades}",
    f"  Schema rows:              {n_schema_rows}",
    f"  Schema columns:           {n_schema_cols}",
    f"  UAMS coverage:            {schema_coverage}%",
    f"  Features engineered:      {n_features_added}",
    f"  Fuzzy rules:              {n_fuzzy_rules}",
    f"  Recommendations:          {n_recs}",
    f"  Registered papers:        {n_registered}",
    "",
    "SECTION 4: READY RECKONER TABLE",
    "-" * 40,
])

if not reckoner_final.empty:
    report_lines.append(reckoner_final.to_string(index=False))

report_path = OUTPUTS_DIR / "AAIF_Layer_Efficiency_Report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"  Comprehensive report saved: {report_path}")

print("\n" + "=" * 80)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"  1. outputs/Ready_Reckoner_Table_Updated.csv")
print(f"  2. outputs/Ready_Reckoner_Updated.xlsx")
print(f"  3. outputs/AAIF_Layer_Efficiency_Report.txt")
print(f"  4. outputs/AAIF_Final_Report.html (pipeline report)")
print(f"  5. outputs/model_metrics.xlsx (model details)")
print("=" * 80)
