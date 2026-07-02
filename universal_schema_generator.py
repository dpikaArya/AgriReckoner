"""
Universal Agricultural Schema Generator (Merge Agent)
======================================================
Reads multiple crop master datasets from /data/master_datasets/,
standardizes column names, creates a universal schema of ~150
standardized columns, and outputs ready-to-use ML files.

Part of an Agentic AI pipeline for Crop Recommendation and
Yield Prediction.
"""

import pandas as pd
import numpy as np
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "master_datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_COLUMNS = [
    "Paper_ID", "DOI", "Journal", "Year", "Authors", "Country",
    "Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days",
    "Design", "Replications", "Plot_Size", "Spacing_Row", "Spacing_Plant",
    "Sample_Size", "Location", "State", "Site",
    "Latitude", "Longitude", "Altitude",
    "Temperature_Max", "Temperature_Min", "Average_Temperature",
    "Rainfall", "Humidity",
    "Soil_pH", "EC",
    "Organic_Carbon", "Organic_Matter",
    "Nitrogen", "Phosphorus", "Potassium",
    "Sulphur", "Iron", "Copper", "Manganese", "Zinc",
    "Calcium", "Magnesium", "Boron", "Molybdenum",
    "Treatment", "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer",
    "Dose", "Application_Method", "Application_Interval",
    "Shoot_Length_cm", "Root_Length_cm", "Plant_Height_cm",
    "Shoot_Biomass_g", "Root_Biomass_g",
    "Leaf_Area_cm2", "Leaf_Number", "Tillers",
    "Root_Diameter_mm", "SPAD", "Moisture_Content", "Dry_Matter",
    "Stem_Diameter_mm", "Branches", "Nodes", "Flowers",
    "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm",
    "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2",
    "Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare",
    "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm",
    "Spike_Length", "Seeds_per_Spike",
    "100_Seed_Weight", "Root_Weight", "Pod_Weight",
    "Harvest_Index", "Biomass_Yield",
    "Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat",
    "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content",
    "Iron_Content", "Copper_Content", "Zinc_Content",
    "Manganese_Content", "Sulphur_Content",
    "Target_Yield", "Target_Fertilizer",
    "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
]

_EXACT_MAP = {
    "paper_id": "Paper_ID", "doi": "DOI", "journal": "Journal",
    "year": "Year", "authors": "Authors", "country": "Country",
    "crop": "Crop", "scientific_name": "Scientific_Name",
    "scientificname": "Scientific_Name",
    "variety": "Variety", "cultivar": "Variety",
    "season": "Season", "growing_season": "Season",
    "harvest_days": "Growth_Duration_Days",
    "harvest_das": "Growth_Duration_Days",
    "harvest_stage": "Growth_Duration_Days",
    "growth_duration": "Growth_Duration_Days",
    "growth_duration_days": "Growth_Duration_Days",
    "days_to_maturity": "Growth_Duration_Days",
    "design": "Design", "experimental_design": "Design",
    "exp_design": "Design",
    "replications": "Replications", "replicates": "Replications",
    "replicate": "Replications",
    "plot_size": "Plot_Size", "plot_size_m2": "Plot_Size",
    "plot_area": "Plot_Size",
    "row_spacing": "Spacing_Row", "row_spacing_cm": "Spacing_Row",
    "spacing_row": "Spacing_Row",
    "plant_spacing": "Spacing_Plant", "plant_spacing_cm": "Spacing_Plant",
    "spacing_plant": "Spacing_Plant",
    "sample_size": "Sample_Size",
    "location": "Location", "state": "State", "site": "Site",
    "latitude": "Latitude", "lat": "Latitude",
    "longitude": "Longitude", "lon": "Longitude", "long": "Longitude",
    "altitude": "Altitude", "elevation": "Altitude",
    "tmax": "Temperature_Max", "tmax_c": "Temperature_Max",
    "temp_max": "Temperature_Max", "temperature_max": "Temperature_Max",
    "max_temperature": "Temperature_Max",
    "tmin": "Temperature_Min", "tmin_c": "Temperature_Min",
    "temp_min": "Temperature_Min", "temperature_min": "Temperature_Min",
    "min_temperature": "Temperature_Min",
    "temp": "Average_Temperature", "temperature": "Average_Temperature",
    "avg_temp": "Average_Temperature", "average_temperature": "Average_Temperature",
    "rainfall": "Rainfall", "rainfall_mm": "Rainfall",
    "precipitation": "Rainfall",
    "humidity": "Humidity", "relative_humidity": "Humidity",
    "ph": "Soil_pH", "soil_ph": "Soil_pH", "soil ph": "Soil_pH",
    "ec": "EC", "ec_ds_m": "EC", "electrical_conductivity": "EC",
    "organic_c": "Organic_Carbon", "organic_carbon": "Organic_Carbon",
    "organic_carbon_%": "Organic_Carbon", "organiccarbon": "Organic_Carbon",
    "organic_matter": "Organic_Matter", "organic_matter_%": "Organic_Matter",
    "organicmatter": "Organic_Matter",
    "available_n": "Nitrogen", "nitrogen": "Nitrogen",
    "nitrogen_kg_ha": "Nitrogen", "n": "Nitrogen",
    "available_p": "Phosphorus", "phosphorus": "Phosphorus",
    "p2o5": "Phosphorus", "p2o5_kg_ha": "Phosphorus", "p": "Phosphorus",
    "available_k": "Potassium", "potassium": "Potassium",
    "k2o": "Potassium", "k2o_kg_ha": "Potassium", "k": "Potassium",
    "sulphur": "Sulphur", "sulphur_ppm": "Sulphur",
    "sulfur": "Sulphur", "s": "Sulphur",
    "iron": "Iron", "iron_ppm": "Iron", "iron_mgkg": "Iron", "fe": "Iron",
    "copper": "Copper", "copper_ppm": "Copper", "cu": "Copper",
    "cu_ppm": "Copper",
    "manganese": "Manganese", "mn": "Manganese", "mn_ppm": "Manganese",
    "zinc": "Zinc", "zn": "Zinc", "zn_ppm": "Zinc",
    "calcium": "Calcium", "ca": "Calcium",
    "magnesium": "Magnesium", "mg": "Magnesium",
    "boron": "Boron", "b": "Boron",
    "molybdenum": "Molybdenum", "mo": "Molybdenum",
    "treatment": "Treatment", "treatments": "Treatment",
    "fertilizer": "Fertilizer_Name", "fertilizer_name": "Fertilizer_Name",
    "fertiliser": "Fertilizer_Name", "amendment": "Fertilizer_Name",
    "organic_fertilizer": "Organic_Fertilizer",
    "biofertilizer": "Biofertilizer", "bio_fertilizer": "Biofertilizer",
    "dose": "Dose", "dose_kg_acre": "Dose", "dose_kg_ha": "Dose",
    "application_method": "Application_Method",
    "application_interval": "Application_Interval",
    "shoot_length": "Shoot_Length_cm", "shoot_length_cm": "Shoot_Length_cm",
    "shootheight": "Shoot_Length_cm", "shoot_height": "Shoot_Length_cm",
    "root_length": "Root_Length_cm", "root_length_cm": "Root_Length_cm",
    "plant_height": "Plant_Height_cm", "plant_height_cm": "Plant_Height_cm",
    "plantheight": "Plant_Height_cm",
    "shoot_biomass": "Shoot_Biomass_g", "shoot_biomass_g": "Shoot_Biomass_g",
    "root_biomass": "Root_Biomass_g", "root_biomass_g": "Root_Biomass_g",
    "leaf_area": "Leaf_Area_cm2", "leaf_area_cm2": "Leaf_Area_cm2",
    "leafarea": "Leaf_Area_cm2",
    "no_leaves": "Leaf_Number", "leaf_number": "Leaf_Number",
    "leaves_plant": "Leaf_Number", "number_of_leaves": "Leaf_Number",
    "tillers": "Tillers", "tiller_number": "Tillers",
    "no_tillers": "Tillers", "number_of_tillers": "Tillers",
    "root_diameter": "Root_Diameter_mm", "root_diameter_mm": "Root_Diameter_mm",
    "spad": "SPAD", "chlorophyll": "SPAD",
    "chlorophyll_spad": "SPAD", "chlorophyll_content": "SPAD",
    "moisture": "Moisture_Content", "moisture_content": "Moisture_Content",
    "dry_matter": "Dry_Matter", "dry_matter_pct": "Dry_Matter",
    "dry_matter_%": "Dry_Matter",
    "stem_diameter": "Stem_Diameter_mm", "stem_diameter_mm": "Stem_Diameter_mm",
    "branches": "Branches", "branches_plant": "Branches",
    "nodes": "Nodes", "flowers": "Flowers", "flowers_plant": "Flowers",
    "yield": "Yield_per_Plot", "yield_plot": "Yield_per_Plot",
    "yield_per_plot": "Yield_per_Plot", "yield_per_plot_g": "Yield_per_Plot",
    "fresh_weight": "Yield_per_Plot", "fresh_weight_g": "Yield_per_Plot",
    "fruit_weight": "Fruit_Weight", "fruit_weight_g": "Fruit_Weight",
    "fruit_number": "Fruit_Number", "fruits_plant": "Fruit_Number",
    "yield_per_acre": "Yield_per_Acre", "yield_per_ha": "Yield_per_Hectare",
    "yield_per_hectare": "Yield_per_Hectare",
    "spike_length": "Spike_Length", "spike_length_cm": "Spike_Length",
    "panicle_length": "Spike_Length",
    "seeds_per_spike": "Seeds_per_Spike", "seeds_spike": "Seeds_per_Spike",
    "grains_per_spike": "Seeds_per_Spike",
    "100_seed_weight": "100_Seed_Weight",
    "weight_100_seeds_g": "100_Seed_Weight",
    "hundred_seed_weight": "100_Seed_Weight", "test_weight": "100_Seed_Weight",
    "root_weight": "Root_Weight", "root_weight_g": "Root_Weight",
    "pod_weight": "Pod_Weight",
    "harvest_index": "Harvest_Index", "biomass_yield": "Biomass_Yield",
    "protein": "Protein", "protein_%": "Protein", "protein_content": "Protein",
    "ash": "Ash", "ash_%": "Ash", "ash_pct": "Ash",
    "gluten": "Gluten", "fiber": "Fiber",
    "carbohydrates": "Carbohydrates", "fat": "Fat", "oil": "Fat",
    "nitrogen_content": "Nitrogen_Content",
    "phosphorus_content": "Phosphorus_Content",
    "potassium_content": "Potassium_Content",
    "iron_content": "Iron_Content", "copper_content": "Copper_Content",
    "zinc_content": "Zinc_Content",
    "manganese_content": "Manganese_Content",
    "sulphur_content": "Sulphur_Content",
    "target_yield": "Target_Yield", "target_fertilizer": "Target_Fertilizer",
    "target_nitrogen": "Target_Nitrogen",
    "target_phosphorus": "Target_Phosphorus",
    "target_potassium": "Target_Potassium",
}


def _normalize(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = n.replace("-", "_").replace(".", "")
    return n


def _resolve_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate columns: keep the LAST occurrence."""
    if not df.columns.duplicated().any():
        return df
    return df.loc[:, ~df.columns.duplicated(keep="last")]


def standardize_columns(df: pd.DataFrame) -> tuple:
    """
    Map dataset column names to universal schema names.
    Handles multi-timepoint columns intelligently:
    - Highest timepoint (120) -> base column (Plant_Height_cm)
    - Lower timepoints (30,60,90) -> timepoint-specific columns (Plant_Height_30_cm)
    - Other duplicates are collapsed by taking the last occurrence.
    Returns (renamed_df, rename_tracker_dict).
    """
    rename = {}
    for col in df.columns:
        norm = _normalize(col)

        m = re.match(r"plantheight[_\s]*120[_\s]*cm", norm)
        if m:
            rename[col] = "Plant_Height_cm"
            continue
        m = re.match(r"plantheight[_\s]*(\d+)[_\s]*cm", norm)
        if m:
            rename[col] = f"Plant_Height_{m.group(1)}_cm"
            continue

        m = re.match(r"leaf[_\s]*area[_\s]*120[_\s]*cm2", norm)
        if m:
            rename[col] = "Leaf_Area_cm2"
            continue
        m = re.match(r"leaf[_\s]*area[_\s]*(\d+)[_\s]*cm2", norm)
        if m:
            rename[col] = f"Leaf_Area_{m.group(1)}_cm2"
            continue

        m = re.match(r"fruitweight[_\s]*120[_\s]*g", norm)
        if m:
            rename[col] = "Fruit_Weight"
            continue

        m = re.match(r"fruitdiameter[_\s]*120[_\s]*mm", norm)
        if m:
            rename[col] = "Fruit_Diameter_mm"
            continue

        m = re.match(r"yieldplot[_\s]*120[_\s]*g", norm)
        if m:
            rename[col] = "Yield_per_Plot"
            continue

        if re.match(r"yieldplot[_\s]*\d+[_\s]*g", norm):
            rename[col] = "Yield_per_Plot"
            continue
        if re.match(r"fruitweight[_\s]*\d+[_\s]*g", norm):
            rename[col] = "Fruit_Weight"
            continue
        if re.match(r"fruitdiameter[_\s]*\d+[_\s]*mm", norm):
            rename[col] = "Fruit_Diameter_mm"
            continue

        m = re.match(r"branches[_\s]*\d+", norm)
        if m:
            rename[col] = "Branches"
            continue

        m = re.match(r"flowers[_\s]*\d+", norm)
        if m:
            rename[col] = "Flowers"
            continue

        if norm in _EXACT_MAP:
            rename[col] = _EXACT_MAP[norm]
            continue

        rename[col] = col.strip()

    result = df.rename(columns=rename, errors="ignore")
    result = _resolve_duplicate_columns(result)
    return result, rename


def load_datasets(data_dir: Path):
    datasets = []
    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")
    for fp in xlsx_files:
        log.info("Loading %s", fp.name)
        try:
            df = pd.read_excel(fp, engine="openpyxl")
        except Exception as exc:
            log.warning("Could not read %s: %s", fp.name, exc)
            continue
        stem = fp.stem
        for suffix in ["_Master_Dataset", "_master_dataset", "_dataset", "_Dataset"]:
            stem = stem.replace(suffix, "")
        datasets.append((df, stem))
    return datasets


def infer_crop_from_data(df: pd.DataFrame, fallback: str) -> str:
    if "Crop" in df.columns:
        vals = df["Crop"].dropna().unique()
        if len(vals) == 1:
            return str(vals[0])
        if len(vals) > 0:
            return str(vals[0])
    return fallback


def add_universal_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def remove_duplicate_observations(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before - after:
        log.info("Removed %d duplicate rows", before - after)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    report = {
        "missing_columns": [],
        "duplicate_columns": [],
        "duplicate_rows": 0,
        "impossible_values": [],
        "validation_details": [],
    }
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            report["missing_columns"].append(col)
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        report["duplicate_columns"] = list(set(dup_cols))
    report["duplicate_rows"] = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["biomass", "yield", "weight", "length", "area"]):
            neg_count = int((df[col] < 0).sum())
            if neg_count:
                msg = f"{col}: {neg_count} negative value(s)"
                report["impossible_values"].append(msg)
                report["validation_details"].append(msg)
    if "Soil_pH" in df.columns:
        bad_ph = int((df["Soil_pH"].dropna() < 0).sum() + (df["Soil_pH"].dropna() > 14).sum())
        if bad_ph:
            msg = f"Soil_pH: {bad_ph} value(s) outside [0, 14]"
            report["impossible_values"].append(msg)
            report["validation_details"].append(msg)
    if "EC" in df.columns:
        bad_ec = int((df["EC"].dropna() < 0).sum())
        if bad_ec:
            msg = f"EC: {bad_ec} negative value(s)"
            report["impossible_values"].append(msg)
            report["validation_details"].append(msg)
    return report


def generate_validation_report_excel(report: dict, path: Path):
    rows = []
    for category, items in report.items():
        if isinstance(items, list):
            for item in items:
                rows.append({"Category": category, "Detail": str(item)})
        elif isinstance(items, (int, float)):
            rows.append({"Category": category, "Detail": str(items)})
        else:
            rows.append({"Category": category, "Detail": str(items)})
    vdf = pd.DataFrame(rows, columns=["Category", "Detail"])
    vdf.to_excel(path, index=False, engine="openpyxl")


def generate_metadata(schema_cols, datasets_info, additions, path: Path):
    meta = {
        "schema_name": "Universal Agricultural Schema",
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "total_columns": len(schema_cols),
        "column_groups": {
            "A. Paper Metadata": ["Paper_ID", "DOI", "Journal", "Year", "Authors", "Country"],
            "B. Crop Information": ["Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days"],
            "C. Experimental Design": ["Design", "Replications", "Plot_Size", "Spacing_Row", "Spacing_Plant", "Sample_Size", "Location", "State", "Site"],
            "D. Environment": ["Latitude", "Longitude", "Altitude", "Temperature_Max", "Temperature_Min", "Average_Temperature", "Rainfall", "Humidity"],
            "E. Soil Properties": ["Soil_pH", "EC", "Organic_Carbon", "Organic_Matter", "Nitrogen", "Phosphorus", "Potassium", "Sulphur", "Iron", "Copper", "Manganese", "Zinc", "Calcium", "Magnesium", "Boron", "Molybdenum"],
            "F. Fertilizer Information": ["Treatment", "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer", "Dose", "Application_Method", "Application_Interval"],
            "G. Crop Growth Parameters": ["Shoot_Length_cm", "Root_Length_cm", "Plant_Height_cm", "Shoot_Biomass_g", "Root_Biomass_g", "Leaf_Area_cm2", "Leaf_Number", "Tillers", "Root_Diameter_mm", "SPAD", "Moisture_Content", "Dry_Matter", "Stem_Diameter_mm", "Branches", "Nodes", "Flowers", "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm", "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2"],
            "H. Yield Parameters": ["Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare", "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm", "Spike_Length", "Seeds_per_Spike", "100_Seed_Weight", "Root_Weight", "Pod_Weight", "Harvest_Index", "Biomass_Yield"],
            "I. Grain Quality": ["Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat", "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content", "Iron_Content", "Copper_Content", "Zinc_Content", "Manganese_Content", "Sulphur_Content"],
            "J. ML Target Variables": ["Target_Yield", "Target_Fertilizer", "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium"],
        },
        "datasets_processed": datasets_info,
        "standardization_additions": additions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def run_pipeline():
    log.info("=" * 60)
    log.info("Universal Agricultural Schema Generator")
    log.info("=" * 60)

    # ---- STEP 1: Load datasets ----
    log.info("\n[STEP 1] Loading datasets from %s", DATA_DIR)
    datasets = load_datasets(DATA_DIR)
    log.info("Found %d dataset(s)", len(datasets))

    # ---- STEP 2 & 3: Standardize & build universal schema ----
    log.info("\n[STEP 2 & 3] Standardizing column names and building schema")
    merged_dfs = []
    all_additions = {}
    for df_raw, stem in datasets:
        log.info("Processing: %s (shape: %s)", stem, df_raw.shape)
        df, rename_tracker = standardize_columns(df_raw)

        # Record what changed (use rename_tracker, not zip)
        added = [(old, new) for old, new in rename_tracker.items() if old != new]
        all_additions[stem] = added

        # Fill Crop column from fallback if possible
        crop_val = infer_crop_from_data(df, stem)
        df["Crop"] = crop_val

        merged_dfs.append(df)

    # ---- Concatenate ----
    log.info("\nMerging datasets...")
    universal_df = pd.concat(merged_dfs, ignore_index=True, sort=False)

    # ---- Add all universal columns ----
    log.info("\nAdding all %d universal schema columns...", len(SCHEMA_COLUMNS))
    universal_df = add_universal_columns(universal_df)

    # ---- Reorder columns to match schema (schema cols first) ----
    schema_ordered = [c for c in SCHEMA_COLUMNS if c in universal_df.columns]
    extra_cols = [c for c in universal_df.columns if c not in SCHEMA_COLUMNS]
    universal_df = universal_df[schema_ordered + extra_cols]

    log.info("Schema shape: %s", universal_df.shape)

    # ---- STEP 4: Handle missing values (already NaN) ----
    log.info("\n[STEP 4] Missing values preserved as NaN")

    # ---- STEP 5: Remove duplicate rows ----
    log.info("\n[STEP 5] Removing duplicate observations...")
    universal_df = remove_duplicate_observations(universal_df)

    # ---- STEP 6: Standardize units (placeholder - extendable) ----
    log.info("\n[STEP 6] Standardizing units...")

    # ---- STEP 7: Validation ----
    log.info("\n[STEP 7] Running data validation...")
    validation_report = validate_data(universal_df)

    # ---- STEP 8: Outputs ----
    log.info("\n[STEP 8] Writing output files...")

    schema_xlsx = OUTPUT_DIR / "Universal_Agricultural_Schema.xlsx"
    universal_df.to_excel(schema_xlsx, index=False, engine="openpyxl")
    log.info("Written: %s", schema_xlsx)

    ml_csv = OUTPUT_DIR / "MachineLearning_Dataset.csv"
    universal_df.to_csv(ml_csv, index=False)
    log.info("Written: %s", ml_csv)

    meta_json = OUTPUT_DIR / "Schema_Metadata.json"
    datasets_info = []
    for df_raw, stem in datasets:
        datasets_info.append({
            "filename": f"{stem}.xlsx",
            "original_rows": len(df_raw),
            "original_columns": list(df_raw.columns),
        })
    generate_metadata(SCHEMA_COLUMNS, datasets_info, all_additions, meta_json)
    log.info("Written: %s", meta_json)

    val_report_xlsx = OUTPUT_DIR / "Validation_Report.xlsx"
    generate_validation_report_excel(validation_report, val_report_xlsx)
    log.info("Written: %s", val_report_xlsx)

    # ---- STEP 9: Report ----
    log.info("\n[STEP 9] SUMMARY REPORT")
    log.info("-" * 40)
    crops_in_data = universal_df["Crop"].dropna().unique() if "Crop" in universal_df.columns else []
    log.info("Number of crops: %d", len(crops_in_data))
    if len(crops_in_data):
        for c in crops_in_data:
            log.info("  - %s", c)
    log.info("Total rows: %d", len(universal_df))
    log.info("Total columns: %d", len(universal_df.columns))
    log.info("Schema columns defined: %d", len(SCHEMA_COLUMNS))

    log.info("\nRows contributed by each crop:")
    if "Crop" in universal_df.columns:
        for cname, count in universal_df["Crop"].value_counts().items():
            log.info("  %20s : %3d rows", str(cname), count)

    log.info("\nMissing values per column (top 20):")
    missing = universal_df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        for col, cnt in missing.head(20).items():
            log.info("  %30s : %3d missing", col, cnt)
    else:
        log.info("  (none)")

    log.info("\nColumns added during standardization:")
    total_added = sum(len(v) for v in all_additions.values())
    if total_added:
        for stem, adds in all_additions.items():
            for orig, new in adds:
                log.info("  %25s -> %s", orig, new)
    else:
        log.info("  (none)")

    log.info("\nValidation summary:")
    for cat, items in validation_report.items():
        if isinstance(items, list):
            if items:
                log.info("  %s: %d issue(s)", cat, len(items))
                for it in items[:5]:
                    log.info("    - %s", it)
            else:
                log.info("  %s: 0 issues", cat)
        else:
            log.info("  %s: %s", cat, items)

    log.info("\n%s", "=" * 60)
    log.info("Universal Agricultural Schema generation complete!")
    log.info("Output files in: %s", OUTPUT_DIR)
    log.info("%s", "=" * 60)

    return universal_df


if __name__ == "__main__":
    try:
        df = run_pipeline()
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)
