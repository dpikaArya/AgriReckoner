"""
Universal Agricultural Machine Learning Schema (UAMS) v1.0
"""

UAMS_VERSION = "1.0"

UAMS_COLUMNS = [
    # A. Paper Metadata
    "Paper_ID", "DOI", "Journal", "Year", "Authors", "Country",
    # B. Crop Information
    "Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days", "Growth_Stage",
    # C. Experimental Design
    "Design", "Replications", "Plot_Size", "Spacing_Row", "Spacing_Plant",
    "Sample_Size", "Location", "State", "Site",
    # D. Environment
    "Latitude", "Longitude", "Altitude",
    "Temperature_Max", "Temperature_Min", "Average_Temperature",
    "Rainfall", "Humidity",
    # E. Soil Properties
    "Soil_pH", "EC",
    "Organic_Carbon", "Organic_Matter",
    "Nitrogen", "Phosphorus", "Potassium",
    "Sulphur", "Iron", "Copper", "Manganese", "Zinc",
    "Calcium", "Magnesium", "Boron", "Molybdenum",
    # F. Fertilizer Information
    "Treatment", "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer",
    "Dose", "Application_Method", "Application_Interval",
    # G. Crop Growth Parameters
    "Shoot_Length_cm", "Root_Length_cm", "Plant_Height_cm",
    "Shoot_Biomass_g", "Root_Biomass_g",
    "Leaf_Area_cm2", "Leaf_Number", "Tillers",
    "Root_Diameter_mm", "SPAD", "Moisture_Content", "Dry_Matter",
    "Stem_Diameter_mm", "Branches", "Nodes", "Flowers",
    "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm",
    "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2",
    # H. Yield Parameters
    "Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare",
    "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm",
    "Spike_Length", "Seeds_per_Spike",
    "100_Seed_Weight", "Root_Weight", "Pod_Weight",
    "Harvest_Index", "Biomass_Yield",
    # I. Grain Quality
    "Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat",
    "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content",
    "Iron_Content", "Copper_Content", "Zinc_Content",
    "Manganese_Content", "Sulphur_Content",
    # J. ML Target Variables
    "Target_Yield", "Target_Fertilizer",
    "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
    # K. Engineered Features (Agent 06)
    "Growing_Degree_Days", "Heat_Units", "Harvest_Index_Calc",
    "Nitrogen_Use_Efficiency", "Water_Use_Efficiency",
    "Rainfall_Anomaly", "Stress_Index", "Disease_Risk_Index",
    "Yield_per_Plant", "Yield_per_Plot_Calc", "Yield_per_Hectare_Calc",
    "Temp_x_Rainfall", "N_x_P", "Temp_ squared",
    "Rainfall_7d_MA", "Temp_7d_MA",
    # L. Leakage Labels (Agent 07)
    "Feature_Available_Before_Prediction",
    # M. Encoded Variables (Agent 08)
    "Crop_Code", "Season_Code", "Variety_Code",
    "Fertilizer_Code", "Soil_Texture_Code", "Country_Code",
    # N. ML Predictions
    "Predicted_Yield", "Expected_Biomass", "Expected_Plant_Height",
    "Recommended_Fertilizer", "Recommended_Dose", "Recommended_Application_Interval",
    "Expected_Yield_Increase", "Confidence_Score",
    "Recommendation_Summary",
]

SCHEMA_GROUPS = {
    "A. Paper Metadata": ["Paper_ID", "DOI", "Journal", "Year", "Authors", "Country"],
    "B. Crop Information": ["Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days", "Growth_Stage"],
    "C. Experimental Design": ["Design", "Replications", "Plot_Size", "Spacing_Row", "Spacing_Plant", "Sample_Size", "Location", "State", "Site"],
    "D. Environment": ["Latitude", "Longitude", "Altitude", "Temperature_Max", "Temperature_Min", "Average_Temperature", "Rainfall", "Humidity"],
    "E. Soil Properties": ["Soil_pH", "EC", "Organic_Carbon", "Organic_Matter", "Nitrogen", "Phosphorus", "Potassium", "Sulphur", "Iron", "Copper", "Manganese", "Zinc", "Calcium", "Magnesium", "Boron", "Molybdenum"],
    "F. Fertilizer Information": ["Treatment", "Fertilizer_Name", "Organic_Fertilizer", "Biofertilizer", "Dose", "Application_Method", "Application_Interval"],
    "G. Crop Growth Parameters": ["Shoot_Length_cm", "Root_Length_cm", "Plant_Height_cm", "Shoot_Biomass_g", "Root_Biomass_g", "Leaf_Area_cm2", "Leaf_Number", "Tillers", "Root_Diameter_mm", "SPAD", "Moisture_Content", "Dry_Matter", "Stem_Diameter_mm", "Branches", "Nodes", "Flowers", "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm", "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2"],
    "H. Yield Parameters": ["Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare", "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm", "Spike_Length", "Seeds_per_Spike", "100_Seed_Weight", "Root_Weight", "Pod_Weight", "Harvest_Index", "Biomass_Yield"],
    "I. Grain Quality": ["Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat", "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content", "Iron_Content", "Copper_Content", "Zinc_Content", "Manganese_Content", "Sulphur_Content"],
    "J. ML Target Variables": ["Target_Yield", "Target_Fertilizer", "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium"],
    "K. Engineered Features": ["Growing_Degree_Days", "Heat_Units", "Harvest_Index_Calc", "Nitrogen_Use_Efficiency", "Water_Use_Efficiency", "Rainfall_Anomaly", "Stress_Index", "Disease_Risk_Index", "Yield_per_Plant", "Yield_per_Plot_Calc", "Yield_per_Hectare_Calc", "Temp_x_Rainfall", "N_x_P", "Temp_ squared", "Rainfall_7d_MA", "Temp_7d_MA"],
    "L. Leakage Labels": ["Feature_Available_Before_Prediction"],
    "M. Encoded Variables": ["Crop_Code", "Season_Code", "Variety_Code", "Fertilizer_Code", "Soil_Texture_Code", "Country_Code"],
    "N. ML Predictions": ["Predicted_Yield", "Expected_Biomass", "Expected_Plant_Height", "Recommended_Fertilizer", "Recommended_Dose", "Recommended_Application_Interval", "Expected_Yield_Increase", "Confidence_Score", "Recommendation_Summary"],
}

VARIANT_MAP = {
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
    "growth_stage": "Growth_Stage", "phenological_stage": "Growth_Stage",
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
    "pod_weight": "Pod_Weight", "pod_weight_g": "Pod_Weight",
    "fruit_diameter": "Fruit_Diameter_mm", "fruit_diameter_mm": "Fruit_Diameter_mm",
    "harvest_index": "Harvest_Index", "biomass_yield": "Biomass_Yield",
    "protein": "Protein", "protein_%": "Protein", "protein_content": "Protein",
    "ash": "Ash", "ash_%": "Ash", "ash_pct": "Ash",
    "gluten": "Gluten", "fiber": "Fiber", "crude_fiber": "Fiber",
    "carbohydrates": "Carbohydrates", "carbs": "Carbohydrates",
    "fat": "Fat", "oil": "Fat", "crude_fat": "Fat",
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
    # N. ML Predictions
    "predicted_yield": "Predicted_Yield",
    "expected_biomass": "Expected_Biomass",
    "expected_plant_height": "Expected_Plant_Height",
    "recommended_fertilizer": "Recommended_Fertilizer",
    "recommended_dose": "Recommended_Dose",
    "recommended_application_interval": "Recommended_Application_Interval",
    "expected_yield_increase": "Expected_Yield_Increase",
    "recommendation_summary": "Recommendation_Summary",
    "confidence": "Confidence_Score",
    "confidence_score": "Confidence_Score",
}

NON_FEATURE_COLS = frozenset({
    "Treatment", "Table_Row", "Row_Index", "_source_page", "_reader",
    "_confidence", "Source_File", "Paper_ID", "DOI", "Journal",
    "Authors", "Country", "Location", "State", "Site",
    "Feature_Available_Before_Prediction",
    "Crop_Code", "Season_Code", "Variety_Code", "Fertilizer_Code",
    "Soil_Texture_Code", "Country_Code",
})

POST_HARVEST_VARIABLES = frozenset({
    "Protein", "Ash", "Gluten", "Fiber", "Carbohydrates", "Fat",
    "Nitrogen_Content", "Phosphorus_Content", "Potassium_Content",
    "Iron_Content", "Copper_Content", "Zinc_Content",
    "Manganese_Content", "Sulphur_Content",
    "100_Seed_Weight", "Root_Weight", "Pod_Weight",
    "Harvest_Index", "Biomass_Yield",
    "Yield_per_Plot", "Yield_per_Acre", "Yield_per_Hectare",
    "Fruit_Number", "Fruit_Weight", "Fruit_Diameter_mm",
    "Spike_Length", "Seeds_per_Spike",
    "Target_Yield", "Target_Fertilizer",
    "Target_Nitrogen", "Target_Phosphorus", "Target_Potassium",
    "Yield_per_Plant", "Yield_per_Plot_Calc", "Yield_per_Hectare_Calc",
    "Nitrogen_Use_Efficiency", "Water_Use_Efficiency",
    "Harvest_Index_Calc",
})

PRE_HARVEST_MEASUREMENTS = frozenset({
    "Leaf_Number", "Tillers", "Branches", "Nodes", "Flowers",
    "Plant_Height_30_cm", "Plant_Height_60_cm", "Plant_Height_90_cm",
    "Leaf_Area_30_cm2", "Leaf_Area_60_cm2", "Leaf_Area_90_cm2",
    "SPAD", "Moisture_Content", "Dry_Matter",
    "Shoot_Length_cm", "Root_Length_cm",
    "Stem_Diameter_mm", "Root_Diameter_mm",
    "Plant_Height_cm",
})

# Measured agronomic outcomes to model when no synthetic Target_* column is populated,
# in order of preference. Mirrors the columns the validated monolith actually trained on.
MEASURED_TARGETS = [
    "Yield_per_Hectare", "Yield_per_Plot", "Plant_Height_cm", "SPAD", "Shoot_Biomass_g",
]

MIN_TARGET_ROWS = 5


def resolve_target_column(df, requested=None, extra_targets=None, min_non_null=MIN_TARGET_ROWS):
    """Return a usable target column name, or None.

    Prefers the requested column, then any synthetic Target_* passed as extra_targets,
    then the measured agronomic outcomes; a candidate is accepted only if it exists and
    has at least ``min_non_null`` non-null values.
    """
    candidates = []
    if requested:
        candidates.append(requested)
    if extra_targets:
        candidates.extend(extra_targets)
    candidates.extend(MEASURED_TARGETS)
    seen = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in df.columns and df[col].notna().sum() >= min_non_null:
            return col
    return None
