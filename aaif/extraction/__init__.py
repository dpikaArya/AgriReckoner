import os, re, sys
from io import StringIO

try:
    from agri_ai_agent.config.schema import UAMS_COLUMNS, NON_FEATURE_COLS, POST_HARVEST_VARIABLES, PRE_HARVEST_MEASUREMENTS
except ImportError:
    UAMS_COLUMNS = [
        'Paper_ID', 'DOI', 'Journal', 'Year', 'Authors', 'Country',
        'Crop', 'Scientific_Name', 'Variety', 'Season', 'Growth_Duration_Days', 'Growth_Stage',
        'Design', 'Replications', 'Plot_Size', 'Spacing_Row', 'Spacing_Plant',
        'Sample_Size', 'Location', 'State', 'Site',
        'Latitude', 'Longitude', 'Altitude',
        'Temperature_Max', 'Temperature_Min', 'Average_Temperature', 'Rainfall', 'Humidity',
        'Soil_pH', 'EC', 'Organic_Carbon', 'Organic_Matter',
        'Nitrogen', 'Phosphorus', 'Potassium',
        'Sulphur', 'Iron', 'Copper', 'Manganese', 'Zinc',
        'Calcium', 'Magnesium', 'Boron', 'Molybdenum',
        'Treatment', 'Fertilizer_Name', 'Organic_Fertilizer', 'Biofertilizer',
        'Dose', 'Application_Method', 'Application_Interval',
        'Shoot_Length_cm', 'Root_Length_cm', 'Plant_Height_cm',
        'Shoot_Biomass_g', 'Root_Biomass_g',
        'Leaf_Area_cm2', 'Leaf_Number', 'Tillers',
        'Root_Diameter_mm', 'SPAD', 'Moisture_Content', 'Dry_Matter',
        'Stem_Diameter_mm', 'Branches', 'Nodes', 'Flowers',
        'Plant_Height_30_cm', 'Plant_Height_60_cm', 'Plant_Height_90_cm',
        'Leaf_Area_30_cm2', 'Leaf_Area_60_cm2', 'Leaf_Area_90_cm2',
        'Yield_per_Plot', 'Yield_per_Acre', 'Yield_per_Hectare',
        'Fruit_Number', 'Fruit_Weight', 'Fruit_Diameter_mm',
        'Spike_Length', 'Seeds_per_Spike',
        '100_Seed_Weight', 'Root_Weight', 'Pod_Weight',
        'Harvest_Index', 'Biomass_Yield',
        'Protein', 'Ash', 'Gluten', 'Fiber', 'Carbohydrates', 'Fat',
        'Nitrogen_Content', 'Phosphorus_Content', 'Potassium_Content',
        'Iron_Content', 'Copper_Content', 'Zinc_Content',
        'Manganese_Content', 'Sulphur_Content',
        'Target_Yield', 'Target_Fertilizer',
        'Target_Nitrogen', 'Target_Phosphorus', 'Target_Potassium',
        'Growing_Degree_Days', 'Heat_Units', 'Harvest_Index_Calc',
        'Nitrogen_Use_Efficiency', 'Water_Use_Efficiency',
        'Rainfall_Anomaly', 'Stress_Index', 'Disease_Risk_Index',
        'Yield_per_Plant', 'Yield_per_Plot_Calc', 'Yield_per_Hectare_Calc',
        'Temp_x_Rainfall', 'N_x_P', 'Temp_squared',
        'Rainfall_7d_MA', 'Temp_7d_MA',
        'Feature_Available_Before_Prediction',
        'Crop_Code', 'Season_Code', 'Variety_Code',
        'Fertilizer_Code', 'Soil_Texture_Code', 'Country_Code',
        'Predicted_Yield', 'Expected_Biomass', 'Expected_Plant_Height',
        'Recommended_Fertilizer', 'Recommended_Dose', 'Recommended_Application_Interval',
        'Expected_Yield_Increase', 'Confidence_Score', 'Recommendation_Summary',
    ]
    NON_FEATURE_COLS = frozenset({
        'Treatment', 'Table_Row', 'Row_Index', '_source_page', '_reader',
        '_confidence', 'Source_File', 'Paper_ID', 'DOI', 'Journal',
        'Authors', 'Country', 'Location', 'State', 'Site',
        'Feature_Available_Before_Prediction',
        'Crop_Code', 'Season_Code', 'Variety_Code', 'Fertilizer_Code',
        'Soil_Texture_Code', 'Country_Code',
    })
    POST_HARVEST_VARIABLES = frozenset({
        'Protein', 'Ash', 'Gluten', 'Fiber', 'Carbohydrates', 'Fat',
        'Nitrogen_Content', 'Phosphorus_Content', 'Potassium_Content',
        'Iron_Content', 'Copper_Content', 'Zinc_Content',
        'Manganese_Content', 'Sulphur_Content',
        '100_Seed_Weight', 'Root_Weight', 'Pod_Weight',
        'Harvest_Index', 'Biomass_Yield',
        'Yield_per_Plot', 'Yield_per_Acre', 'Yield_per_Hectare',
        'Fruit_Number', 'Fruit_Weight', 'Fruit_Diameter_mm',
        'Spike_Length', 'Seeds_per_Spike',
        'Target_Yield', 'Target_Fertilizer',
        'Target_Nitrogen', 'Target_Phosphorus', 'Target_Potassium',
        'Yield_per_Plant', 'Yield_per_Plot_Calc', 'Yield_per_Hectare_Calc',
        'Nitrogen_Use_Efficiency', 'Water_Use_Efficiency',
        'Harvest_Index_Calc',
    })
    PRE_HARVEST_MEASUREMENTS = frozenset({
        'Leaf_Number', 'Tillers', 'Branches', 'Nodes', 'Flowers',
        'Plant_Height_30_cm', 'Plant_Height_60_cm', 'Plant_Height_90_cm',
        'Leaf_Area_30_cm2', 'Leaf_Area_60_cm2', 'Leaf_Area_90_cm2',
        'SPAD', 'Moisture_Content', 'Dry_Matter',
        'Shoot_Length_cm', 'Root_Length_cm',
        'Stem_Diameter_mm', 'Root_Diameter_mm',
        'Plant_Height_cm',
    })

UNIT_ALIASES = {
    'kg/ha': 'kg_per_ha', 'kg ha': 'kg_per_ha', 'kg ha-1': 'kg_per_ha',
    't/ha': 't_per_ha', 'q/ha': 'q_per_ha', 'q ha': 'q_per_ha',
    'kg/acre': 'kg_per_acre', 'kg acre': 'kg_per_acre',
    'g/plot': 'g_per_plot', 'g plot': 'g_per_plot',
    'cm': 'cm', 'mm': 'mm', 'm': 'm',
}

YIELD_COLS = ['Yield_per_Plot', 'Yield_per_Hectare', 'Yield_per_Acre', 'Biomass_Yield', 'Harvest_Index']
TREATMENT_RE = re.compile(
    r'^(T\d+[A-Z]?|[Ww]\d+[A-Z]?|Se\d+|CRF|SRF|SF|HAF|ZSF|SWF|'
    r'[Ff][SOB]{0,3}|Control|Ctrl|Check|CK|T0|T\d+[A-Z]?\d*|'
    r'\d{1,2}[A-Z]?)$'
)

CROP_VALIDATION = {
    'wheat': {'seasons': ['rabi', 'winter'], 'n_range': (40, 150), 'p_range': (20, 80), 'k_range': (20, 100)},
    'rice': {'seasons': ['kharif', 'summer', 'monsoon'], 'n_range': (60, 200), 'p_range': (20, 80), 'k_range': (20, 100)},
    'maize': {'seasons': ['kharif', 'summer'], 'n_range': (80, 200), 'p_range': (30, 100), 'k_range': (30, 100)},
    'bell pepper': {'seasons': ['summer', 'kharif'], 'n_range': (50, 200), 'p_range': (30, 100), 'k_range': (50, 200)},
    'black wheat': {'seasons': ['rabi', 'winter'], 'n_range': (40, 150), 'p_range': (20, 80), 'k_range': (20, 100)},
    'carrot': {'seasons': ['rabi', 'winter'], 'n_range': (40, 120), 'p_range': (20, 60), 'k_range': (40, 120)},
    'spinach': {'seasons': ['rabi', 'winter', 'summer'], 'n_range': (40, 150), 'p_range': (20, 80), 'k_range': (30, 100)},
    'chickpea': {'seasons': ['rabi', 'winter'], 'n_range': (20, 60), 'p_range': (20, 60), 'k_range': (20, 60)},
    'cowpea': {'seasons': ['kharif', 'summer'], 'n_range': (20, 80), 'p_range': (20, 60), 'k_range': (20, 60)},
    'soybean': {'seasons': ['kharif', 'monsoon'], 'n_range': (20, 60), 'p_range': (40, 100), 'k_range': (20, 80)},
}

def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower()) if s else ''

def parse_value(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in ('-', '–', '—', '', 'ns', 'NA', 'N/A', '.'):
        return None
    s = re.sub(r'[†‡*]', '', s)
    s = re.sub(r'\s*±\s*.*', '', s)
    s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        match = re.search(r'(\d+\.?\d*)', s)
        return float(match.group(1)) if match else None

def is_treatment(val):
    if not val:
        return False
    return bool(TREATMENT_RE.match(val.strip().rstrip('*†‡').strip()))
