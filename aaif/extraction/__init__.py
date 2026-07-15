import os, re, sys
from io import StringIO

UAMS_COLUMNS = [
    'Paper_ID', 'Year', 'Crop', 'Variety', 'Location', 'Season', 'DOI', 'Source_File',
    'Treatment', 'Design', 'Replications', 'Rainfall', 'Temperature_C',
    'Plant_Height_cm', 'Tillers', 'Spike_Length', 'Seeds_per_Spike', '100_Seed_Weight',
    'Yield_per_Plot', 'Yield_per_Hectare', 'Yield_per_Acre', 'Biomass_Yield',
    'Harvest_Index', 'SPAD', 'Leaf_Area_cm2', 'Fruit_Weight', 'Fruit_Diameter_mm',
    'Fruit_Length', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Soil_pH',
    'Protein', 'Amylose', 'Iron_ppm', 'Zinc_ppm', 'Copper', 'Manganese_ppm',
    'Sodium', 'Magnesium', 'Calcium', 'Sulfur', 'Boron',
]

UNIT_ALIASES = {
    'kg/ha': 'kg_per_ha', 'kg ha': 'kg_per_ha', 'kg ha-1': 'kg_per_ha',
    't/ha': 't_per_ha', 'q/ha': 'q_per_ha', 'q ha': 'q_per_ha',
    'kg/acre': 'kg_per_acre', 'kg acre': 'kg_per_acre',
    'g/plot': 'g_per_plot', 'g plot': 'g_per_plot',
    'cm': 'cm', 'mm': 'mm', 'm': 'm',
}

YIELD_COLS = ['Yield_per_Plot', 'Yield_per_Hectare', 'Yield_per_Acre', 'Biomass_Yield', 'Harvest_Index']
TREATMENT_RE = re.compile(r'^(T\d+|[Ww]\d+|Se\d+|CRF|SRF|SF|HAF|ZSF|SWF|[Ff][SOB]{0,3})$')

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
