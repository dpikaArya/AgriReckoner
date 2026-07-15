import os, re, sys, json
import pandas as pd
import pdfplumber
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from agri_ai_agent.config.schema import UAMS_COLUMNS
except ImportError:
    UAMS_COLUMNS = ['Paper_ID', 'Year', 'Crop', 'DOI', 'Source_File', 'Treatment',
        'Design', 'Rainfall', 'Plant_Height_cm', 'Tillers', 'Spike_Length',
        'Seeds_per_Spike', '100_Seed_Weight', 'Yield_per_Plot', 'Yield_per_Hectare',
        'Yield_per_Acre', 'Biomass_Yield', 'Harvest_Index']

UAMS_MAP = {
    'spad': 'SPAD', 'spad value': 'SPAD', 'chlorophyll': 'SPAD',
    'plant height': 'Plant_Height_cm', 'plant length': 'Plant_Height_cm',
    'shoot length': 'Plant_Height_cm', 'shoot': 'Plant_Height_cm',
    'root length': 'Plant_Height_cm', 'root': 'Plant_Height_cm',
    'height': 'Plant_Height_cm',
    'tillers': 'Tillers', 'tiller': 'Tillers', 'productive tillers': 'Tillers', 'tiller number': 'Tillers',
    'spike length': 'Spike_Length', 'ear length': 'Spike_Length', 'panicle length': 'Spike_Length',
    'spike number': 'Spike_Length', 'spikenumber': 'Spike_Length',
    'seeds per spike': 'Seeds_per_Spike', 'grains per spike': 'Seeds_per_Spike',
    'grain per spike': 'Seeds_per_Spike', 'grains per ear': 'Seeds_per_Spike',
    '100 seed weight': '100_Seed_Weight', '1000 grain weight': '100_Seed_Weight',
    '1000-grain weight': '100_Seed_Weight', 'test weight': '100_Seed_Weight',
    'seed index': '100_Seed_Weight', 'grain weight': '100_Seed_Weight',
    'seed weight': '100_Seed_Weight',
    'grain yield': 'Yield_per_Hectare', 'grainyield': 'Yield_per_Hectare',
    'plot yield': 'Yield_per_Plot', 'yield per plot': 'Yield_per_Plot',
    'yield per hectare': 'Yield_per_Hectare', 'yield t ha': 'Yield_per_Hectare',
    't ha': 'Yield_per_Hectare',
    'yield per acre': 'Yield_per_Acre', 'q per acre': 'Yield_per_Acre',
    'biomass': 'Biomass_Yield', 'fresh weight': 'Biomass_Yield', 'dry weight': 'Biomass_Yield',
    'harvest index': 'Harvest_Index',
    'fruit weight': 'Fruit_Weight', 'fruit diameter': 'Fruit_Diameter_mm', 'fruit length': 'Fruit_Length',
    'root diameter': 'Fruit_Diameter_mm',
    'leaf area': 'Leaf_Area_cm2', 'leaf area cm2': 'Leaf_Area_cm2',
    'nitrogen': 'Nitrogen', 'phosphorus': 'Phosphorus', 'potassium': 'Potassium',
    'organic carbon': 'Organic_Carbon', 'soil ph': 'Soil_pH',
    'protein': 'Protein', 'protein content': 'Protein', 'grain protein content': 'Protein',
    'iron': 'Iron_ppm', 'fe': 'Iron_ppm', 'zinc': 'Zinc_ppm', 'zn': 'Zinc_ppm',
    'copper': 'Copper', 'cu': 'Copper',
    'sodium': 'Sodium', 'na': 'Sodium', 'magnesium': 'Magnesium', 'mg': 'Magnesium',
    'calcium': 'Calcium', 'ca': 'Calcium',
    'amylose': 'Amylose', 'amylose content': 'Amylose',
    'of leaves': 'Leaf_Count',
}

UNIT_PAT = re.compile(r'\([^)]*\)')
TID_PAT = re.compile(r'[†‡*]')

TREATMENT_RE = re.compile(r'^(T\d+|[Ww]\d+|Se\d+|CRF|SRF|SF|HAF|ZSF|SWF|[Ff][SOB]{0,3})$')

def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip().lower()) if s else ''

def parse_value(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s or s in ('-', '–', '—', '', 'ns', 'NA', 'N/A'):
        return None
    s = re.sub(r'[†‡*]', '', s)
    s = re.sub(r'\s*±\s*.*', '', s)
    s = s.replace(',', '')
    try:
        return float(s)
    except ValueError:
        match = re.search(r'(\d+\.?\d*)', s)
        return float(match.group(1)) if match else None

def detect_treatment_column(header_row):
    for ci, cell in enumerate(header_row):
        c = norm(cell)
        if c in ('treatment', 'treatments', 'trt', 'treatment code'):
            return ci
        if TREATMENT_RE.match(cell.strip().rstrip('*†‡').strip()) if cell and cell.strip() else False:
            return ci
    return 0 if header_row and header_row[0] and norm(header_row[0]) == '-' else -1

def extract_variable(cell):
    c = norm(cell)
    stripped = UNIT_PAT.sub('', c).strip()
    if stripped in UAMS_MAP:
        return UAMS_MAP[stripped]
    tokens = stripped.split()
    for key, val in sorted(UAMS_MAP.items(), key=lambda x: -len(x[0])):
        if key in stripped:
            return val
    result = c.replace(' ', '_').replace('/', '_per_').replace('-', '_')
    result = re.sub(r'_+', '_', result).strip('_')
    return result.title().replace(' ', '')

def is_treatment_row(row, treat_col):
    if treat_col < 0 or treat_col >= len(row):
        return False
    val = str(row[treat_col]).strip() if row[treat_col] else ''
    if not val:
        return False
    if TREATMENT_RE.match(val.rstrip('*†‡').strip()):
        return True
    return False

def extract_tables_pdfplumber(pdf_path):
    rows = []
    meta = {'file': os.path.basename(pdf_path)}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            meta['pages'] = len(pdf.pages)
            for pi, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if len(table) < 3:
                        continue
                    header = table[0]
                    treat_col = detect_treatment_column(header)
                    if treat_col < 0 and len(table) > 1:
                        treat_col = detect_treatment_column(table[1]) if table[1] else -1
                    if treat_col < 0:
                        continue
                    header_map = {}
                    for ci, cell in enumerate(header):
                        if ci == treat_col:
                            header_map[ci] = 'Treatment'
                        elif cell and norm(cell) not in ('-', '', 'seasons', 'season'):
                            var = extract_variable(cell)
                            header_map[ci] = var
                    if not header_map:
                        continue
                    for ri in range(1, len(table)):
                        row = table[ri]
                        if not row or not any(row):
                            continue
                        if not is_treatment_row(row, treat_col):
                            continue
                        t_val = str(row[treat_col]).strip().rstrip('*†‡').strip() if row[treat_col] else ''
                        if not t_val:
                            continue
                        data = {
                            'Treatment': t_val,
                            'Source_File': meta['file'],
                            'Page': pi + 1,
                            'Table_Row': ri,
                        }
                        for ci, varname in header_map.items():
                            if ci < len(row) and ci != treat_col and varname:
                                v = parse_value(row[ci])
                                if v is not None:
                                    data[varname] = v
                        rows.append(data)
    except Exception as e:
        pass
    return rows, meta

def extract_simple_tables(pdf_path, existing):
    new_rows = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if len(table) < 3:
                        continue
                    treat_col = -1
                    for ri in range(min(3, len(table))):
                        for ci, cell in enumerate(table[ri]):
                            c = str(cell).strip() if cell else ''
                            if TREATMENT_RE.match(c.rstrip('*†‡').strip()):
                                treat_col = ci
                                break
                        if treat_col >= 0:
                            break
                    if treat_col < 0:
                        continue
                    values_map = {}
                    for ri in range(len(table)):
                        row = table[ri]
                        if not row or treat_col >= len(row):
                            continue
                        t_val = str(row[treat_col]).strip().rstrip('*†‡').strip() if row[treat_col] else ''
                        if not t_val or not TREATMENT_RE.match(t_val):
                            continue
                        for ci, cell in enumerate(row):
                            if ci == treat_col:
                                continue
                            if cell is None or str(cell).strip() in ('-', '', '–'):
                                continue
                            header_text = str(table[0][ci]).strip() if table[0] and ci < len(table[0]) and table[0][ci] else ''
                            header_norm = norm(header_text)
                            if header_norm in ('-', '', 'seasons', 'season'):
                                continue
                            if '(' in header_text and ')' in header_text:
                                unit_part = header_text[header_text.index('('):header_text.index(')')+1]
                                clean_header = header_text[:header_text.index('(')].strip() + ' ' + unit_part
                            else:
                                clean_header = header_text
                            var = extract_variable(clean_header)
                            if var == norm(clean_header).replace(' ', '_'):
                                pass
                            v = parse_value(cell)
                            if v is not None:
                                if t_val not in values_map:
                                    values_map[t_val] = {}
                                if var not in values_map[t_val]:
                                    values_map[t_val][var] = v
                    for t_val, vals in values_map.items():
                        row_data = {'Treatment': t_val, 'Source_File': os.path.basename(pdf_path)}
                        for k, v in vals.items():
                            if v is not None:
                                row_data[k] = v
                        new_rows.append(row_data)
    except Exception as e:
        pass
    return new_rows

def extract_fpls16_tables(pdf_path):
    rows = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if len(table) < 5:
                        continue
                    h1 = [str(c).strip() if c else '' for c in (table[0] or [])]
                    h2 = [str(c).strip() if c else '' for c in (table[1] or [])]
                    if 'treatments' not in norm(h2[1] if len(h2) > 1 else ''):
                        continue
                    metric_names = {}
                    for ci in range(2, len(h1), 4):
                        raw = h1[ci]
                        if not raw:
                            continue
                        var = extract_variable(raw)
                        if var not in ('Yield_per_Hectare', 'Spike_Length', 'Protein', 'Amylose',
                                       'Iron_ppm', 'Zinc_ppm'):
                            normal = norm(UNIT_PAT.sub('', raw))
                            if 'yield' in normal or 'grain' in normal:
                                var = 'Yield_per_Hectare'
                            elif 'spike' in normal:
                                var = 'Spike_Length'
                            elif 'protein' in normal:
                                var = 'Protein'
                            elif 'amylose' in normal:
                                var = 'Amylose'
                            elif 'fe' in normal or 'iron' in normal:
                                var = 'Iron_ppm'
                            elif 'zn' in normal or 'zinc' in normal:
                                var = 'Zinc_ppm'
                            elif 'selenium' in normal or 'se' == normal.strip():
                                var = 'Selenium'
                        if var:
                            for j in range(4):
                                ci2 = ci + j
                                if ci2 < len(h1):
                                    metric_names[ci2] = var
                    for ri in range(3, len(table)):
                        row = table[ri]
                        if not row or len(row) < 2 or not row[1]:
                            continue
                        t_val = str(row[1]).strip().rstrip('*†‡').strip()
                        t_match = re.match(r'^[Ww](\d)$', t_val)
                        if not t_match:
                            continue
                        tid = 'W' + t_match.group(1)
                        data = {'Treatment': tid, 'Source_File': os.path.basename(pdf_path)}
                        for ci in range(2, min(len(row), len(h1))):
                            if ci in metric_names and row[ci]:
                                v = parse_value(row[ci])
                                if v is not None:
                                    var = metric_names[ci]
                                    if var in ('Yield_per_Hectare',):
                                        data[var] = v * 10
                                    else:
                                        data[var] = v
                        rows.append(data)
    except Exception as e:
        pass
    return rows

def process_pdf(pdf_path):
    """Process a single PDF and return treatment rows."""
    pdf_name = os.path.basename(pdf_path)
    source_rows = []

    tables, meta = extract_tables_pdfplumber(pdf_path)
    source_rows.extend(tables)

    fpls_rows = extract_fpls16_tables(pdf_path)
    seen_treatments = set()
    for sr in source_rows:
        seen_treatments.add(sr.get('Treatment', '') + str(sr.get('Page', '')) + str(sr.get('Table_Row', '')))
    for r in fpls_rows:
        key = r.get('Treatment', '') + 'fpls'
        if key not in seen_treatments:
            source_rows.append(r)
            seen_treatments.add(key)

    if not source_rows:
        simple = extract_simple_tables(pdf_path, [])
        source_rows.extend(simple)

    return source_rows, meta

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Path to directory containing PDFs')
    parser.add_argument('--output', default='outputs/pdfplumber_extraction.xlsx')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    all_rows = []
    pdf_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')])
    print(f"Processing {len(pdf_files)} PDFs with pdfplumber...")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_name)
        rows, meta = process_pdf(pdf_path)
        print(f"  {pdf_name}: {len(rows)} treatment rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No treatment rows found. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    cols = ['Source_File', 'Treatment'] + sorted([c for c in df.columns if c not in ('Source_File', 'Treatment')])
    df = df[[c for c in cols if c in df.columns]]

    for c in UAMS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_excel(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"  Treatment rows: {len(df)}")
    print(f"  Columns with data: {[c for c in df.columns if df[c].notna().any()]}")

    yield_cols = ['Yield_per_Plot', 'Yield_per_Hectare', 'Yield_per_Acre', 'Biomass_Yield', 'Harvest_Index']
    for yc in yield_cols:
        if yc in df.columns:
            n = df[yc].notna().sum()
            print(f"  {yc}: {n} non-null")

if __name__ == '__main__':
    main()
