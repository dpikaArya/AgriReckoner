import os, re
from . import UAMS_COLUMNS, parse_value, is_treatment, YIELD_COLS, norm

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_OK = True
except ImportError:
    PDFMINER_OK = False

SECTION_PATTERNS = {
    'title': re.compile(r'^.{0,3}(?:title|abstract|introduction)', re.I),
    'abstract': re.compile(r'abstract', re.I),
    'materials_methods': re.compile(r'(?:materials?\s*(?:and|&)\s*methods|methodology|experimental\s*setup)', re.I),
    'results': re.compile(r'^results?\s', re.I),
    'discussion': re.compile(r'^discussion\s', re.I),
    'conclusion': re.compile(r'^conclusion', re.I),
    'references': re.compile(r'^references\b', re.I),
}

TREATMENT_PATS = [
    re.compile(r'^(T\d+)$'),
    re.compile(r'^(F[SOB]{0,3}|FS|FO|FB|FSOB)$'),
    re.compile(r'^([Ww]\d+)$'),
    re.compile(r'^(Se\d+)$'),
    re.compile(r'^(CRF|SRF|SF|HAF|ZSF|SWF)$'),
]

HEADER_MAP = {
    'yield per plot': 'Yield_per_Plot', 'plot yield': 'Yield_per_Plot',
    'grain yield': 'Yield_per_Hectare', 'grainyield': 'Yield_per_Hectare',
    'yield per hectare': 'Yield_per_Hectare', 'yield t ha': 'Yield_per_Hectare',
    't ha': 'Yield_per_Hectare', 'yield (t/ha)': 'Yield_per_Hectare',
    'yield per acre': 'Yield_per_Acre', 'q per acre': 'Yield_per_Acre',
    'biomass': 'Biomass_Yield', 'fresh weight': 'Biomass_Yield', 'dry weight': 'Biomass_Yield',
    'harvest index': 'Harvest_Index',
    'plant height': 'Plant_Height_cm', 'shoot length': 'Plant_Height_cm',
    'spad': 'SPAD', 'spad value': 'SPAD', 'chlorophyll': 'SPAD',
    'tillers': 'Tillers', 'productive tillers': 'Tillers',
    'spike length': 'Spike_Length', 'ear length': 'Spike_Length', 'panicle length': 'Spike_Length',
    'spike number': 'Spike_Length',
    'seeds per spike': 'Seeds_per_Spike', 'grains per spike': 'Seeds_per_Spike',
    '100 seed weight': '100_Seed_Weight', '1000 grain weight': '100_Seed_Weight', 'test weight': '100_Seed_Weight',
    'leaf area': 'Leaf_Area_cm2',
    'fruit weight': 'Fruit_Weight', 'fruit diameter': 'Fruit_Diameter_mm',
    'nitrogen': 'Nitrogen', 'phosphorus': 'Phosphorus', 'potassium': 'Potassium',
    'organic carbon': 'Organic_Carbon', 'soil ph': 'Soil_pH',
    'protein': 'Protein', 'protein content': 'Protein',
    'iron': 'Iron_ppm', 'zinc': 'Zinc_ppm',
    'copper': 'Copper', 'calcium': 'Calcium', 'magnesium': 'Magnesium',
    'sodium': 'Sodium',
}

class PdfminerReader:
    def __init__(self):
        self.name = 'pdfminer'
        self.success = False
        self.confidence = 0.0
        self.metadata = {}
        self.rows = []
        self.tables = []

    def extract(self, pdf_path):
        result = {
            'reader': self.name,
            'success': False,
            'confidence': 0.0,
            'rows': [],
            'tables': [],
            'metadata': {},
        }
        if not PDFMINER_OK:
            result['error'] = 'pdfminer.six not installed'
            return result

        try:
            la_params = LAParams(detect_vertical=True, all_texts=True)
            text = extract_text(pdf_path, laparams=la_params)
            lines = text.split('\n')

            sections = self._split_sections(lines)
            sections['Source_File'] = os.path.basename(pdf_path)
            result['metadata'] = sections

            table_rows = self._extract_tables_from_text(lines)
            if not table_rows:
                table_rows = self._column_major_fallback(lines)
            deduped = self._deduplicate_rows(table_rows)
            for r in deduped:
                r['Source_File'] = os.path.basename(pdf_path)
            result['rows'] = deduped
            result['tables'] = self._build_tables(deduped)

            n_treatments = len(set(r.get('Treatment', '') for r in deduped))
            result['confidence'] = min(0.7, 0.3 + n_treatments * 0.05)
            result['success'] = len(deduped) > 0 or bool(sections.get('title'))

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        self.tables = result['tables']
        self.metadata = result['metadata']
        return result

    def _deduplicate_rows(self, rows):
        seen = {}
        for r in rows:
            tid = r.get('Treatment', '')
            best = seen.get(tid, {})
            n_new = sum(1 for k, v in r.items() if v is not None and k not in ('Treatment',))
            n_best = sum(1 for k, v in best.items() if v is not None and k not in ('Treatment',))
            if n_new > n_best:
                seen[tid] = r
            elif n_new == n_best and n_new > 0:
                for k, v in r.items():
                    if v is not None and k not in best:
                        best[k] = v
        return list(seen.values())

    def _split_sections(self, lines):
        sections = {
            'title': '', 'abstract': '', 'materials_methods': '',
            'results': '', 'discussion': '', 'conclusion': '',
            'full_text': '\n'.join(lines[:2000]),
        }
        current = None
        buf = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            for key, pat in SECTION_PATTERNS.items():
                if pat.search(s) and len(s) < 200:
                    if current and buf:
                        sections[current] = '\n'.join(buf)
                    current = key
                    buf = [s]
                    break
            else:
                if current:
                    buf.append(s)
        if current and buf:
            sections[current] = '\n'.join(buf)
        return sections

    def _extract_tables_from_text(self, lines):
        rows = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            tid_val = line.rstrip('*†‡').strip()
            if is_treatment(tid_val):
                block = [line]
                i += 1
                while i < len(lines):
                    nxt = lines[i].strip()
                    if not nxt:
                        if i + 1 < len(lines) and not lines[i+1].strip():
                            break
                        i += 1
                        continue
                    if is_treatment(nxt.rstrip('*†‡').strip()):
                        break
                    block.append(nxt)
                    i += 1
                if len(block) >= 3:
                    row_data = self._parse_block(tid_val, block[1:])
                    if row_data:
                        rows.append(row_data)
                continue
            i += 1
        return rows

    def _column_major_fallback(self, lines):
        TREATMENT_SEQ_PATS = [
            re.compile(r'^(T\d+)$', re.I),
            re.compile(r'^(T\d+[A-Z]?)$', re.I),
            re.compile(r'^(W\d+)$', re.I),
            re.compile(r'^(Se\d+)$', re.I),
        ]
        VARIABLE_MAP = {
            'spad': 'SPAD', 'chlorophyll': 'SPAD',
            'plant height': 'Plant_Height_cm', 'height': 'Plant_Height_cm',
            'shoot length': 'Plant_Height_cm', 'root length': 'Plant_Height_cm',
            'tillers': 'Tillers', 'tiller': 'Tillers',
            'spike length': 'Spike_Length', 'spike number': 'Spike_Length',
            'seeds per spike': 'Seeds_per_Spike', 'grains per spike': 'Seeds_per_Spike',
            '100 seed weight': '100_Seed_Weight', 'test weight': '100_Seed_Weight',
            '1000 grain weight': '100_Seed_Weight',
            'yield': 'Yield_per_Plot', 'grain yield': 'Yield_per_Plot',
            'biomass': 'Biomass_Yield',
            'harvest index': 'Harvest_Index',
            'leaf area': 'Leaf_Area_cm2',
        }

        rows = []
        tid_positions = []
        for i, line in enumerate(lines):
            s = line.strip().rstrip('*†‡').strip()
            for pat in TREATMENT_SEQ_PATS:
                m = pat.match(s)
                if m:
                    tid_positions.append((i, m.group(1)))
                    break

        if len(tid_positions) < 3:
            return rows

        seq_groups = []
        cur = [tid_positions[0]]
        for i in range(1, len(tid_positions)):
            if tid_positions[i][0] == tid_positions[i-1][0] + 1:
                cur.append(tid_positions[i])
            else:
                if len(cur) >= 3:
                    seq_groups.append(cur)
                cur = [tid_positions[i]]
        if len(cur) >= 3:
            seq_groups.append(cur)

        for group in seq_groups:
            tids = [t[1] for t in group]
            block_start = group[-1][0] + 1
            block_end = min(block_start + 120, len(lines))

            header_positions = []
            for i in range(block_start, block_end):
                s = lines[i].strip()
                if s and len(s) < 80 and not s.replace('.', '').replace('-', '').isdigit() and not is_treatment(s.rstrip('*†‡').strip()):
                    header_positions.append((i, s))
                    if len(header_positions) >= 4:
                        break

            header_map = {}
            for hp_i, (hp_line_idx, hp_text) in enumerate(header_positions):
                h_norm = norm(hp_text)
                if h_norm in VARIABLE_MAP:
                    header_map[hp_i] = VARIABLE_MAP[h_norm]

            if not header_map:
                for hp_i, (hp_line_idx, hp_text) in enumerate(header_positions):
                    for key, var in sorted(VARIABLE_MAP.items(), key=lambda x: -len(x[0])):
                        if key in norm(hp_text):
                            header_map[hp_i] = var
                            break

            if header_map:
                for ti, tid in enumerate(tids):
                    data = {'Treatment': tid}
                    for hp_i, var in header_map.items():
                        val_idx = block_start + hp_i * len(tids) + ti
                        if val_idx < len(lines):
                            v = parse_value(lines[val_idx])
                            if v is not None and v > 0:
                                data[var] = v
                    if len(data) > 1:
                        rows.append(data)

            if not rows:
                for ti, tid in enumerate(tids):
                    data = {'Treatment': tid}
                    value_idx = 0
                    for hi in range(block_start, block_end):
                        v = parse_value(lines[hi])
                        if v is not None and v > 0.01:
                            var_name = f'Value_{value_idx}'
                            for yc in ['Yield_per_Plot', 'Biomass_Yield', 'Plant_Height_cm', 'SPAD']:
                                if yc not in data:
                                    data[yc] = v
                                    break
                            value_idx += 1
                            if value_idx >= 3:
                                break
                    if len(data) > 1:
                        rows.append(data)

        return rows

    def _parse_block(self, treatment, values):
        data = {'Treatment': treatment}
        for v in values:
            v = v.strip()
            if not v or v in ('-', '–', '—', ''):
                continue
            parts = re.split(r'\s{2,}|\t', v)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                eq = re.match(r'(\w[\w\s]*?)\s*[=:]\s*([\d.]+)', p)
                if eq:
                    key = norm(eq.group(1))
                    if key in HEADER_MAP:
                        val = parse_value(eq.group(2))
                        if val is not None:
                            data[HEADER_MAP[key]] = val
        return data if len(data) > 1 else None

    def _build_tables(self, rows):
        if not rows:
            return []
        import pandas as pd
        return [pd.DataFrame(rows)]

    def extract_metadata_regex(self, text):
        meta = {}
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', text, re.I)
        if doi_match:
            meta['DOI'] = doi_match.group(1).rstrip('.')
        year_match = re.search(r'\b(19|20)\d{2}\b', text[:500])
        if year_match:
            meta['Year'] = int(year_match.group(0))
        crop_match = re.search(
            r'\b(wheat|rice|maize|corn|sorghum|millet|barley|oat|soybean|cowpea|pigeonpea|'
            r'chickpea|lentil|beans|peas|groundnut|peanut|sunflower|mustard|cotton|sugarcane|'
            r'potato|tomato|pepper|chilli|onion|garlic|carrot|spinach|cabbage|cauliflower|'
            r'brinjal|cucumber|pumpkin|watermelon|muskmelon|banana|mango|orange|grape|apple)',
            text[:1500], re.I)
        if crop_match:
            meta['Crop'] = crop_match.group(0).title()
        return meta
