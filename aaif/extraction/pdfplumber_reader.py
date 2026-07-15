import os, re
from . import norm, parse_value, is_treatment, UAMS_COLUMNS, YIELD_COLS
from .pdfminer_reader import HEADER_MAP

PDFPLUMBER_OK = False
try:
    import pdfplumber
    PDFPLUMBER_OK = True
except ImportError:
    pass

UNIT_PAT = re.compile(r'\([^)]*\)')

class PdfplumberReader:
    def __init__(self):
        self.name = 'pdfplumber'
        self.success = False
        self.confidence = 0.0
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
        if not PDFPLUMBER_OK:
            result['error'] = 'pdfplumber not installed'
            return result

        try:
            source_file = os.path.basename(pdf_path)
            with pdfplumber.open(pdf_path) as pdf:
                for pi, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        if len(table) < 3:
                            continue
                        parsed = self._parse_table(table, pi + 1, source_file)
                        result['rows'].extend(parsed['rows'])
                        if parsed['df'] is not None:
                            result['tables'].append(parsed['df'])

            result['confidence'] = min(0.75, 0.1 + len(result['rows']) * 0.02)
            result['success'] = len(result['rows']) > 0

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        self.tables = result['tables']
        return result

    def _parse_table(self, table, page_num, source_file):
        rows = []
        df = None
        treat_col = -1
        header_row = 0

        for ri in range(min(3, len(table))):
            row = table[ri]
            if not row:
                continue
            for ci, cell in enumerate(row):
                val = str(cell).strip().lower() if cell else ''
                if val in ('treatment', 'treatments', 'trt', 'treatment code'):
                    treat_col = ci
                    header_row = ri
                    break
                raw = str(cell).strip().rstrip('*†‡').strip() if cell else ''
                if is_treatment(raw):
                    treat_col = ci
                    header_row = ri if ri > 0 else 0
                    break
            if treat_col >= 0:
                break

        if treat_col < 0:
            return {'rows': [], 'df': None}

        col_map = {}
        hdr = table[header_row] if header_row < len(table) else table[0]
        for ci, cell in enumerate(hdr):
            if ci == treat_col:
                col_map[ci] = 'Treatment'
                continue
            raw = str(cell).strip() if cell else ''
            if raw == '-' or not raw:
                continue
            clean = UNIT_PAT.sub('', norm(raw)).strip()
            if not clean or clean == '-':
                continue
            if clean in HEADER_MAP:
                col_map[ci] = HEADER_MAP[clean]
            else:
                for key, val in sorted(HEADER_MAP.items(), key=lambda x: -len(x[0])):
                    if key in clean:
                        col_map[ci] = val
                        break

        for ri in range(header_row + 1, len(table)):
            row = table[ri]
            if not row or treat_col >= len(row):
                continue
            t_val = str(row[treat_col]).strip().rstrip('*†‡').strip() if row[treat_col] else ''
            if not t_val or not is_treatment(t_val):
                continue
            data = {'Treatment': t_val}
            for ci, varname in col_map.items():
                if ci < len(row) and ci != treat_col and varname:
                    v = parse_value(row[ci])
                    if v is not None:
                        if varname in data:
                            pass
                        else:
                            data[varname] = v
            if len(data) > 1:
                data['_source_page'] = page_num
                rows.append(data)

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)

        return {'rows': rows, 'df': df}
