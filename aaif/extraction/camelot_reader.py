import os, re
from . import norm, parse_value, is_treatment
from .pdfminer_reader import HEADER_MAP

CAMELOT_OK = False
try:
    import camelot
    CAMELOT_OK = True
except ImportError:
    pass

UNIT_PAT = re.compile(r'\([^)]*\)')

class CamelotReader:
    def __init__(self):
        self.name = 'camelot'
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
        }
        if not CAMELOT_OK:
            result['error'] = 'camelot-py not installed'
            return result

        try:
            all_tables = []

            for flavor in ['lattice', 'stream']:
                try:
                    tables = camelot.read_pdf(pdf_path, flavor=flavor, pages='all')
                    if tables and len(tables) > 0:
                        all_tables.extend([t for t in tables])
                except Exception:
                    continue

            if not all_tables:
                result['error'] = 'no tables detected'
                return result

            dfs = []
            for table in all_tables:
                df = table.df
                dfs.append(df)
                rows = self._parse_table(df)
                result['rows'].extend(rows)

            result['tables'] = dfs
            result['confidence'] = min(0.8, 0.2 + len(result['rows']) * 0.02)
            result['success'] = len(result['rows']) > 0

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        self.tables = result['tables']
        return result

    def _parse_table(self, df):
        rows = []
        if df.empty:
            return rows

        cols = df.columns.tolist()
        treat_col = -1
        col_map = {}

        for ci in range(len(cols)):
            for ri in range(min(3, len(df))):
                val = str(df.iloc[ri, ci]).strip()
                if is_treatment(val.rstrip('*†‡').strip()):
                    treat_col = ci
                    break
            if treat_col >= 0:
                break

        if treat_col < 0:
            header_row = 0
            for ci in range(len(cols)):
                val = str(df.iloc[0, ci]).strip().lower()
                if val in ('treatment', 'treatments', 'trt', 'treatment code'):
                    treat_col = ci
                    break
            if treat_col < 0:
                return rows
        else:
            header_row = 0
            for ri in range(min(3, len(df))):
                vals = [str(df.iloc[ri, ci]).strip().lower() for ci in range(len(cols))]
                if any(v in ('treatment', 'treatments', 'trt') for v in vals):
                    header_row = ri
                    break
                if not all(is_treatment(str(df.iloc[ri, ci]).strip().rstrip('*†‡').strip())
                           if ci == treat_col else True for ci in range(len(cols))):
                    header_row = ri
                    break

        for ci in range(len(cols)):
            if ci == treat_col:
                col_map[ci] = 'Treatment'
                continue
            raw = str(df.iloc[header_row, ci]).strip() if header_row < len(df) else ''
            c = norm(UNIT_PAT.sub('', raw))
            if not c or c in ('-', ''):
                continue
            if c in HEADER_MAP:
                col_map[ci] = HEADER_MAP[c]
            else:
                for key, val in sorted(HEADER_MAP.items(), key=lambda x: -len(x[0])):
                    if key in c:
                        col_map[ci] = val
                        break

        for ri in range(header_row + 1, len(df)):
            row_vals = [str(df.iloc[ri, ci]).strip() for ci in range(len(cols))]
            if not any(row_vals):
                continue
            t_val = row_vals[treat_col].rstrip('*†‡').strip() if treat_col < len(row_vals) else ''
            if not t_val or not is_treatment(t_val):
                continue
            data = {'Treatment': t_val}
            for ci, varname in col_map.items():
                if ci < len(row_vals) and ci != treat_col and varname:
                    v = parse_value(row_vals[ci])
                    if v is not None:
                        data[varname] = v
            if len(data) > 1:
                rows.append(data)

        return rows
