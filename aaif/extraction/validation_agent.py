import os, re, json
import pandas as pd
from collections import defaultdict
from . import norm, parse_value, is_treatment, UAMS_COLUMNS, YIELD_COLS, TREATMENT_RE
from .pdfminer_reader import HEADER_MAP

UNIT_CONVERSIONS = {
    't_per_ha': {'to': 'kg_per_ha', 'factor': 1000},
    'q_per_ha': {'to': 'kg_per_ha', 'factor': 100},
    'kg_per_acre': {'to': 'kg_per_ha', 'factor': 2.471},
    'g_per_plot': {'to': 'kg_per_ha', 'factor': 0.001},
}

UNIT_PAT = re.compile(r'\(([^)]*)\)')
TID_CLEAN = re.compile(r'[†‡*]')

class ValidationAgent:
    def __init__(self):
        self.name = 'validation'
        self.success = False
        self.confidence = 0.0
        self.rows = []
        self.report = {}

    def merge(self, stage_results, source_file=None):
        result = {
            'success': True,
            'confidence': 0.0,
            'rows': [],
            'report': {},
        }

        all_rows = []
        for sr in stage_results:
            if sr.get('success') and sr.get('rows'):
                for r in sr['rows']:
                    r['_reader'] = sr.get('reader', 'unknown')
                    r['_confidence'] = sr.get('confidence', 0.0)
                    all_rows.append(r)

        if not all_rows:
            if source_file:
                result['rows'] = self._fallback_metadata(source_file)
            result['success'] = bool(result['rows'])
            result['confidence'] = 0.1
            self.rows = result['rows']
            self.report = result['report']
            return result

        df = pd.DataFrame(all_rows)
        merged = self._merge_treatments(df)
        result['rows'] = merged
        result['confidence'] = min(0.9, sum(sr.get('confidence', 0) for sr in stage_results if sr.get('success')) / max(1, sum(1 for sr in stage_results if sr.get('success'))))
        result['report'] = self._build_report(stage_results, len(merged))
        result['success'] = True

        self.rows = result['rows']
        self.report = result['report']
        self.confidence = result['confidence']
        return result

    def _merge_treatments(self, df):
        grouped = defaultdict(dict)
        readers = defaultdict(dict)

        for _, row in df.iterrows():
            key = (row.get('Source_File', ''), row.get('Treatment', ''))
            conf = row.get('_confidence', 0.0)

            for col in df.columns:
                if col in ('_reader', '_confidence', '_source_page', 'Source_File'):
                    continue
                v = row.get(col)
                if pd.notna(v) and v is not None:
                    if col not in grouped[key] or conf > readers[key].get(col, 0):
                        grouped[key][col] = v
                        readers[key][col] = conf
                    elif col in YIELD_COLS or col in ('Crop', 'Location', 'Design'):
                        grouped[key][col] = v

            readers[key]['_max_conf'] = max(readers[key].get('_max_conf', 0), conf)

        merged_rows = []
        for (src, tid), vals in grouped.items():
            row = {'Source_File': src, 'Treatment': tid}
            row.update(vals)
            merged_rows.append(row)

        for r in merged_rows:
            tid = r.get('Treatment', '')
            if tid:
                r['Treatment'] = TID_CLEAN.sub('', tid).strip()

        return merged_rows

    def normalize_units(self, rows):
        for row in rows:
            for yc in YIELD_COLS:
                v = row.get(yc)
                if v is None or not isinstance(v, (int, float)):
                    continue
                if yc == 'Yield_per_Hectare':
                    if v > 50000:
                        row[yc] = v / 1000
                elif yc == 'Yield_per_Plot' and v < 1:
                    row[yc] = v * 1000
        return rows

    def _fallback_metadata(self, source_file):
        pdf_name = os.path.basename(source_file)
        meta_row = {'Source_File': pdf_name, 'Treatment': 'UNKNOWN'}
        try:
            from .pdfminer_reader import PdfminerReader
            reader = PdfminerReader()
            res = reader.extract(source_file)
            if res.get('metadata'):
                m = res['metadata']
                if res.get('metadata', {}).get('title'):
                    meta_row['Paper_ID'] = m['title'][:100]
        except Exception:
            pass
        return [meta_row]

    def _build_report(self, stage_results, n_merged):
        report = {
            'stages_run': len(stage_results),
            'stages_success': sum(1 for s in stage_results if s.get('success')),
            'total_rows_raw': sum(len(s.get('rows', [])) for s in stage_results),
            'rows_after_merge': n_merged,
            'stage_details': {},
        }
        for sr in stage_results:
            report['stage_details'][sr.get('reader', '?')] = {
                'success': sr.get('success', False),
                'confidence': sr.get('confidence', 0),
                'rows': len(sr.get('rows', [])),
                'error': sr.get('error'),
            }
        return report

    def fill_uas_columns(self, rows):
        for row in rows:
            for col in UAMS_COLUMNS:
                if col not in row:
                    row[col] = None
        return rows

    def to_dataframe(self, rows):
        return pd.DataFrame(rows)

    def map_to_uas(self, df, output_path):
        df = df.copy()
        for col in UAMS_COLUMNS:
            if col not in df.columns:
                df[col] = None

        col_order = [c for c in UAMS_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in UAMS_COLUMNS and c not in ('_reader', '_confidence', '_source_page')]
        df = df[col_order + extra]

        try:
            if os.path.exists(output_path):
                existing = pd.read_excel(output_path)
                merged = pd.concat([existing, df], ignore_index=True)
                merged = merged.drop_duplicates(subset=['Source_File', 'Treatment'], keep='first')
                merged.to_excel(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
        except Exception:
            df.to_excel(output_path, index=False)

        return df

    def validate_duplicates(self, rows):
        if not rows:
            return rows
        seen = {}
        duplicates = []
        clean_rows = []
        for r in rows:
            key = (r.get('Source_File', ''), r.get('Treatment', ''))
            if key in seen:
                prev = seen[key]
                n_new = sum(1 for k, v in r.items() if v is not None and k not in ('Treatment', 'Source_File', '_reader', '_confidence'))
                n_prev = sum(1 for k, v in prev.items() if v is not None and k not in ('Treatment', 'Source_File', '_reader', '_confidence'))
                if n_new > n_prev:
                    seen[key] = r
                    duplicates.append(key)
                else:
                    duplicates.append(key)
            else:
                seen[key] = r
        clean_rows = list(seen.values())
        if duplicates:
            self.report['duplicate_treatments'] = len(duplicates)
        return clean_rows

    def validate_crop_consistency(self, rows):
        if not rows:
            return rows
        paper_crops = defaultdict(set)
        for r in rows:
            src = r.get('Source_File', '')
            crop = r.get('Crop', '')
            if src and crop:
                paper_crops[src].add(crop)
        inconsistent = {src: crops for src, crops in paper_crops.items() if len(crops) > 1}
        if inconsistent:
            for src, crops in inconsistent.items():
                majority_crop = max(crops, key=lambda c: sum(1 for r in rows if r.get('Source_File') == src and r.get('Crop') == c))
                for r in rows:
                    if r.get('Source_File') == src and r.get('Crop') in crops - {majority_crop}:
                        r['Crop'] = majority_crop
                        r['_crop_corrected'] = True
            self.report['crop_inconsistencies_fixed'] = len(inconsistent)
        return rows

    def validate_treatment_ids(self, rows):
        invalid = []
        for r in rows:
            tid = r.get('Treatment', '')
            if tid and not is_treatment(tid):
                invalid.append(tid)
        if invalid:
            self.report['invalid_treatment_ids'] = invalid
        return rows
