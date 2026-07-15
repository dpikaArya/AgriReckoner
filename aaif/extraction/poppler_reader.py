"""
Poppler Reader — PDF text & table extraction using Poppler utilities.

Uses pdftotext (with -layout mode for table-aware extraction) and pdfinfo
for metadata. Auto-discovers poppler binaries from common install locations.

Part of the AAIF Hybrid Extraction Engine.
"""
import os
import re
import shutil
import subprocess
from typing import Optional

from . import norm, parse_value, is_treatment, UAMS_COLUMNS, YIELD_COLS

SECTION_PATTERNS = {
    'title': re.compile(r'^.{0,3}(?:title|abstract|introduction)', re.I),
    'abstract': re.compile(r'abstract', re.I),
    'materials_methods': re.compile(r'(?:materials?\s*(?:and|&)\s*methods|methodology|experimental\s*setup)', re.I),
    'results': re.compile(r'^results?\s', re.I),
    'discussion': re.compile(r'^discussion\s', re.I),
    'conclusion': re.compile(r'^conclusion', re.I),
    'references': re.compile(r'^references\b', re.I),
}

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
    'seeds per spike': 'Seeds_per_Spike', 'grains per spike': 'Seeds_per_Spike',
    '100 seed weight': '100_Seed_Weight', '1000 grain weight': '100_Seed_Weight',
    'test weight': '100_Seed_Weight',
    'leaf area': 'Leaf_Area_cm2',
    'fruit weight': 'Fruit_Weight', 'fruit diameter': 'Fruit_Diameter_mm',
    'nitrogen': 'Nitrogen', 'phosphorus': 'Phosphorus', 'potassium': 'Potassium',
    'organic carbon': 'Organic_Carbon', 'soil ph': 'Soil_pH',
    'protein': 'Protein', 'protein content': 'Protein',
    'iron': 'Iron_ppm', 'zinc': 'Zinc_ppm',
    'copper': 'Copper', 'calcium': 'Calcium', 'magnesium': 'Magnesium',
    'sodium': 'Sodium',
}

POPPLER_SEARCH_PATHS = [
    r'C:\poppler\poppler-24.08.0\Library\bin',
    r'C:\poppler\Library\bin',
    r'C:\poppler\bin',
    r'C:\Program Files\poppler\Library\bin',
    r'C:\Program Files (x86)\poppler\Library\bin',
    '/usr/local/bin',
    '/usr/bin',
    '/opt/homebrew/bin',
]


def _find_poppler_bin(name: str) -> Optional[str]:
    """Locate a poppler binary by name, checking PATH and common locations."""
    found = shutil.which(name)
    if found:
        return found
    for search_dir in POPPLER_SEARCH_PATHS:
        candidate = os.path.join(search_dir, name)
        if os.path.isfile(candidate):
            return candidate
        # Also check for .exe on Windows
        candidate_exe = os.path.join(search_dir, name + '.exe')
        if os.path.isfile(candidate_exe):
            return candidate_exe
    return None


def _run_poppler_cmd(cmd: list, timeout: int = 30) -> tuple[str, str, int]:
    """Run a poppler command and return (stdout, stderr, returncode)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        stdout = proc.stdout.decode('utf-8', errors='replace') if proc.stdout else ''
        stderr = proc.stderr.decode('utf-8', errors='replace') if proc.stderr else ''
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return '', f'Command timed out after {timeout}s', -1
    except FileNotFoundError:
        return '', f'Binary not found: {cmd[0]}', -1
    except Exception as e:
        return '', str(e), -1


class PopplerReader:
    """PDF extraction using Poppler utilities (pdftotext, pdfinfo)."""

    def __init__(self):
        self.name = 'poppler'
        self.success = False
        self.confidence = 0.0
        self.rows = []
        self.tables = []
        self.metadata = {}
        self._pdftotext_bin = _find_poppler_bin('pdftotext')
        self._pdfinfo_bin = _find_poppler_bin('pdfinfo')

    @property
    def available(self) -> bool:
        return self._pdftotext_bin is not None

    def extract(self, pdf_path: str) -> dict:
        result = {
            'reader': self.name,
            'success': False,
            'confidence': 0.0,
            'rows': [],
            'tables': [],
            'metadata': {},
        }

        if not self.available:
            result['error'] = 'pdftotext not found. Install poppler and add to PATH.'
            return result

        try:
            source_file = os.path.basename(pdf_path)

            # Step 1: Extract full text with -layout (preserves table structure)
            text = self._extract_text_layout(pdf_path)
            if not text or not text.strip():
                result['error'] = 'pdftotext returned empty text'
                return result

            lines = text.split('\n')

            # Step 2: Extract metadata from text
            metadata = self._extract_metadata(lines, source_file)
            result['metadata'] = metadata

            # Step 3: Get pdfinfo metadata if available
            pdfinfo = self._extract_pdfinfo(pdf_path)
            if pdfinfo:
                metadata['pdfinfo'] = pdfinfo
                metadata['page_count'] = pdfinfo.get('Pages', 0)

            # Step 4: Extract tables from layout text
            table_rows = self._extract_tables_from_layout(lines)
            for r in table_rows:
                r['Source_File'] = source_file
            result['rows'] = table_rows

            # Step 5: Build table DataFrames
            if table_rows:
                import pandas as pd
                result['tables'] = [pd.DataFrame(table_rows)]

            n_treatments = len(set(r.get('Treatment', '') for r in table_rows))
            result['confidence'] = min(0.80, 0.25 + n_treatments * 0.06)
            result['success'] = len(table_rows) > 0 or bool(metadata.get('title'))

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        self.tables = result['tables']
        self.metadata = result['metadata']
        return result

    def _extract_text_layout(self, pdf_path: str) -> str:
        """Extract text using pdftotext -layout (preserves table column alignment)."""
        cmd = [self._pdftotext_bin, '-layout', pdf_path, '-']
        stdout, stderr, rc = _run_poppler_cmd(cmd, timeout=30)
        if rc == 0 and stdout and stdout.strip():
            return stdout
        # Fallback: try without -layout
        cmd = [self._pdftotext_bin, pdf_path, '-']
        stdout, stderr, rc = _run_poppler_cmd(cmd, timeout=30)
        return stdout if rc == 0 and stdout else ''

    def _extract_text_raw(self, pdf_path: str) -> str:
        """Extract text using pdftotext -raw (reading order)."""
        cmd = [self._pdftotext_bin, '-raw', pdf_path, '-']
        stdout, stderr, rc = _run_poppler_cmd(cmd, timeout=30)
        return stdout if rc == 0 else ''

    def _extract_pdfinfo(self, pdf_path: str) -> dict:
        """Extract metadata via pdfinfo."""
        if not self._pdfinfo_bin:
            return {}
        cmd = [self._pdfinfo_bin, pdf_path]
        stdout, stderr, rc = _run_poppler_cmd(cmd, timeout=10)
        if rc != 0 or not stdout:
            return {}
        info = {}
        for line in stdout.split('\n'):
            if ':' in line:
                key, _, val = line.partition(':')
                key = key.strip()
                val = val.strip()
                if val.isdigit():
                    val = int(val)
                info[key] = val
        return info

    def _extract_metadata(self, lines: list, source_file: str) -> dict:
        """Extract document metadata from text lines."""
        meta = {'Source_File': source_file, 'full_text': '\n'.join(lines[:2000])}

        # DOI
        full_text = '\n'.join(lines[:200])
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', full_text)
        if doi_match:
            meta['DOI'] = doi_match.group(1).rstrip('.')

        # Year
        year_match = re.search(r'\b(19|20)\d{2}\b', full_text[:500])
        if year_match:
            meta['Year'] = int(year_match.group(0))

        # Crop
        crop_match = re.search(
            r'\b(wheat|rice|maize|corn|sorghum|millet|barley|oat|soybean|cowpea|pigeonpea|'
            r'chickpea|lentil|beans|peas|groundnut|peanut|sunflower|mustard|cotton|sugarcane|'
            r'potato|tomato|pepper|chilli|onion|garlic|carrot|spinach|cabbage|cauliflower|'
            r'brinjal|cucumber|pumpkin|watermelon|muskmelon|banana|mango|orange|grape|apple|'
            r'capsicum|bell\s*pepper|black\s*wheat|cowpea)',
            full_text[:2000], re.I)
        if crop_match:
            meta['Crop'] = crop_match.group(0).title()

        # Title (first substantial line)
        for line in lines[:10]:
            s = line.strip()
            if len(s) > 20 and any(c.isalpha() for c in s):
                meta['title'] = s[:200]
                break

        # Section splitting
        current = None
        buf = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            for key, pat in SECTION_PATTERNS.items():
                if pat.search(s) and len(s) < 200:
                    if current and buf:
                        meta[current] = '\n'.join(buf)
                    current = key
                    buf = [s]
                    break
            else:
                if current:
                    buf.append(s)
        if current and buf:
            meta[current] = '\n'.join(buf)

        return meta

    def _extract_tables_from_layout(self, lines: list) -> list:
        """
        Extract treatment-level table rows from pdftotext -layout output.

        Strategy 1: Row-major — find treatment IDs, parse following key=value lines.
        Strategy 2: Column-major — find treatment ID blocks, scan for header+value columns.
        Strategy 3: Spaced-value — detect column-aligned numeric data blocks.
        """
        rows = []

        # Strategy 1: Row-major parsing (treatment followed by key=value data)
        rows = self._row_major_parse(lines)
        if rows:
            return rows

        # Strategy 2: Column-major parsing (treatment ID columns + header rows)
        rows = self._column_major_parse(lines)
        if rows:
            return rows

        # Strategy 3: Spaced-value parsing (aligned numeric columns)
        rows = self._spaced_value_parse(lines)
        return rows

    def _row_major_parse(self, lines: list) -> list:
        """Find treatment IDs and parse key=value data in following lines."""
        rows = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            tid_val = line.rstrip('*†‡').strip()
            if is_treatment(tid_val):
                block = []
                i += 1
                while i < len(lines):
                    nxt = lines[i].strip()
                    if not nxt:
                        if i + 1 < len(lines) and not lines[i + 1].strip():
                            break
                        i += 1
                        continue
                    if is_treatment(nxt.rstrip('*†‡').strip()):
                        break
                    block.append(nxt)
                    i += 1
                if block:
                    row_data = self._parse_key_value_block(tid_val, block)
                    if row_data:
                        rows.append(row_data)
                continue
            i += 1
        return rows

    def _parse_key_value_block(self, treatment: str, lines: list) -> dict:
        """Parse key=value or 'key  value' pairs from a block of text."""
        data = {'Treatment': treatment}
        for line in lines:
            line = line.strip()
            if not line or line in ('-', '–', '—'):
                continue

            # Pattern: key = value or key : value
            eq_match = re.match(r'([\w\s]+?)\s*[=:]\s*([\d.,]+)', line)
            if eq_match:
                key = norm(eq_match.group(1))
                val = parse_value(eq_match.group(2))
                if val is not None and key in HEADER_MAP:
                    data[HEADER_MAP[key]] = val
                continue

            # Pattern: key  value (tab or multiple spaces separated)
            parts = re.split(r'\s{2,}|\t', line)
            if len(parts) >= 2:
                for p in parts:
                    p = p.strip()
                    key_match = re.match(r'([\w\s]+?)\s+([\d.,]+)', p)
                    if key_match:
                        key = norm(key_match.group(1))
                        val = parse_value(key_match.group(2))
                        if val is not None and key in HEADER_MAP:
                            data[HEADER_MAP[key]] = val

        return data if len(data) > 1 else None

    def _column_major_parse(self, lines: list) -> list:
        """Parse column-major tables where treatment IDs appear as a block."""
        rows = []
        tid_pats = [
            re.compile(r'^(T\d+)$'),
            re.compile(r'^(T\d+[A-Z]?)$'),
            re.compile(r'^([Ww]\d+)$'),
            re.compile(r'^(Se\d+)$'),
            re.compile(r'^(F[SOB]{0,3})$'),
            re.compile(r'^(CRF|SRF|SF|HAF|ZSF|SWF)$'),
        ]

        # Find consecutive treatment ID blocks
        tid_positions = []
        for i, line in enumerate(lines):
            s = line.strip().rstrip('*†‡').strip()
            for pat in tid_pats:
                m = pat.match(s)
                if m:
                    tid_positions.append((i, m.group(1)))
                    break

        if len(tid_positions) < 3:
            return rows

        # Group consecutive positions
        groups = []
        cur = [tid_positions[0]]
        for i in range(1, len(tid_positions)):
            if tid_positions[i][0] - tid_positions[i - 1][0] <= 2:
                cur.append(tid_positions[i])
            else:
                if len(cur) >= 3:
                    groups.append(cur)
                cur = [tid_positions[i]]
        if len(cur) >= 3:
            groups.append(cur)

        for group in groups:
            tids = [t[1] for t in group]
            block_start = group[-1][0] + 1
            block_end = min(block_start + 150, len(lines))

            # Scan for header lines (text with no digits, not treatment IDs)
            header_positions = []
            for i in range(block_start, min(block_start + 30, block_end)):
                s = lines[i].strip()
                if (s and len(s) < 100
                        and not re.match(r'^[\d\s.,%±+-]+$', s)
                        and not is_treatment(s.rstrip('*†‡').strip())):
                    # Check if this line has recognizable variable names
                    s_lower = norm(s)
                    has_var = any(v in s_lower for v in HEADER_MAP.keys())
                    if has_var or len(s.split()) <= 6:
                        header_positions.append((i, s))
                        if len(header_positions) >= 5:
                            break

            if not header_positions:
                continue

            # Map header positions to variable names
            header_map = {}
            for hi, (_, htext) in enumerate(header_positions):
                h_lower = norm(htext)
                if h_lower in HEADER_MAP:
                    header_map[hi] = HEADER_MAP[h_lower]
                else:
                    for key, var in sorted(HEADER_MAP.items(), key=lambda x: -len(x[0])):
                        if key in h_lower:
                            header_map[hi] = var
                            break

            if not header_map:
                continue

            # Extract values: for each treatment, read values under each header column
            for ti, tid in enumerate(tids):
                data = {'Treatment': tid}
                for hi, var in header_positions:
                    if hi not in header_map:
                        continue
                    var = header_map[hi]
                    # In layout mode, values are typically aligned below headers
                    # Look for numeric values in the column range
                    for offset in range(1, len(tids) + 2):
                        val_line_idx = hi + offset * len(tids) + ti
                        if val_line_idx >= block_end:
                            break
                        line = lines[val_line_idx].strip()
                        v = parse_value(line)
                        if v is not None and v > 0:
                            if var not in data:
                                data[var] = v
                            break
                if len(data) > 1:
                    rows.append(data)

        return rows

    def _spaced_value_parse(self, lines: list) -> list:
        """
        Detect aligned numeric data blocks in layout text.
        Looks for lines with multiple numeric values separated by spaces,
        with a header line above them.
        """
        rows = []
        tid_re = re.compile(r'^(T\d+|T\d+[A-Z]?|[Ww]\d+|Se\d+|F[SOB]{0,3})\s')

        for i, line in enumerate(lines):
            if not tid_re.match(line.strip()):
                continue

            # Collect consecutive lines that start with a treatment ID
            block = []
            j = i
            while j < len(lines):
                l = lines[j].strip()
                if not l:
                    j += 1
                    if j < len(lines) and not lines[j].strip():
                        break
                    continue
                if tid_re.match(l) or is_treatment(l.split()[0] if l.split() else ''):
                    # Extract treatment ID and remaining values
                    parts = l.split()
                    if parts:
                        tid = parts[0].rstrip('*†‡').strip()
                        vals = [parse_value(p) for p in parts[1:]]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            block.append((tid, vals))
                elif re.match(r'^[\d\s.,%±]+$', l):
                    # Pure numeric line, might be continuation
                    pass
                else:
                    break
                j += 1

            if len(block) >= 3:
                # Try to assign variable names based on header search above
                header_line = lines[i - 1].strip() if i > 0 else ''
                if header_line and not tid_re.match(header_line):
                    header_parts = re.split(r'\s{2,}|\t', header_line)
                    var_names = []
                    for hp in header_parts:
                        hp_norm = norm(hp)
                        if hp_norm in HEADER_MAP:
                            var_names.append(HEADER_MAP[hp_norm])
                        else:
                            matched = False
                            for key, var in sorted(HEADER_MAP.items(), key=lambda x: -len(x[0])):
                                if key in hp_norm:
                                    var_names.append(var)
                                    matched = True
                                    break
                            if not matched:
                                var_names.append(None)

                    for tid, vals in block:
                        data = {'Treatment': tid}
                        for vi, v in enumerate(vals):
                            if vi < len(var_names) and var_names[vi] and var_names[vi] not in data:
                                data[var_names[vi]] = v
                        if len(data) > 1:
                            rows.append(data)
                else:
                    # No header, just capture values
                    for tid, vals in block:
                        data = {'Treatment': tid}
                        for vi, v in enumerate(vals):
                            for yc in YIELD_COLS + ['Plant_Height_cm', 'SPAD', 'Tillers']:
                                if yc not in data:
                                    data[yc] = v
                                    break
                        if len(data) > 1:
                            rows.append(data)
                break  # Only process first block per treatment sequence

        return rows
