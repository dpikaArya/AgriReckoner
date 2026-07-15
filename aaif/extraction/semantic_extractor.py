import os, re, json
from . import norm, parse_value, is_treatment, YIELD_COLS
from .pdfminer_reader import HEADER_MAP

YIELD_PAT = re.compile(
    r'(?:yield|grain\s*yield|biomass)[\s:.]*?(\d+[.\d]*)',
    re.I
)

TRANSFORMER_OK = True

CROP_KEYWORDS = [
    'wheat', 'rice', 'maize', 'corn', 'sorghum', 'millet', 'barley', 'oat',
    'soybean', 'cowpea', 'common bean', 'chickpea', 'lentil', 'peas', 'pigeonpea',
    'groundnut', 'peanut', 'sunflower', 'mustard', 'rapeseed', 'sesame',
    'cotton', 'sugarcane', 'potato', 'tomato', 'bell pepper', 'chilli', 'capsicum',
    'onion', 'garlic', 'carrot', 'spinach', 'cabbage', 'cauliflower', 'broccoli',
    'brinjal', 'eggplant', 'cucumber', 'pumpkin', 'watermelon', 'muskmelon',
    'banana', 'mango', 'orange', 'grape', 'apple', 'strawberry',
]
LOCATION_PAT = re.compile(
    r'(?:at|in|near|location|site|field)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    re.I
)
DESIGN_PAT = re.compile(
    r'(?:RCBD|randomized\s+complete\s+block|split\s+plot|CRD|completely\s+randomized'
    r'|factorial|latin\s+square|strip\s+plot|alpha\s+lattice)',
    re.I
)
TREATMENT_DESC_PAT = re.compile(
    r'(T\d+|F[SOB]{0,3}|[Ww]\d+)\s*[=:]\s*([^,;.]+)',
    re.I
)
STATS_PAT = re.compile(
    r'(?:CD\s*\(?0\.05\)?|[Cc]ritical\s+difference|LSD|[Ss]tandard\s+error|'
    r'S\.Em\.?|[Cc]oefficient\s+of\s+variation|CV\s*%)',
    re.I
)
DOSE_PAT = re.compile(
    r'(\d+[.\d]*)\s*(kg|g|mg|t|ml|L)\s*(?:ha|acre|plot|pot)[\s-]*(\d+(?:\.\d+)?)?\s*'
    r'(?:N|P|K|NPK|nitrogen|phosphorus|potash|urea|DAP|MOP|SSP)?',
    re.I
)

class SemanticExtractor:
    def __init__(self):
        self.name = 'semantic'
        self.success = False
        self.confidence = 0.0
        self.rows = []
        self.metadata = {}
        self.model = None

    def extract(self, pdf_path):
        result = {
            'reader': self.name,
            'success': False,
            'confidence': 0.0,
            'rows': [],
            'metadata': {},
        }
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            result['error'] = f'sentence-transformers load failed: {e}'
            return result
            text = self._read_text(pdf_path)
            if not text:
                result['error'] = 'no text extracted'
                return result

            meta = self._extract_semantic(text)
            result['metadata'] = meta

            rows = self._extract_treatments(text, meta)
            result['rows'] = rows

            n_fields = sum(1 for v in meta.values() if v)
            result['confidence'] = min(0.6, 0.1 + n_fields * 0.02 + len(rows) * 0.02)
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        self.metadata = result['metadata']
        return result

    def _read_text(self, pdf_path):
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams
            la = LAParams(detect_vertical=True, all_texts=True)
            return extract_text(pdf_path, laparams=la)
        except Exception:
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    return '\n'.join(p.extract_text() or '' for p in pdf.pages)
            except Exception:
                return None

    def _extract_semantic(self, text):
        meta = {
            'Crop': None, 'Variety': None, 'Location': None,
            'Season': None, 'Design': None, 'Replications': None,
            'Dose': None, 'Statistical_Methods': None,
            'Rainfall': None, 'Temperature_C': None,
        }
        text_lower = text.lower()

        for crop in sorted(CROP_KEYWORDS, key=len, reverse=True):
            pat = re.compile(r'\b' + re.escape(crop) + r'\b', re.I)
            if pat.search(text_lower[:2000]):
                meta['Crop'] = crop.title()
                break

        loc_match = LOCATION_PAT.search(text)
        if loc_match:
            meta['Location'] = loc_match.group(1)

        des_match = DESIGN_PAT.search(text)
        if des_match:
            meta['Design'] = des_match.group(0)

        rep_match = re.search(r'(\d+)\s*replications?\s', text_lower[:3000])
        if rep_match:
            meta['Replications'] = int(rep_match.group(1))

        dose_match = DOSE_PAT.search(text)
        if dose_match:
            meta['Dose'] = dose_match.group(0)

        stats_match = STATS_PAT.search(text)
        if stats_match:
            meta['Statistical_Methods'] = stats_match.group(0)

        season_match = re.search(
            r'\b(kharif|rabi|zaid|summer|winter|spring|rainy|dry|monsoon)\b',
            text_lower[:3000], re.I)
        if season_match:
            meta['Season'] = season_match.group(0).title()

        rainfall_match = re.search(
            r'(\d+[.\d]*)\s*(?:mm|cm)\s*(?:rainfall|rain|precipitation)',
            text_lower, re.I)
        if rainfall_match:
            val = parse_value(rainfall_match.group(1))
            if val is not None:
                meta['Rainfall'] = val

        temp_match = re.search(
            r'(\d+[.\d]*)\s*[°°]?\s*[Cc]\s*(?:temperature|temp)',
            text_lower, re.I)
        if temp_match:
            val = parse_value(temp_match.group(1))
            if val is not None:
                meta['Temperature_C'] = val

        return meta

    def _extract_treatments(self, text, meta):
        rows = []

        for match in TREATMENT_DESC_PAT.finditer(text):
            tid = match.group(1).upper()
            if not is_treatment(tid.rstrip('*†‡').strip()):
                continue
            desc = match.group(2).strip()
            data = {'Treatment': tid}
            if meta.get('Crop'):
                data['Crop'] = meta['Crop']
            if meta.get('Location'):
                data['Location'] = meta['Location']
            if meta.get('Design'):
                data['Design'] = meta['Design']
            rows.append(data)

        yield_matches = YIELD_PAT.findall(text)
        for i, row in enumerate(rows):
            if i < len(yield_matches):
                pv = parse_value(yield_matches[i])
                if pv is not None:
                    row['Yield_per_Plot'] = pv

        return rows

    def get_embeddings(self, texts):
        if not self.model:
            return None
        return self.model.encode(texts, convert_to_tensor=True)
