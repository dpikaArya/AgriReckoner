import os, re, io, tempfile
from . import norm, parse_value, is_treatment, YIELD_COLS
from .pdfminer_reader import HEADER_MAP

OCR_OK = False
PDF2IMAGE_OK = False
try:
    import pytesseract
    OCR_OK = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    try:
        from PIL import Image
        PDF2IMAGE_OK = False
    except ImportError:
        pass

TABLE_PAT = re.compile(
    r'(T\d+|T\d+[A-Z]?)[\s,;:]+(\d+[.\d]*)[\s,;:]+'
    r'(\d+[.\d]*)?[\s,;:]*(\d+[.\d]*)?[\s,;:]*(\d+[.\d]*)?',
    re.I
)
YIELD_PAT = re.compile(
    r'(?:yield|grain\s*yield|biomass)[\s:.]*?(\d+[.\d]*)',
    re.I
)

class OcrReader:
    def __init__(self):
        self.name = 'ocr'
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
        if not OCR_OK:
            result['error'] = 'pytesseract not installed'
            return result

        try:
            images = self._pdf_to_images(pdf_path)
            if not images:
                result['error'] = 'could not convert PDF to images'
                return result

            all_text = []
            for img in images:
                text = pytesseract.image_to_string(img, config='--psm 6')
                all_text.append(text)

            combined = '\n'.join(all_text)
            result['metadata']['ocr_text'] = combined[:5000]

            rows = self._extract_rows_from_ocr(combined)
            result['rows'] = rows
            result['confidence'] = min(0.5, 0.1 + len(rows) * 0.03)
            result['success'] = len(rows) > 0

        except Exception as e:
            result['error'] = str(e)

        self.success = result['success']
        self.confidence = result['confidence']
        self.rows = result['rows']
        return result

    def _pdf_to_images(self, pdf_path):
        if PDF2IMAGE_OK:
            try:
                return convert_from_path(pdf_path, dpi=300, first_page=1, last_page=10)
            except Exception:
                pass
        try:
            from .poppler_reader import _find_poppler_bin
            pdftoppm = _find_poppler_bin('pdftoppm')
            if pdftoppm:
                import subprocess
                with tempfile.TemporaryDirectory() as tmpdir:
                    prefix = os.path.join(tmpdir, 'page')
                    subprocess.run(
                        [pdftoppm, '-r', '300', '-png', pdf_path, prefix],
                        capture_output=True, timeout=60,
                    )
                    images = []
                    for f in sorted(os.listdir(tmpdir)):
                        if f.endswith('.png'):
                            from PIL import Image
                            images.append(Image.open(os.path.join(tmpdir, f)).convert('RGB'))
                    if images:
                        return images
        except Exception:
            pass
        try:
            from PIL import Image
            imgs = []
            try:
                img = Image.open(pdf_path)
                imgs.append(img.convert('RGB'))
            except Exception:
                pass
            return imgs
        except Exception:
            return None

    def _extract_rows_from_ocr(self, text):
        rows = []
        seen = set()

        for match in TABLE_PAT.finditer(text):
            tid = match.group(1).upper()
            if not is_treatment(tid.rstrip('*†‡').strip()):
                continue
            if tid in seen:
                continue
            seen.add(tid)
            vals = [g for g in match.groups()[1:] if g]
            data = {'Treatment': tid}
            for v in vals:
                pv = parse_value(v)
                if pv is not None:
                    for yc in YIELD_COLS:
                        if yc not in data:
                            data[yc] = pv
                            break
            if len(data) > 1:
                rows.append(data)

        return rows
