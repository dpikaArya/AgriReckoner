"""
Enhanced treatment-level extraction from agricultural PDFs.
Handles multiple treatment ID patterns and column-major table formats.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agri_ai_agent.config.schema import UAMS_COLUMNS

PDFMINER_AVAILABLE = False
try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

    PDFMINER_AVAILABLE = True
except ImportError:
    import PyPDF2


def extract_pdf_text(pdf_path):
    if PDFMINER_AVAILABLE:
        la_params = LAParams(detect_vertical=True, all_texts=True)
        text = extract_text(pdf_path, laparams=la_params)
        return text.split("\n"), text
    else:
        reader = PyPDF2.PdfReader(pdf_path)
        lines = []
        for pg in reader.pages:
            t = pg.extract_text()
            if t:
                lines.extend(t.split("\n"))
        return lines, "\n".join(lines)


CROP_MAP = {
    "bell pepper": "Bell Pepper",
    "capsicum annuum": "Bell Pepper",
    "black wheat": "Black Wheat",
    "triticum": "Black Wheat",
    "carrot": "Carrot",
    "daucus carota": "Carrot",
    "cowpea": "Cowpea",
    "vigna unguiculata": "Cowpea",
    "spinach": "Spinach",
    "spinacia oleracea": "Spinach",
    "chickpea": "Chickpea",
    "cicer arietinum": "Chickpea",
    "barley": "Barley",
    "hordeum": "Barley",
    "maize": "Maize",
    "zea mays": "Maize",
}
DESIGN_MAP = {
    "randomized block design": "RBD",
    "randomized complete block design": "RCBD",
    "split plot": "Split Plot",
    "completely randomized design": "CRD",
}

# Treatment ID patterns: (regex, sort_key_func)
TREATMENT_PATS = [
    (re.compile(r"^(T\d+)$"), lambda x: int(x[1:])),  # T0-T9
    (re.compile(r"^(F[SOB]{0,3})$"), lambda x: x),  # F, FS, FO, FB, FSOB
    (re.compile(r"^(SRF|CRF|SF|HAF|ZSF|SWF)$"), lambda x: x),  # Fertilizer codes
    (re.compile(r"^(W\d+)$"), lambda x: x),  # W0, W1, W2
    (re.compile(r"^(Se\d+)$"), lambda x: x),  # Se0, Se2160
]

HEADER_MAP = [
    (
        "Yield_per_Plot",
        [
            r"yield.*plot",
            r"plot.*yield",
            r"total yield",
            r"fruit yield",
            r"fresh weight",
            r"^yield$",
            r"plot.*weight",
        ],
    ),
    ("Yield_per_Hectare", [r"yield.*ha\b", r"yield.*hectare", r"grain yield", r"seed yield"]),
    ("Yield_per_Acre", [r"yield.*acre", r"q per acre", r"quintal"]),
    (
        "Biomass_Yield",
        [r"biomass yield", r"biological yield", r"straw yield", r"dry yield", r"total biomass"],
    ),
    ("Harvest_Index", [r"harvest", r"\bhi\b"]),
    ("Plant_Height_cm", [r"plant height", r"plant length", r"^height$", r"shoot length"]),
    ("Shoot_Length_cm", [r"shoot.*length", r"shoot.*height"]),
    ("Root_Length_cm", [r"root.*length", r"root.*height"]),
    ("Shoot_Biomass_g", [r"shoot.*biomass", r"shoot.*weight"]),
    ("Root_Biomass_g", [r"root.*biomass", r"root.*weight"]),
    ("Leaf_Area_cm2", [r"leaf area", r"leaf.*cm2"]),
    ("Leaf_Number", [r"leaf.*number", r"no.*leaf", r"leaves plant"]),
    ("Tillers", [r"tiller"]),
    ("Branches", [r"branch"]),
    ("Flowers", [r"flower"]),
    ("Nodes", [r"node"]),
    ("SPAD", [r"spad", r"chlorophyll"]),
    ("Moisture_Content", [r"moisture"]),
    ("Dry_Matter", [r"dry matter"]),
    ("Stem_Diameter_mm", [r"stem diameter"]),
    ("Spike_Length", [r"spike length", r"panicle length"]),
    ("Seeds_per_Spike", [r"grain.*ear", r"grain.*spike", r"seed.*spike", r"no.*seed"]),
    ("100_Seed_Weight", [r"1000.*grain", r"thousand.*grain", r"100.*seed", r"test weight"]),
    ("Fruit_Weight", [r"fruit weight", r"fruit wt"]),
    ("Fruit_Diameter_mm", [r"fruit.*diameter", r"fruit.*dia"]),
    ("Fruit_Number", [r"fruit.*number", r"no.*fruit", r"fruit count"]),
    ("Root_Diameter_mm", [r"root.*diameter"]),
    ("Root_Weight", [r"root.*weight"]),
    ("Pod_Weight", [r"pod weight"]),
    ("Soil_pH", [r"\bph\b", r"soil ph"]),
    ("EC", [r"\bec\b", r"electrical conduct"]),
    ("Organic_Carbon", [r"organic carbon", r"\boc\b"]),
    ("Organic_Matter", [r"organic matter", r"\bom\b"]),
    ("Nitrogen", [r"available n", r"\bnitrogen\b", r"total n\b"]),
    ("Phosphorus", [r"available p", r"phosphorus", r"total p\b"]),
    ("Potassium", [r"available k", r"potassium", r"total k\b"]),
    ("Zinc", [r"\bzn\b", r"zinc"]),
    ("Iron", [r"\bfe\b", r"iron"]),
    ("Copper", [r"\bcu\b", r"copper"]),
    ("Manganese", [r"\bmn\b", r"manganese"]),
    ("Sulphur", [r"sulphur", r"sulfur"]),
    ("Protein", [r"protein"]),
    ("Ash", [r"\bash\b"]),
    ("Fiber", [r"fiber", r"fibre"]),
    ("Carbohydrates", [r"carbohydrate"]),
    ("Fat", [r"\bfat\b", r"\boil\b"]),
    ("Average_Temperature", [r"average.*temp", r"mean.*temp"]),
    ("Temperature_Max", [r"tmax", r"temp.*max", r"max.*temp"]),
    ("Temperature_Min", [r"tmin", r"temp.*min", r"min.*temp"]),
    ("Rainfall", [r"rainfall", r"precipitation"]),
    ("Humidity", [r"humidity"]),
]


def match_header(h):
    h = h.lower().strip()
    for var, pats in HEADER_MAP:
        for p in pats:
            if re.search(p, h):
                return var
    return ""


def parse_value(v):
    if not v or not isinstance(v, str):
        return None
    c = re.sub(r"[\*†‡abcdABCD\*\*†‡]", "", v).strip()
    m = re.match(r"([\d.]+)\s*[±±]\s*[\d.]+", c)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    if re.match(r"^-?\d+\.?\d*$", c):
        try:
            return float(c)
        except Exception:
            return None
    return None


def extract_context(lines, full_text):
    m = {}
    doi_m = re.search(r"(10\.\d{4,}/[^\s]+)", full_text)
    m["DOI"] = doi_m.group(1).rstrip(".") if doi_m else ""
    for kw, cr in CROP_MAP.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", full_text, re.IGNORECASE):
            m["Crop"] = cr
            break
    if "Crop" not in m:
        for kw, cr in CROP_MAP.items():
            if re.search(kw, full_text, re.IGNORECASE):
                m["Crop"] = cr
                break
    yrs = re.findall(r"\b(19\d{2}|20\d{2})\b", full_text)
    if yrs:
        m["Year"] = max(set(yrs), key=yrs.count)
    for kw, vl in DESIGN_MAP.items():
        if re.search(kw, full_text, re.IGNORECASE):
            m["Design"] = vl
            break
    for pat, key in [
        (r"(?:tmax|max\s*temp)[\s:=.]*?(\d+\.?\d*)", "Temperature_Max"),
        (r"(?:tmin|min\s*temp)[\s:=.]*?(\d+\.?\d*)", "Temperature_Min"),
        (r"(?:rainfall|precip)[^:.]*?(\d+\.?\d*)\s*(?:mm|cm)", "Rainfall"),
    ]:
        mm = re.search(pat, full_text, re.IGNORECASE)
        if mm:
            m[key] = float(mm.group(1))
    return m


YIELD_UNIT = r"(kg\s*ha-?1|t\s*ha-?1|kg\s*/\s*ha|t\s*/\s*ha|q\s*/\s*ha|q\s*ha-?1)"

# A yield is only usable if its unit is stated: "2.5" alone could be t/ha, q/ha or a
# percentage increase, and guessing is a 100-1000x error in the target variable.
YIELD_PATTERNS = [
    rf"(?:grain|seed|biological)\s*yield\s*(?:of|was|is|:|=)\s*(\d+\.?\d*)\s*{YIELD_UNIT}",
    rf"(?:recorded|observed|maximum|highest|average)\s*(?:grain|seed)?\s*yield"
    rf"\s*(?:of|was|is)?\s*(\d+\.?\d*)\s*{YIELD_UNIT}",
    rf"yield\s*(?:was|ranged|varied)\s*(?:from)?\s*(\d+\.?\d*)\s*{YIELD_UNIT}",
    rf"(\d+\.?\d*)\s*{YIELD_UNIT}",
]


def yield_regex_extraction(full_text):
    """Extract a single yield from prose when table parsing fails, converted to kg/ha.

    Every pattern requires an explicit unit, which is captured and converted rather than
    assumed. A match whose unit cannot be converted is skipped instead of being stored
    under the wrong unit.
    """
    from agri_ai_agent.extractors.units import convert, normalize_unit

    for pat in YIELD_PATTERNS:
        for m in re.finditer(pat, full_text, re.IGNORECASE):
            unit = normalize_unit(m.group(2))
            if unit is None:
                continue
            value, ok, _ = convert(float(m.group(1)), unit, "kg/ha")
            if ok:
                return {"Yield_per_Hectare": value}
    return {}


def scan_tables(lines, tids, tid_pos_set, ctx):
    """Scan for column-major tables given a treatment ID list."""
    rows = []
    all_entries = []
    for tid in tids:
        for pos in tid_pos_set.get(tid, []):
            all_entries.append((pos, tid))
    all_entries.sort(key=lambda x: x[0])

    i = 0
    while i < len(all_entries):
        block = []
        ei = 0
        j = i
        while j < len(all_entries) and ei < len(tids):
            pos, tid = all_entries[j]
            if tid == tids[ei]:
                block.append((pos, tid))
                ei += 1
                j += 1
            elif tid in tid_pos_set:
                ci = tids.index(tid)
                if ci >= ei:
                    break
                else:
                    j += 1
            else:
                break

        if len(block) >= max(3, len(tids) * 0.7):
            # Check if roughly consecutive (gaps <= 3)
            consec = all(block[k][0] - block[k - 1][0] <= 3 for k in range(1, len(block)))
            if consec:
                # Scan for data columns after block
                pos = block[-1][0] + 1
                columns = []
                empty_streak = 0
                while pos < min(block[-1][0] + 180, len(lines)):
                    line = lines[pos].strip()
                    pos += 1
                    if not line:
                        empty_streak += 1
                        if empty_streak >= 5:
                            break
                        continue
                    empty_streak = 0
                    if line.startswith("Figure") or line.startswith("Fig."):
                        break
                    if re.match(r"^#", line):
                        continue

                    # Check for new treatment block
                    sl = re.sub(r"[*†‡]+$", "", line).strip()
                    if sl in tid_pos_set:
                        pos -= 1
                        break  # Put it back

                    if re.search(r"[a-zA-Z]{3,}", line):
                        header = line
                        values = []
                        for _ in range(len(tids)):
                            if pos >= len(lines):
                                break
                            nxt = lines[pos].strip()
                            if not nxt:
                                pos += 1
                                continue
                            if nxt.startswith("#"):
                                break
                            if nxt.rstrip("*†‡").strip() in tid_pos_set:
                                break
                            if (
                                re.search(r"[a-zA-Z]{3,}", nxt)
                                and not re.search(r"\d", nxt)
                                and len(nxt) > 3
                            ):
                                break
                            if re.search(r"\d", nxt):
                                values.append(nxt)
                                pos += 1
                            else:
                                break
                        if values:
                            var = match_header(header)
                            if var:
                                columns.append((var, values))
                if columns:
                    for _, tid in block:
                        g_idx = tids.index(tid)
                        row = dict(ctx)
                        row["Treatment"] = tid
                        for var, vals in columns:
                            if g_idx < len(vals):
                                v = parse_value(vals[g_idx])
                                if v is not None:
                                    row[var] = v
                        rows.append(row)
            i = j
        else:
            i += 1
    return rows


def process_one_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    lines, full_text = extract_pdf_text(pdf_path)
    ctx = extract_context(lines, full_text)
    ctx["Source_File"] = filename
    ctx["Paper_ID"] = filename.replace(".pdf", "").replace("-", "_")

    # Index all treatment IDs by pattern
    tid_pos = {}
    for pat, _ in TREATMENT_PATS:
        for i, line in enumerate(lines):
            s = re.sub(r"[*†‡]+$", "", line.strip()).strip()
            m = pat.match(s)
            if m:
                tid = m.group(1)
                if tid not in tid_pos:
                    tid_pos[tid] = []
                tid_pos[tid].append(i)

    all_rows = []

    for pat, sort_key in TREATMENT_PATS:
        matching_tids = [t for t in tid_pos if pat.match(t)]
        if len(matching_tids) >= 3:
            matching_tids.sort(key=sort_key)
            # Check if they appear as a consecutive block somewhere
            all_positions = []
            for t in matching_tids:
                for p in tid_pos.get(t, []):
                    all_positions.append((p, t))
            all_positions.sort(key=lambda x: x[0])

            # Find first block of consecutive IDs
            has_consecutive_block = False
            for k in range(len(all_positions) - len(matching_tids) + 1):
                check = True
                for m in range(1, len(matching_tids)):
                    if all_positions[k + m][1] != matching_tids[m]:
                        check = False
                        break
                if check:
                    has_consecutive_block = True
                    break

            if has_consecutive_block:
                tid_set = {t: tid_pos[t] for t in matching_tids}
                table_rows = scan_tables(lines, matching_tids, tid_set, ctx)
                all_rows.extend(table_rows)

    # If table parsing found nothing, try regex extraction
    if not any(
        row.get("Yield_per_Hectare") or row.get("Yield_per_Plot") or row.get("Biomass_Yield")
        for row in all_rows
    ):
        yield_vals = yield_regex_extraction(full_text)
        if yield_vals:
            row = dict(ctx)
            row.update(yield_vals)
            all_rows.append(row)

    if not all_rows:
        return [ctx]
    return all_rows


def add_missing_yield_regex(all_rows, full_text):
    """Apply regex to fill missing yield for paper-level rows."""
    yield_vals = yield_regex_extraction(full_text)
    for row in all_rows:
        if not (row.get("Yield_per_Hectare") or row.get("Yield_per_Plot")):
            row.update(yield_vals)


def process_all(data_dir):
    all_rows = []
    pdf_dir = Path(data_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Processing {len(pdf_files)} PDFs...")
    for pdf_path in pdf_files:
        print(f"  {pdf_path.name}...", end=" ", flush=True)
        try:
            rr = process_one_pdf(str(pdf_path))
            print(f"{len(rr)} rows")
            all_rows.extend(rr)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    for c in UAMS_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    ordc = [c for c in UAMS_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in UAMS_COLUMNS]
    return df[ordc + extra]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--output", "-o", default="outputs/treatment_level_extraction.xlsx")
    a = ap.parse_args()
    df = process_all(a.data_dir)
    if df.empty:
        print("No data!")
        sys.exit(1)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    np_ = df["Source_File"].nunique() if "Source_File" in df.columns else 0
    ac = sum(1 for c in df.columns if df[c].notna().any())
    print("\n=== EXTRACTION REPORT ===")
    print(f"Saved: {out}")
    print(f"  Papers processed: {np_}")
    print(f"  Treatment rows: {len(df)}")
    print(f"  Active columns: {ac}/{len(df.columns)}")

    for yc in [
        "Yield_per_Hectare",
        "Yield_per_Plot",
        "Yield_per_Acre",
        "Biomass_Yield",
        "Harvest_Index",
    ]:
        if yc in df.columns:
            n = df[yc].notna().sum()
            print(f"  {yc}: {n} non-null ({n / len(df) * 100:.1f}%)")

    print("\n  Missing % for target variables:")
    for var in [
        "Yield_per_Hectare",
        "Yield_per_Plot",
        "Plant_Height_cm",
        "SPAD",
        "Soil_pH",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Organic_Carbon",
        "Protein",
    ]:
        if var in df.columns:
            pct = df[var].isna().sum() / len(df) * 100
            print(f"    {var}: {pct:.1f}%")
