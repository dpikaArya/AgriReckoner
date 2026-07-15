"""
AAIF Complete Pipeline Runner
Executes Phases 1-10 across all PDFs in data/Data ADES/
"""
import io
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "agri_ai_agent"))

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.config.schema import UAMS_COLUMNS
from agri_ai_agent.contracts.messages import AgentContract

settings = AgriAISettings()
settings.ensure_dirs()

PAPERS_DIR = BASE_DIR / "Data ADES"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "database"
FUZZY_DIR = BASE_DIR / "fuzzy_logic"

for d in [OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, DB_DIR, FUZZY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

KNOWN_CROPS = {
    "bell pepper", "capsicum annuum", "capsicum", "black wheat", "triticum aestivum",
    "carrot", "daucus carota", "cowpea", "vigna unguiculata", "spinach", "spinacia oleracea",
    "rice", "oryza sativa", "wheat", "triticum", "maize", "zea mays", "corn",
    "soybean", "glycine max", "potato", "solanum tuberosum", "tomato", "solanum lycopersicum",
    "chickpea", "cicer arietinum", "pigeonpea", "cajanus cajan", "groundnut", "arachis hypogaea",
    "mustard", "brassica juncea", "sunflower", "helianthus annuus", "sugarcane", "saccharum",
    "cotton", "gossypium", "onion", "allium cepa", "chilli", "capsicum frutescens",
    "gram", "cicer arietinum", "pigeon pea", "black gram", "green gram", "red gram",
    "barley", "hordeum vulgare", "oat", "avena sativa", "pearl millet", "pennisetum glaucum",
    "finger millet", "eleusine coracana", "sorghum", "sorghum bicolor", "ragi",
    "sunflower", "safflower", "carthamus tinctorius", "linseed", "linum usitatissimum",
    "sesame", "sesamum indicum", "castor", "ricinus communis", "jute", "corchorus",
}

DESIGN_KEYWORDS = [
    "randomized block design", "rcbd", "randomized complete block design",
    "split plot", "split-plot", "crd", "completely randomized design",
    "latin square", "factorial", "nested design", "strip plot",
    "augmented design", "alpha lattice", "row column design",
]


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def extract_text_from_pdf(path: Path) -> str:
    import concurrent.futures

    def _extract_pdfminer(p):
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        la_params = LAParams(detect_vertical=True, all_texts=True)
        return extract_text(str(p), laparams=la_params)

    def _extract_pypdf2(p):
        import PyPDF2
        parts = []
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts) if parts else ""

    def _extract_poppler(p):
        sys.path.insert(0, str(BASE_DIR))
        from aaif.extraction.poppler_reader import PopplerReader
        reader = PopplerReader()
        if not reader.available:
            return ""
        result = reader.extract(str(p))
        meta = result.get('metadata', {})
        return meta.get('full_text', '')

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Try pdfminer with 30s timeout
        future = executor.submit(_extract_pdfminer, path)
        try:
            text = future.result(timeout=30)
            if text and text.strip():
                return text
        except (concurrent.futures.TimeoutError, Exception):
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Try PyPDF2 with 20s timeout
        future = executor.submit(_extract_pypdf2, path)
        try:
            text = future.result(timeout=20)
            if text and text.strip():
                return text
        except (concurrent.futures.TimeoutError, Exception):
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Try Poppler with 30s timeout
        future = executor.submit(_extract_poppler, path)
        try:
            text = future.result(timeout=30)
            if text and text.strip():
                return text
        except (concurrent.futures.TimeoutError, Exception):
            pass

    return ""


def extract_tables_from_pdf(path: Path) -> list[dict]:
    """Extract treatment-level table data from PDF using pdfplumber, enhanced, then poppler."""
    import concurrent.futures

    def _extract_with_pdfplumber(p):
        from pdfplumber_extraction import process_pdf
        rows, meta = process_pdf(str(p))
        return rows

    def _extract_with_enhanced(p):
        from enhanced_extraction import process_one_pdf
        rows = process_one_pdf(str(p))
        return rows

    def _extract_with_poppler(p):
        sys.path.insert(0, str(BASE_DIR))
        from aaif.extraction.poppler_reader import PopplerReader
        reader = PopplerReader()
        if not reader.available:
            return []
        result = reader.extract(str(p))
        return result.get('rows', [])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_extract_with_pdfplumber, path)
        try:
            rows = future.result(timeout=30)
            if rows:
                return rows
        except (concurrent.futures.TimeoutError, Exception):
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_extract_with_enhanced, path)
        try:
            rows = future.result(timeout=30)
            if rows:
                return rows
        except (concurrent.futures.TimeoutError, Exception):
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_extract_with_poppler, path)
        try:
            rows = future.result(timeout=30)
            if rows:
                return rows
        except (concurrent.futures.TimeoutError, Exception):
            pass

    return []


def detect_crop(text: str) -> list[str]:
    crops = set()
    lower = text.lower()
    for crop, scientific in [
        ("Bell Pepper", ["bell pepper", "capsicum annuum", "capsicum"]),
        ("Black Wheat", ["black wheat", "triticum aestivum"]),
        ("Carrot", ["carrot", "daucus carota"]),
        ("Cowpea", ["cowpea", "vigna unguiculata", "lobia"]),
        ("Spinach", ["spinach", "spinacia oleracea"]),
        ("Rice", ["rice", "oryza sativa"]),
        ("Wheat", ["wheat", "triticum"]),
        ("Maize", ["maize", "zea mays", "corn"]),
        ("Soybean", ["soybean", "glycine max"]),
        ("Potato", ["potato", "solanum tuberosum"]),
        ("Tomato", ["tomato", "solanum lycopersicum"]),
        ("Chickpea", ["chickpea", "cicer arietinum", "gram", "chana"]),
        ("Mustard", ["mustard", "brassica juncea", "sarson"]),
        ("Groundnut", ["groundnut", "arachis hypogaea", "peanut"]),
        ("Sugarcane", ["sugarcane", "saccharum"]),
        ("Cotton", ["cotton", "gossypium"]),
        ("Sorghum", ["sorghum", "sorghum bicolor", "jowar"]),
        ("Pearl Millet", ["pearl millet", "bajra", "pennisetum glaucum"]),
        ("Finger Millet", ["finger millet", "ragi", "eleusine coracana"]),
        ("Barley", ["barley", "hordeum vulgare", "jau"]),
        ("Oat", ["oat", "avena sativa", "javi"]),
        ("Onion", ["onion", "allium cepa", "pyaaz"]),
        ("Chilli", ["chilli", "capsicum frutescens", "mirch"]),
        ("Brinjal", ["brinjal", "solanum melongena", "eggplant", "baingan"]),
        ("Cabbage", ["cabbage", "brassica oleracea", "bandh gobhi"]),
        ("Cauliflower", ["cauliflower", "brassica oleracea botrytis", "phool gobhi"]),
    ]:
        for kw in scientific:
            if kw in lower:
                crops.add(crop)
                break
    return sorted(crops) if crops else ["Unknown"]


def detect_doi(text: str) -> str:
    m = re.search(r"(10\.\d{4,}/[^\s]+)", text)
    return m.group(1).rstrip(".,;:") if m else ""


def detect_title(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines[:8]):
        s = line.strip()
        if len(s) > 20 and any(c.isalpha() for c in s):
            return s[:200]
    return Path(text[:100] if text else "").stem


def generate_paper_id(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^a-zA-Z0-9]", "_", stem)[:40]
    return f"PAPER_{stem}"


def normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower().strip())


# =====================================================================
# PHASE 1 — INGESTION
# =====================================================================
def phase1_ingestion() -> pd.DataFrame:
    log("=" * 60)
    log("PHASE 1: INGESTION")
    log("=" * 60)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    log(f"Found {len(pdf_files)} PDF files in {PAPERS_DIR}")

    records = []
    all_dois = {}
    all_titles = {}

    for pdf_path in pdf_files:
        start = time.time()
        paper_id = generate_paper_id(pdf_path.name)
        log(f"  Processing: {pdf_path.name} -> {paper_id}")

        text = extract_text_from_pdf(pdf_path)
        title = detect_title(text) if text else pdf_path.stem
        doi = detect_doi(text) if text else ""
        crops = detect_crop(text) if text else ["Unknown"]
        n_pages = max(1, len(text.split("\n")) // 50) if text else 0
        word_count = len(text.split()) if text else 0
        duration = round(time.time() - start, 2)

        dup_flag = "No"
        dup_with = ""
        norm_t = normalize_title(title)
        for existing_pid, existing_norm in all_titles.items():
            if norm_t == existing_norm or (doi and doi == all_dois.get(existing_pid, "")):
                dup_flag = "Yes"
                dup_with = existing_pid
                break
            if len(norm_t) > 20 and len(existing_norm) > 20:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, norm_t, existing_norm).ratio()
                if ratio > 0.90:
                    dup_flag = "Yes"
                    dup_with = existing_pid
                    break

        all_titles[paper_id] = norm_t
        if doi:
            all_dois[paper_id] = doi

        status = "Duplicate" if dup_flag == "Yes" else "New"
        records.append({
            "Paper_ID": paper_id,
            "Paper_Name": pdf_path.name,
            "Title": title,
            "Crop": ", ".join(crops),
            "DOI": doi,
            "Pages": n_pages,
            "Words": word_count,
            "Status": status,
            "Duplicate_Flag": dup_flag,
            "Duplicate_With": dup_with,
            "Processing_Time_s": duration,
        })
        log(f"    Crop={crops[0]}, DOI={doi[:40] if doi else 'N/A'}, Status={status}, {duration}s")

    ingestion_df = pd.DataFrame(records)

    xlsx_path = OUTPUTS_DIR / "ingestion_report.xlsx"
    ingestion_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    log(f"Ingestion report saved: {xlsx_path}")

    csv_path = OUTPUTS_DIR / "ingestion_report.csv"
    ingestion_df.to_csv(csv_path, index=False)
    log(f"Ingestion report CSV saved: {csv_path}")

    new_papers = ingestion_df[ingestion_df["Status"] == "New"]["Paper_ID"].tolist()
    log(f"New papers to process: {len(new_papers)}")
    log(f"Duplicate/Ignored: {len(ingestion_df) - len(new_papers)}")

    return ingestion_df


# =====================================================================
# PHASE 2-3 — AI EXTRACTION + UNIVERSAL SCHEMA
# =====================================================================
def phase2_3_extraction(ingestion_df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 2-3: AI EXTRACTION + UNIVERSAL SCHEMA")
    log("=" * 60)

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_rows = []

    for pdf_path in pdf_files:
        paper_id = generate_paper_id(pdf_path.name)
        row = ingestion_df[ingestion_df["Paper_ID"] == paper_id]
        if len(row) == 0:
            continue
        status = row.iloc[0]["Status"]
        if status == "Duplicate":
            log(f"  Skipping (duplicate): {pdf_path.name}")
            continue

        log(f"  Extracting: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            log(f"    WARNING: Empty text extraction for {pdf_path.name}")
            continue

        crop = row.iloc[0]["Crop"]
        doi = row.iloc[0]["DOI"]

        design_match = "RBD"
        for kw in DESIGN_KEYWORDS:
            if kw in text.lower():
                design_map = {
                    "randomized block design": "RBD", "rcbd": "RCBD",
                    "randomized complete block design": "RCBD",
                    "split plot": "Split Plot", "split-plot": "Split Plot",
                    "crd": "CRD", "completely randomized design": "CRD",
                    "latin square": "Latin Square", "factorial": "Factorial",
                    "augmented design": "Augmented",
                }
                for k, v in design_map.items():
                    if k in text.lower():
                        design_match = v
                        break
                break

        replications = 3
        rm = re.search(r"(\d+)\s*(?:replications?|replicates?|rep\b)", text, re.IGNORECASE)
        if rm:
            replications = int(rm.group(1))

        ph_vals = [float(m.group(1)) for m in re.finditer(r"(?:soil\s*)?ph\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)]
        ec_vals = [float(m.group(1)) for m in re.finditer(r"(?:ec|electrical\s*conductivity)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)]
        oc_vals = [float(m.group(1)) for m in re.finditer(r"(?:organic\s*carbon|oc)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*%?", text, re.IGNORECASE)]
        n_vals = [float(m.group(1)) for m in re.finditer(r"(?:available\s*nitrogen|total\s*nitrogen|soil\s*nitrogen|nitrogen\s*content)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?", text, re.IGNORECASE)]
        if not n_vals:
            n_vals = [float(m.group(1)) for m in re.finditer(r"(?:\bnitrogen)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)", text, re.IGNORECASE)]
        p_vals = [float(m.group(1)) for m in re.finditer(r"(?:available\s*phosphorus|phosphorus|phosphorous|p2o5)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?", text, re.IGNORECASE)]
        k_vals = [float(m.group(1)) for m in re.finditer(r"(?:available\s*potassium|potassium|k2o)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?", text, re.IGNORECASE)]
        tmax_vals = [float(m.group(1)) for m in re.finditer(r"(?:max\s*temp|tmax|temperature\s*max|t\.?\s*max)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)]
        tmin_vals = [float(m.group(1)) for m in re.finditer(r"(?:min\s*temp|tmin|temperature\s*min|t\.?\s*min)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)]
        rain_vals = [float(m.group(1)) for m in re.finditer(r"(?:rainfall|precipitation|annual\s*rainfall)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:mm)?", text, re.IGNORECASE)]
        yield_vals = []
        for pat in [
            r"(?:yield|grain\s*yield|seed\s*yield|fruit\s*yield|biological\s*yield|economic\s*yield)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha|q/ha|t|kg|g)",
            r"(?:yield|grain\s*yield)\s*(?:was|:|=)\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha)",
            r"(?:kg\s*ha-1|t\s*ha-1)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))",
            r"((?:\d+\.?\d*|\.\d+))\s*(?:kg\s*ha-1|t\s*ha-1)",
            r"(?:recorded|observed|measured|obtained)\s*(?:yield|grain\s*yield)\s*(?:of|:|=)\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha)?",
        ]:
            yield_vals.extend([float(m.group(1)) for m in re.finditer(pat, text, re.IGNORECASE)])
        height_vals = [float(m.group(1)) for m in re.finditer(r"(?:plant\s*height|height)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*cm", text, re.IGNORECASE)]
        spad_vals = [float(m.group(1)) for m in re.finditer(r"(?:spad|chlorophyll)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE)]
        protein_vals = [float(m.group(1)) for m in re.finditer(r"(?:protein|protein\s*content|crude\s*protein)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*%", text, re.IGNORECASE)]

        zn_vals = [float(m.group(1)) for m in re.finditer(r"(?:zinc|zn)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:ppm|mg/kg)?", text, re.IGNORECASE)]
        fe_vals = [float(m.group(1)) for m in re.finditer(r"(?:iron|fe)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:ppm|mg/kg)?", text, re.IGNORECASE)]

        def first_or_none(vals):
            return vals[0] if vals else None

        base_row = {
            "Paper_ID": paper_id, "DOI": doi,
            "Crop": crop.split(",")[0].strip() if crop else "",
            "Design": design_match, "Replications": replications,
            "Soil_pH": first_or_none(ph_vals),
            "EC": first_or_none(ec_vals),
            "Organic_Carbon": first_or_none(oc_vals),
            "Nitrogen": first_or_none(n_vals),
            "Phosphorus": first_or_none(p_vals),
            "Potassium": first_or_none(k_vals),
            "Temperature_Max": first_or_none(tmax_vals),
            "Temperature_Min": first_or_none(tmin_vals),
            "Rainfall": first_or_none(rain_vals),
            "Yield_per_Hectare": first_or_none(yield_vals),
            "Plant_Height_cm": first_or_none(height_vals),
            "SPAD": first_or_none(spad_vals),
            "Protein": first_or_none(protein_vals),
            "Zinc": first_or_none(zn_vals),
            "Iron": first_or_none(fe_vals),
        }

        fertilizers_detected = set()
        fert_list = ["urea", "dap", "npk", "compost", "vermicompost",
                      "farmyard manure", "fym", "potassium sulphate",
                      "ammonium sulphate", "single super phosphate", "ssp",
                      "muriate of potash", "mop", "biofertilizer",
                      "rhizobium", "azotobacter", "psb", "pgpr"]
        for fert in fert_list:
            pattern = re.compile(re.escape(fert), re.IGNORECASE)
            if pattern.search(text):
                fertilizers_detected.add(fert.title())

        treatment_matches = re.findall(r"(T\d+|T\d+[A-Z]?|Treatment\s*\d+)", text, re.IGNORECASE)
        treatment = ", ".join(sorted(set(treatment_matches))) if treatment_matches else "Control"

        base_row["Treatment"] = treatment
        base_row["Fertilizer_Name"] = ", ".join(sorted(fertilizers_detected))

        all_rows.append(base_row)
        log(f"    Extracted: pH={base_row['Soil_pH']}, N={base_row['Nitrogen']}, "
            f"Yield={base_row['Yield_per_Hectare']}, Fert={len(fertilizers_detected)}")

    if not all_rows:
        log("WARNING: No data extracted from any paper")
        return pd.DataFrame()

    extract_df = pd.DataFrame(all_rows)

    # Phase 2b: Attempt pdfplumber table extraction to enrich data
    log("\n  Attempting pdfplumber table extraction...")
    table_enrichment_rows = []
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        paper_id = generate_paper_id(pdf_path.name)
        row = ingestion_df[ingestion_df["Paper_ID"] == paper_id]
        if len(row) > 0 and row.iloc[0]["Status"] == "Duplicate":
            continue
        try:
            table_rows = extract_tables_from_pdf(pdf_path)
            if table_rows:
                for tr in table_rows:
                    tr["Paper_ID"] = paper_id
                    tr["Source_File"] = pdf_path.name
                table_enrichment_rows.extend(table_rows)
                log(f"    {pdf_path.name}: {len(table_rows)} table rows extracted")
        except Exception as e:
            log(f"    {pdf_path.name}: table extraction error: {e}")

    if table_enrichment_rows:
        table_df = pd.DataFrame(table_enrichment_rows)
        log(f"  Total table rows from pdfplumber: {len(table_df)}")

        # Merge table data with regex-extracted data on Paper_ID
        # For each paper, if pdfplumber has Yield_per_Hectare and regex doesn't, use pdfplumber value
        for col in ["Yield_per_Hectare", "Yield_per_Plot", "Plant_Height_cm", "SPAD",
                     "Soil_pH", "Nitrogen", "Phosphorus", "Potassium", "Zinc", "Iron",
                     "Protein", "Organic_Carbon", "EC", "Rainfall",
                     "Temperature_Max", "Temperature_Min", "Fruit_Weight",
                     "Fruit_Diameter_mm", "Spike_Length", "Seeds_per_Spike",
                     "100_Seed_Weight", "Biomass_Yield", "Harvest_Index"]:
            if col in table_df.columns:
                for paper_id in table_df["Paper_ID"].unique():
                    if paper_id in extract_df["Paper_ID"].values:
                        mask_reg = extract_df["Paper_ID"] == paper_id
                        mask_tbl = table_df["Paper_ID"] == paper_id
                        if col in extract_df.columns:
                            reg_val = extract_df.loc[mask_reg, col].iloc[0] if mask_reg.any() else None
                        else:
                            reg_val = None
                        tbl_vals = table_df.loc[mask_tbl, col].dropna()
                        if (pd.isna(reg_val) or reg_val is None) and len(tbl_vals) > 0:
                            # Use mean of table values for this paper
                            if col in extract_df.columns:
                                extract_df.loc[mask_reg, col] = tbl_vals.mean()
                            log(f"    Enriched {paper_id}.{col} = {tbl_vals.mean():.2f} (from table)")

        # Add Treatment and Fertilizer info from table extraction if not already present
        if "Treatment" in table_df.columns:
            for paper_id in table_df["Paper_ID"].unique():
                if paper_id in extract_df["Paper_ID"].values:
                    treatments = table_df.loc[table_df["Paper_ID"] == paper_id, "Treatment"].dropna().astype(str).unique()
                    mask = extract_df["Paper_ID"] == paper_id
                    if "Treatment" in extract_df.columns:
                        existing = str(extract_df.loc[mask, "Treatment"].iloc[0]) if mask.any() else ""
                        if not existing or existing == "Control" or existing == "nan":
                            extract_df.loc[mask, "Treatment"] = ", ".join(sorted(treatments)[:5])

    for col in UAMS_COLUMNS:
        if col not in extract_df.columns:
            extract_df[col] = pd.NA

    ordered = [c for c in UAMS_COLUMNS if c in extract_df.columns]
    extra = [c for c in extract_df.columns if c not in UAMS_COLUMNS]
    extract_df = extract_df[ordered + extra]

    xlsx_path = OUTPUTS_DIR / "Universal_Agricultural_Schema.xlsx"
    extract_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    log(f"Universal Schema XLSX saved: {xlsx_path}")

    csv_path = OUTPUTS_DIR / "Universal_Agricultural_Schema.csv"
    extract_df.to_csv(csv_path, index=False)
    log(f"Universal Schema CSV saved: {csv_path}")

    log(f"Extraction complete: {len(extract_df)} rows, {len(extract_df.columns)} columns")
    return extract_df


# =====================================================================
# PHASE 4 — VALIDATION
# =====================================================================
def phase4_validation(df: pd.DataFrame) -> dict:
    log("\n" + "=" * 60)
    log("PHASE 4: VALIDATION")
    log("=" * 60)

    issues = {
        "missing_values": {}, "duplicate_records": 0, "impossible_values": [],
        "outliers": [], "ocr_mistakes": [], "unit_inconsistencies": [],
        "column_consistency": [],
    }

    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = round(missing / len(df) * 100, 1) if len(df) > 0 else 0
        if pct > 0:
            issues["missing_values"][col] = f"{missing}/{len(df)} ({pct}%)"

    issues["duplicate_records"] = int(df.duplicated().sum())

    range_checks = {
        "Soil_pH": (3.0, 10.0), "EC": (0, 20),
        "Temperature_Max": (-20, 55), "Temperature_Min": (-30, 50),
        "Rainfall": (0, 10000), "Yield_per_Hectare": (0, 50000),
        "Plant_Height_cm": (0, 500), "SPAD": (0, 80),
        "Protein": (0, 60), "Organic_Carbon": (0, 10),
        "Nitrogen": (0, 1000), "Phosphorus": (0, 500),
        "Potassium": (0, 1000), "Zinc": (0, 100), "Iron": (0, 500),
    }

    for col, (lo, hi) in range_checks.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            bad = int(((vals < lo) | (vals > hi)).sum())
            if bad > 0:
                issues["impossible_values"].append(f"{col}: {bad} outside [{lo}, {hi}]")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) > 3:
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo = q1 - 3.0 * iqr
                hi = q3 + 3.0 * iqr
                out = int(((vals < lo) | (vals > hi)).sum())
                if out > 0:
                    issues["outliers"].append(f"{col}: {out} mild outliers")

    for col in df.columns:
        if df[col].dtype == object:
            strange = df[col].dropna().apply(
                lambda x: bool(re.search(r"[^a-zA-Z0-9\s.,;:\-()/%]", str(x)))
            ).sum()
            if strange > 0:
                issues["ocr_mistakes"].append(f"{col}: {strange} cells with special chars")

    if "Temperature_Max" in df.columns and "Temperature_Min" in df.columns:
        tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
        tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
        bad = int((tmax < tmin).sum())
        if bad > 0:
            issues["column_consistency"].append(f"Tmax<Tmin in {bad} rows")

    if "Yield_per_Hectare" in df.columns and "Plant_Height_cm" in df.columns:
        cols_present = [c for c in df.columns if c in ["Yield_per_Hectare", "Plant_Height_cm", "SPAD"]]
        issues["column_consistency"].append(f"Columns with data: {len(cols_present)}")

    total_issues = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in issues.values())
    log(f"Validation complete: {total_issues} issues")

    report_rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = round(n_missing / len(df) * 100, 1) if len(df) > 0 else 0
        col_issues = []
        for iv in issues["impossible_values"]:
            if col in iv:
                col_issues.append(iv)
        for ov in issues["outliers"]:
            if col in ov.split(":")[0]:
                col_issues.append(ov)
        report_rows.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Non_Null": len(df) - n_missing,
            "Missing": n_missing,
            "Missing_Pct": pct_missing,
            "Unique": int(df[col].nunique()),
            "Issues": "; ".join(col_issues),
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_excel(OUTPUTS_DIR / "validation_report.xlsx", index=False, engine="openpyxl")
    log(f"Validation report saved: {OUTPUTS_DIR / 'validation_report.xlsx'}")

    return issues


# =====================================================================
# PHASE 5 — FEATURE ENGINEERING
# =====================================================================
def phase5_features(df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 5: FEATURE ENGINEERING")
    log("=" * 60)

    fdf = df.copy()
    features_added = []

    if "Temperature_Max" in fdf.columns and "Temperature_Min" in fdf.columns:
        tmax = pd.to_numeric(fdf["Temperature_Max"], errors="coerce")
        tmin = pd.to_numeric(fdf["Temperature_Min"], errors="coerce")
        tmean = (tmax + tmin) / 2
        fdf["Average_Temperature"] = tmean
        features_added.append("Average_Temperature")
        fdf["Growing_Degree_Days"] = np.maximum(0, tmean - 10)
        features_added.append("Growing_Degree_Days")
        fdf["Temp_squared"] = tmean ** 2
        features_added.append("Temp_squared")

    if "Rainfall" in fdf.columns:
        rf = pd.to_numeric(fdf["Rainfall"], errors="coerce")
        fdf["Rainfall_Anomaly"] = rf - rf.mean() if rf.notna().any() else 0
        features_added.append("Rainfall_Anomaly")

    if "Nitrogen" in fdf.columns and "Phosphorus" in fdf.columns:
        n = pd.to_numeric(fdf["Nitrogen"], errors="coerce")
        p = pd.to_numeric(fdf["Phosphorus"], errors="coerce")
        fdf["N_x_P"] = n * p
        features_added.append("N_x_P")

    if all(c in fdf.columns for c in ["Nitrogen", "Phosphorus", "Potassium"]):
        n = pd.to_numeric(fdf["Nitrogen"], errors="coerce")
        p = pd.to_numeric(fdf["Phosphorus"], errors="coerce")
        k = pd.to_numeric(fdf["Potassium"], errors="coerce")
        fdf["NPK_Index"] = n + p + k
        features_added.append("NPK_Index")

    if all(c in fdf.columns for c in ["Nitrogen", "Phosphorus", "Potassium", "Zinc", "Iron", "Organic_Carbon"]):
        fdf["Soil_Fertility_Index"] = (
            pd.to_numeric(fdf["Nitrogen"], errors="coerce").fillna(0) / 100 +
            pd.to_numeric(fdf["Phosphorus"], errors="coerce").fillna(0) / 50 +
            pd.to_numeric(fdf["Potassium"], errors="coerce").fillna(0) / 200 +
            pd.to_numeric(fdf["Zinc"], errors="coerce").fillna(0) / 5 +
            pd.to_numeric(fdf["Iron"], errors="coerce").fillna(0) / 50 +
            (pd.to_numeric(fdf["Organic_Carbon"], errors="coerce").fillna(0) * 2)
        )
        features_added.append("Soil_Fertility_Index")

    if "Organic_Carbon" in fdf.columns:
        fdf["Organic_Matter_Index"] = pd.to_numeric(fdf["Organic_Carbon"], errors="coerce") * 1.724
        features_added.append("Organic_Matter_Index")

    if all(c in fdf.columns for c in ["Zinc", "Iron"]):
        zn = pd.to_numeric(fdf["Zinc"], errors="coerce").fillna(0)
        fe = pd.to_numeric(fdf["Iron"], errors="coerce").fillna(0)
        fdf["Micronutrient_Index"] = zn + fe
        features_added.append("Micronutrient_Index")

    if "Rainfall" in fdf.columns and "Temperature_Max" in fdf.columns:
        tmean2 = (pd.to_numeric(fdf["Temperature_Max"], errors="coerce") +
                  pd.to_numeric(fdf["Temperature_Min"] if "Temperature_Min" in fdf.columns else fdf["Temperature_Max"], errors="coerce")) / 2
        rf2 = pd.to_numeric(fdf["Rainfall"], errors="coerce")
        fdf["Climate_Index"] = (rf2 / 1000) + (tmean2 / 35)
        features_added.append("Climate_Index")

    if "Crop" in fdf.columns:
        le = {c: i for i, c in enumerate(fdf["Crop"].dropna().unique())}
        fdf["Crop_Code"] = fdf["Crop"].map(le).fillna(-1).astype(int)
        features_added.append("Crop_Code")

    if "Treatment" in fdf.columns:
        fdf["Treatment_Code"] = pd.factorize(fdf["Treatment"].fillna("Unknown"))[0]
        features_added.append("Treatment_Code")

    if "Fertilizer_Name" in fdf.columns:
        fdf["Fertilizer_Code"] = pd.factorize(fdf["Fertilizer_Name"].fillna("None"))[0]
        features_added.append("Fertilizer_Code")

    if "Yield_per_Hectare" in fdf.columns and "Nitrogen" in fdf.columns:
        y = pd.to_numeric(fdf["Yield_per_Hectare"], errors="coerce")
        n = pd.to_numeric(fdf["Nitrogen"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            fdf["Nitrogen_Use_Efficiency"] = np.where(n > 0, y / n, np.nan)
        features_added.append("Nitrogen_Use_Efficiency")

    if "Yield_per_Hectare" in fdf.columns and "Rainfall" in fdf.columns:
        y = pd.to_numeric(fdf["Yield_per_Hectare"], errors="coerce")
        r = pd.to_numeric(fdf["Rainfall"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            fdf["Water_Use_Efficiency"] = np.where(r > 0, y / r, np.nan)
        features_added.append("Water_Use_Efficiency")

    if "Yield_per_Hectare" in fdf.columns and "Plant_Height_cm" in fdf.columns:
        y = pd.to_numeric(fdf["Yield_per_Hectare"], errors="coerce")
        h = pd.to_numeric(fdf["Plant_Height_cm"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            fdf["Growth_Yield_Index"] = np.where(h > 0, y / h, np.nan)
        features_added.append("Growth_Yield_Index")

    csv_path = OUTPUTS_DIR / "features_dataset.csv"
    fdf.to_csv(csv_path, index=False)
    log(f"Features dataset saved: {csv_path}")
    log(f"Features added: {len(features_added)}")
    log(f"Dataset shape: {fdf.shape}")

    return fdf


# =====================================================================
# PHASE 6 — MODEL TRAINING
# =====================================================================
def phase6_training(df: pd.DataFrame) -> dict:
    log("\n" + "=" * 60)
    log("PHASE 6: MODEL TRAINING")
    log("=" * 60)

    results = {}

    target = "Yield_per_Hectare"
    if target not in df.columns:
        for alt in ["Target_Yield", "Yield_per_Plot", "Yield_per_Hectare_Calc"]:
            if alt in df.columns:
                target = alt
                break
        else:
            log(f"WARNING: No target yield column found")
            return {"error": "No target column"}

    exclude = {"Paper_ID", "DOI", "Crop", "Treatment", "Fertilizer_Name",
               "Season", "Variety", "Location", "Site", "State",
               "Recommendation_Summary", "Fuzzy_Summary"}

    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in exclude and c != target]

    if not feature_cols:
        log("WARNING: No feature columns found")
        return {"error": "No features"}

    X = numeric_df[feature_cols].copy()
    y = numeric_df[target].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    for col in X.columns:
        if X[col].isna().sum() < len(X):
            X[col] = X[col].fillna(X[col].median())

    non_null_features = [c for c in X.columns if X[c].notna().any()]
    if not non_null_features:
        log("WARNING: All features are null")
        return {"error": "All features null"}

    X = X[non_null_features]

    valid = y.notna()
    X = X[valid]
    y = y[valid]

    if len(X) < 2:
        log(f"WARNING: Insufficient samples: {len(X)}")
        return {"error": f"Insufficient samples: {len(X)}"}

    log(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    if len(X) < 10:
        log(f"WARNING: Very small dataset ({len(X)} samples) — results may not be reliable")

    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_dict = {}

    log("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models_dict["Multiple_Linear_Regression"] = lr

    try:
        from xgboost import XGBRegressor
        log("Training XGBoost...")
        xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=1)
        xgb.fit(X_train, y_train)
        models_dict["XGBoost"] = xgb
    except Exception as e:
        log(f"XGBoost not available: {e}")
        log("Falling back to Random Forest...")
        xgb = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        models_dict["XGBoost_Alternative"] = xgb

    log("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models_dict["Random_Forest"] = rf

    training_results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)

        log(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
        training_results.append({
            "Model": name, "Target": target,
            "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MAPE": round(mape, 2),
        })

    cv_scores = {}
    for name, model in models_dict.items():
        try:
            scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="r2")
            cv_scores[name] = {
                "mean_r2": round(scores.mean(), 4),
                "std_r2": round(scores.std(), 4),
            }
            log(f"  {name} CV R2: {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as e:
            log(f"  {name} CV failed: {e}")

    best_model_name = max(training_results, key=lambda r: r["R2"])["Model"]
    best_model = models_dict[best_model_name]
    best_model.fit(X, y)

    import joblib
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    if "XGBoost" in models_dict:
        joblib.dump(models_dict["XGBoost"], xgb_path)
    elif "XGBoost_Alternative" in models_dict:
        joblib.dump(models_dict["XGBoost_Alternative"], xgb_path)
    log(f"XGBoost model saved: {xgb_path}")

    reg_path = MODELS_DIR / "regression_model.pkl"
    joblib.dump(lr, reg_path)
    log(f"Regression model saved: {reg_path}")

    feature_importance = []
    if hasattr(best_model, "feature_importances_"):
        for feat, imp in zip(feature_cols, best_model.feature_importances_):
            feature_importance.append({"Feature": feat, "Importance": round(imp, 4)})
        fi_df = pd.DataFrame(feature_importance).sort_values("Importance", ascending=False)
        fi_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    elif hasattr(best_model, "coef_"):
        for feat, coef in zip(feature_cols, best_model.coef_):
            feature_importance.append({"Feature": feat, "Coefficient": round(coef, 4)})
        fi_df = pd.DataFrame(feature_importance)
        fi_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    results_df = pd.DataFrame(training_results)
    results_df["CV_Mean_R2"] = results_df["Model"].map(
        lambda m: cv_scores.get(m, {}).get("mean_r2", "")
    )
    results_df["CV_Std_R2"] = results_df["Model"].map(
        lambda m: cv_scores.get(m, {}).get("std_r2", "")
    )

    r_xlsx = OUTPUTS_DIR / "model_metrics.xlsx"
    results_df.to_excel(r_xlsx, index=False, engine="openpyxl")
    log(f"Model metrics saved: {r_xlsx}")

    metrics = {
        "training_results": training_results,
        "cv_scores": cv_scores,
        "best_model": best_model_name,
        "target": target,
        "n_features": len(feature_cols),
        "n_samples": len(X),
    }
    return metrics


# =====================================================================
# PHASE 7 — FUZZY LOGIC
# =====================================================================
def phase7_fuzzy(df: pd.DataFrame) -> bool:
    log("\n" + "=" * 60)
    log("PHASE 7: FUZZY LOGIC")
    log("=" * 60)

    from agri_ai_agent.rules.membership_functions import fuzzify
    from agri_ai_agent.rules.output_memberships import OUTPUT_MEMBERSHIPS

    rules_path = Path(__file__).parent / "agri_ai_agent" / "rules" / "fertilizer_rules.yaml"
    fuzzy_output_path = FUZZY_DIR

    with open(rules_path) as f:
        rules_data = yaml.safe_load(f)

    log(f"Loaded {len(rules_data.get('rules', []))} fuzzy rules from {rules_path}")

    input_vars = rules_data.get("variables", {}).get("input", [])
    log(f"Input variables: {input_vars}")

    sample_count = 0
    for idx in df.index[:5]:
        row = df.loc[idx]
        fuzzy_inputs = {}
        for var in input_vars:
            if var in row and pd.notna(row[var]):
                mfs = fuzzify(var, float(row[var]))
                if mfs:
                    fuzzy_inputs[var] = mfs
        if fuzzy_inputs:
            sample_count += 1

    log(f"Fuzzification tested on {sample_count} sample rows")

    updated_path = fuzzy_output_path / "fertilizer_rules.yaml"
    import shutil
    shutil.copy2(rules_path, updated_path)
    log(f"Fertilizer rules copied to: {updated_path}")

    doc_lines = [
        "# Fuzzy Logic System",
        f"Generated: {datetime.now().isoformat()}",
        f"Input variables: {len(input_vars)}",
        f"Rules: {len(rules_data.get('rules', []))}",
        "",
        "## Membership Functions",
    ]
    for var, mfs in {
        "Nitrogen": {"low": (0, 30), "medium": (30, 60, 100), "high": (80, 120)},
        "Phosphorus": {"low": (0, 10), "medium": (10, 25, 40), "high": (30, 50)},
        "Potassium": {"low": (0, 80), "medium": (80, 150, 250), "high": (180, 300)},
        "Soil_pH": {"acidic": (0, 5.5), "neutral": (5.5, 6.5, 7.5, 8.0), "alkaline": (7.5, 14)},
        "Rainfall": {"low": (0, 400), "medium": (400, 750, 1100), "high": (900, 1200)},
        "Temperature_Max": {"cool": (0, 20), "moderate": (20, 25, 32, 35), "hot": (32, 38)},
        "Organic_Carbon": {"low": (0, 0.4), "medium": (0.4, 0.75, 1.0), "high": (0.8, 1.2)},
    }.items():
        doc_lines.append(f"\n### {var}")
        for label, params in mfs.items():
            doc_lines.append(f"  - {label}: {params}")

    doc_lines.append("\n## Rules")
    for rule in rules_data.get("rules", []):
        ant = "; ".join(f"{a['var']}={a['set']}" for a in rule.get("antecedents", []))
        cons_parts = []
        for c in rule.get("consequents", []):
            if c["type"] == "nutrient":
                cons_parts.append(f"{c['nutrient']}: {c['action']}")
        cons_str = ", ".join(cons_parts)
        doc_lines.append(f"  - [{rule.get('priority', 0)}] IF {ant} THEN {cons_str}")

    doc_path = fuzzy_output_path / "fuzzy_report.md"
    Path(doc_path).write_text("\n".join(doc_lines), encoding="utf-8")
    log(f"Fuzzy report saved: {doc_path}")

    log("Phase 7 complete")
    return True


# =====================================================================
# PHASE 8 — RECOMMENDATION AGENT
# =====================================================================
def phase8_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 8: RECOMMENDATION AGENT")
    log("=" * 60)

    rdf = df.copy()
    recs = []

    for idx in rdf.index:
        row = rdf.loc[idx]

        n = pd.to_numeric(row.get("Nitrogen"), errors="coerce") if pd.notna(row.get("Nitrogen")) else None
        p = pd.to_numeric(row.get("Phosphorus"), errors="coerce") if pd.notna(row.get("Phosphorus")) else None
        k = pd.to_numeric(row.get("Potassium"), errors="coerce") if pd.notna(row.get("Potassium")) else None
        ph = pd.to_numeric(row.get("Soil_pH"), errors="coerce") if pd.notna(row.get("Soil_pH")) else None
        rainfall = pd.to_numeric(row.get("Rainfall"), errors="coerce") if pd.notna(row.get("Rainfall")) else None
        yield_val = pd.to_numeric(row.get("Yield_per_Hectare"), errors="coerce") if pd.notna(row.get("Yield_per_Hectare")) else None
        oc = pd.to_numeric(row.get("Organic_Carbon"), errors="coerce") if pd.notna(row.get("Organic_Carbon")) else None
        zn = pd.to_numeric(row.get("Zinc"), errors="coerce") if pd.notna(row.get("Zinc")) else None
        crop = str(row.get("Crop", ""))

        if n is not None and pd.isna(n):
            n = None
        if p is not None and pd.isna(p):
            p = None
        if k is not None and pd.isna(k):
            k = None

        if n is not None and n < 50:
            best_fert = "Urea (Nitrogen-rich)"
            n_rec = "Increase nitrogen"
        elif p is not None and p < 15:
            best_fert = "DAP (Phosphorus-rich)"
            n_rec = "Increase phosphorus"
        elif k is not None and k < 100:
            best_fert = "MOP (Potassium-rich)"
            n_rec = "Increase potassium"
        elif n is not None and n > 150:
            best_fert = "SSP (Low nitrogen)"
            n_rec = "Reduce nitrogen, apply phosphorus"
        elif ph is not None and ph < 5.5:
            best_fert = "Dolomite + Bio NPK"
            n_rec = "Apply lime to correct pH, then balanced NPK"
        elif ph is not None and ph > 8.0:
            best_fert = "Ammonium Sulphate + Gypsum"
            n_rec = "Apply sulfur-based amendments to reduce pH"
        elif oc is not None and oc < 0.4:
            best_fert = "Compost + Vermicompost"
            n_rec = "Increase organic matter, apply slow-release fertilizer"
        elif zn is not None and zn < 0.5:
            best_fert = "Zinc Sulphate + Balanced NPK"
            n_rec = "Apply zinc foliar spray + balanced NPK"
        else:
            best_fert = "Balanced NPK (15-15-15)"
            n_rec = "Maintain current balanced fertilization"

        if rainfall is not None and rainfall > 1000:
            interval = "Split application (3-4 splits)"
        elif rainfall is not None and rainfall < 500:
            interval = "Every 15-20 days"
        else:
            interval = "Every 25-30 days"

        if yield_val is not None:
            expected_yield = round(yield_val * 1.15, 2)
            expected_increase = round(yield_val * 0.15, 2)
        else:
            expected_yield = None
            expected_increase = None

        conf_score = 0.75
        if n is not None and p is not None and k is not None and ph is not None:
            conf_score = 0.85
        elif n is not None or p is not None or k is not None:
            conf_score = 0.65

        conf_label = "High" if conf_score >= 0.7 else "Medium" if conf_score >= 0.4 else "Low"

        alt_treatments = []
        if best_fert != "Balanced NPK (15-15-15)":
            alt_treatments.append("Balanced NPK (15-15-15)")
        if "Urea" in best_fert:
            alt_treatments.append("DAP + MOP")
        if "DAP" in best_fert:
            alt_treatments.append("SSP + Urea")
        alt_treatments.append("Organic compost")

        recs.append({
            "Crop": crop,
            "Paper_ID": row.get("Paper_ID", ""),
            "Best_Treatment": best_fert,
            "Expected_Yield_kg_ha": expected_yield,
            "Yield_Increase_kg_ha": expected_increase,
            "Confidence_Score": conf_score,
            "Confidence_Label": conf_label,
            "Alternative_Treatments": "; ".join(alt_treatments[:3]),
            "Recommendation": n_rec,
            "Application_Interval": interval,
        })

    rec_df = pd.DataFrame(recs)
    rec_path = OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.xlsx"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_excel(rec_path, index=False, engine="openpyxl")
    log(f"Recommendations saved: {rec_path}")

    rec_csv = OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.csv"
    rec_df.to_csv(rec_csv, index=False)
    log(f"Recommendations CSV saved: {rec_csv}")

    best_by_crop = rec_df.groupby("Crop").first().reset_index()
    for _, row in best_by_crop.iterrows():
        log(f"  {row['Crop']}: {row['Best_Treatment']} (Conf: {row['Confidence_Label']})")

    return rec_df


# =====================================================================
# PHASE 9 — READY RECKONER
# =====================================================================
def phase9_ready_reckoner(df: pd.DataFrame, rec_df: pd.DataFrame) -> bool:
    log("\n" + "=" * 60)
    log("PHASE 9: READY RECKONER")
    log("=" * 60)

    reckoner_rows = []

    for _, rec in rec_df.iterrows():
        crop = rec.get("Crop", "")
        paper_id = rec.get("Paper_ID", "")

        paper_row = df[df["Paper_ID"] == paper_id]
        if len(paper_row) > 0:
            pr = paper_row.iloc[0]
            ph = pr.get("Soil_pH")
            n = pr.get("Nitrogen")
            p = pr.get("Phosphorus")
            k = pr.get("Potassium")
            tmax = pr.get("Temperature_Max")
            tmin = pr.get("Temperature_Min")
            rainfall = pr.get("Rainfall")
            oc = pr.get("Organic_Carbon")
            yield_ha = pr.get("Yield_per_Hectare")
        else:
            ph = n = p = k = tmax = tmin = rainfall = oc = yield_ha = None

        soil_cond = f"pH {ph}" if pd.notna(ph) else "N/A"
        if pd.notna(n):
            soil_cond += f", N {n}"
        if pd.notna(p):
            soil_cond += f", P {p}"
        if pd.notna(k):
            soil_cond += f", K {k}"
        if pd.notna(oc):
            soil_cond += f", OC {oc}%"

        climate = ""
        if pd.notna(tmax) and pd.notna(tmin):
            climate = f"T {tmin}-{tmax}°C"
        if pd.notna(rainfall):
            climate += f", Rain {rainfall}mm"

        method_map = {
            "Urea": "Soil application (band placement)", "DAP": "Soil application at sowing",
            "MOP": "Soil application", "NPK": "Broadcast + incorporation",
            "Compost": "Broadcast + incorporation", "Zinc": "Foliar spray",
            "SSP": "Soil application at sowing", "Ammonium": "Soil application",
            "Dolomite": "Soil application (broadcast)",
        }
        app_method = "Soil application"
        for kw, method in method_map.items():
            if kw.lower() in str(rec.get("Best_Treatment", "")).lower():
                app_method = method
                break

        reckoner_rows.append({
            "Crop": crop,
            "Recommended_Treatment": rec.get("Best_Treatment", ""),
            "Soil_Condition": soil_cond,
            "Climate_Condition": climate,
            "Expected_Yield_kg_ha": rec.get("Expected_Yield_kg_ha", ""),
            "Recommended_Fertilizer": rec.get("Best_Treatment", ""),
            "Application_Method": app_method,
            "Application_Interval": rec.get("Application_Interval", ""),
            "Confidence_Score": rec.get("Confidence_Score", ""),
            "Recommendation": rec.get("Recommendation", ""),
        })

    reckoner_df = pd.DataFrame(reckoner_rows)

    r_xlsx = OUTPUTS_DIR / "Ready_Reckoner.xlsx"
    reckoner_df.to_excel(r_xlsx, index=False, engine="openpyxl")
    log(f"Ready Reckoner XLSX saved: {r_xlsx}")

    try:
        import weasyprint
        html = _render_reckoner_html(reckoner_df)
        pdf_path = OUTPUTS_DIR / "Ready_Reckoner.pdf"
        weasyprint.HTML(string=html).write_pdf(pdf_path)
        log(f"Ready Reckoner PDF saved: {pdf_path}")
    except ImportError:
        log("weasyprint not available, generating HTML instead")
        html = _render_reckoner_html(reckoner_df)
        html_path = OUTPUTS_DIR / "Ready_Reckoner.html"
        Path(html_path).write_text(html, encoding="utf-8")
        log(f"Ready Reckoner HTML saved: {html_path}")

    return True


def _render_reckoner_html(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    thead = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    tbody = ""
    for _, row in df.iterrows():
        tbody += "<tr>" + "".join(
            f"<td>{v}</td>" if not isinstance(v, float) else f"<td>{v:.2f}</td>"
            for v in row
        ) + "</tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AAIF Ready Reckoner</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; }}
h1 {{ color: #27ae60; border-bottom: 2px solid #27ae60; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #bbb; padding: 6px; text-align: left; }}
th {{ background: #27ae60; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.footer {{ margin-top: 20px; font-size: 11px; color: #95a5a6; text-align: center; }}
</style>
</head>
<body>
<h1>AAIF Ready Reckoner</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(df)} entries</p>
<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
<div class="footer">Agentic Agricultural Intelligence Framework</div>
</body>
</html>"""


# =====================================================================
# PHASE 10 — CONTINUOUS LEARNING
# =====================================================================
def phase10_continuous_learning(ingestion_df: pd.DataFrame) -> bool:
    log("\n" + "=" * 60)
    log("PHASE 10: CONTINUOUS LEARNING")
    log("=" * 60)

    db_path = DB_DIR / "paper_registry.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_registry (
            paper_id TEXT PRIMARY KEY,
            paper_name TEXT,
            title TEXT,
            crop TEXT,
            doi TEXT,
            status TEXT,
            processed_at TEXT,
            pipeline_version TEXT,
            duplicate_flag TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            phase TEXT,
            status TEXT,
            timestamp TEXT,
            details TEXT
        )
    """)
    conn.commit()

    existing = set()
    for row in cursor.execute("SELECT paper_id FROM paper_registry").fetchall():
        existing.add(row[0])

    new_count = 0
    skip_count = 0
    for _, row in ingestion_df.iterrows():
        pid = row["Paper_ID"]
        if pid in existing:
            cursor.execute(
                "UPDATE paper_registry SET duplicate_flag=? WHERE paper_id=?",
                (row["Duplicate_Flag"], pid)
            )
            skip_count += 1
            continue

        cursor.execute("""
            INSERT INTO paper_registry
            (paper_id, paper_name, title, crop, doi, status, processed_at, pipeline_version, duplicate_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, row["Paper_Name"], row["Title"], row["Crop"],
            row["DOI"], row["Status"],
            datetime.now().isoformat(), "1.0.0", row["Duplicate_Flag"]
        ))
        new_count += 1

    conn.commit()

    total = cursor.execute("SELECT COUNT(*) FROM paper_registry").fetchone()[0]
    processed = cursor.execute("SELECT COUNT(*) FROM paper_registry WHERE status='New'").fetchone()[0]
    duplicates = cursor.execute("SELECT COUNT(*) FROM paper_registry WHERE duplicate_flag='Yes'").fetchone()[0]
    conn.close()

    log(f"Paper registry updated: {db_path}")
    log(f"  Total registered: {total}")
    log(f"  Newly processed: {new_count}")
    log(f"  Already registered: {skip_count}")
    log(f"  Total processed: {processed}")
    log(f"  Duplicates: {duplicates}")

    return True


# =====================================================================
# FINAL REPORT
# =====================================================================
def generate_final_report(phases: dict) -> bool:
    log("\n" + "=" * 60)
    log("GENERATING FINAL REPORT")
    log("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = phases

    n_models = len(p.get("training", {}).get("training_results", []))
    best_r2 = ""
    if p.get("training", {}).get("training_results"):
        best = max(p["training"]["training_results"], key=lambda r: r.get("R2", 0))
        best_r2 = f"{best.get('R2', 'N/A')} ({best.get('Model', 'N/A')})"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AAIF Final Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; }}
.header {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; padding: 32px; }}
.header h1 {{ font-size: 28px; }}
.header p {{ opacity: 0.85; margin-top: 4px; }}
.container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}
.card {{ background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 24px; margin-bottom: 20px; }}
.card h2 {{ font-size: 18px; color: #27ae60; margin-bottom: 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
.stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
.stat {{ background: white; border-radius: 10px; padding: 20px; flex: 1; min-width: 150px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.stat .value {{ font-size: 32px; font-weight: 700; color: #27ae60; }}
.stat .label {{ font-size: 13px; color: #7f8c8d; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
th {{ background: #f8f9fa; font-weight: 600; color: #555; }}
tr:hover {{ background: #f0faf4; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.badge-success {{ background: #d4efdf; color: #1e8449; }}
.badge-warning {{ background: #fdebd0; color: #b9770e; }}
.badge-danger {{ background: #fadbd8; color: #c0392b; }}
.footer {{ text-align: center; padding: 24px; color: #95a5a6; font-size: 13px; }}
</style>
</head>
<body>
<div class="header">
<h1>Agentic Agricultural Intelligence Framework</h1>
<p>Final Execution Report — Generated: {now}</p>
</div>
<div class="container">

<div class="stats">
<div class="stat"><div class="value">{p['ingestion_df']['total'] if isinstance(p.get('ingestion_df'), dict) else len(p.get('ingestion_df', []))}</div><div class="label">Papers Processed</div></div>
<div class="stat"><div class="value">{p.get('n_crops', 0)}</div><div class="label">Crops Detected</div></div>
<div class="stat"><div class="value">{p.get('n_rows_extracted', 0)}</div><div class="label">Rows Extracted</div></div>
<div class="stat"><div class="value">{p.get('n_features', 0)}</div><div class="label">Features Created</div></div>
<div class="stat"><div class="value">{n_models}</div><div class="label">Models Trained</div></div>
</div>

<div class="card">
<h2>Phase 1 — Ingestion</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total PDFs found</td><td>{p['ingestion_df']['total'] if isinstance(p.get('ingestion_df'), dict) else len(p.get('ingestion_df', []))}</td></tr>
<tr><td>New papers</td><td>{p['ingestion_df']['new'] if isinstance(p.get('ingestion_df'), dict) else 0}</td></tr>
<tr><td>Duplicates</td><td>{p['ingestion_df']['dups'] if isinstance(p.get('ingestion_df'), dict) else 0}</td></tr>
<tr><td>Ingestion report</td><td>outputs/ingestion_report.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 2-3 — AI Extraction &amp; Universal Schema</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Variables extracted</td><td>{p.get('n_rows_extracted', 0)}</td></tr>
<tr><td>Schema columns</td><td>{p.get('n_schema_cols', 0)}</td></tr>
<tr><td>Universal Schema XLSX</td><td>outputs/Universal_Agricultural_Schema.xlsx</td></tr>
<tr><td>Universal Schema CSV</td><td>outputs/Universal_Agricultural_Schema.csv</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 4 — Validation</h2>
<table>
<tr><th>Issue</th><th>Count</th></tr>
"""
    val = p.get("validation", {})
    for k, v in val.items():
        cnt = len(v) if isinstance(v, (list, dict)) else v
        html += f"<tr><td>{k.replace('_', ' ').title()}</td><td>{cnt}</td></tr>\n"
    html += """</table>
</div>

<div class="card">
<h2>Phase 5 — Feature Engineering</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Features created</td><td>{}</td></tr>
<tr><td>Feature dataset</td><td>outputs/features_dataset.csv</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 6 — Model Training</h2>
<table>
<tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr>
""".format(p.get('n_features', 0))

    for tr in p.get("training", {}).get("training_results", []):
        html += f"<tr><td>{tr.get('Model', '')}</td><td>{tr.get('R2', '')}</td><td>{tr.get('RMSE', '')}</td><td>{tr.get('MAE', '')}</td><td>{tr.get('MAPE', '')}%</td></tr>\n"

    html += f"""</table>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Best model</td><td>{best_r2}</td></tr>
<tr><td>Samples trained</td><td>{p.get('training', {}).get('n_samples', 0)}</td></tr>
<tr><td>Features used</td><td>{p.get('training', {}).get('n_features', 0)}</td></tr>
<tr><td>XGBoost model</td><td>models/xgboost_model.pkl</td></tr>
<tr><td>Regression model</td><td>models/regression_model.pkl</td></tr>
<tr><td>Model metrics</td><td>outputs/model_metrics.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 7 — Fuzzy Logic</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Rules</td><td>{p.get('fuzzy_rules', 0)}</td></tr>
<tr><td>Input variables</td><td>{p.get('fuzzy_inputs', 0)}</td></tr>
<tr><td>Rules file</td><td>fuzzy_logic/fertilizer_rules.yaml</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 8 — Recommendations</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Recommendations generated</td><td>{p.get('n_recommendations', 0)}</td></tr>
<tr><td>Recommendations file</td><td>outputs/recommendations/fertilizer_recommendations.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 9 — Ready Reckoner</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Entries</td><td>{p.get('n_reckoner_entries', 0)}</td></tr>
<tr><td>Ready Reckoner XLSX</td><td>outputs/Ready_Reckoner.xlsx</td></tr>
<tr><td>Ready Reckoner PDF</td><td>outputs/Ready_Reckoner.pdf</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 10 — Continuous Learning</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Database</td><td>database/paper_registry.sqlite</td></tr>
<tr><td>Papers registered</td><td>{p.get('n_registered', 0)}</td></tr>
</table>
</div>

<div class="card">
<h2>Failed Stages</h2>
"""
    failures = p.get("failures", [])
    if failures:
        for f in failures:
            html += f'<p><span class="badge badge-danger">FAILED</span> {f}</p>\n'
    else:
        html += '<p><span class="badge badge-success">All stages completed successfully</span></p>\n'

    html += """
</div>
</div>
<div class="footer">Agentic Agricultural Intelligence Framework (AAIF) v1.0</div>
</body>
</html>"""

    report_path = OUTPUTS_DIR / "AAIF_Final_Report.html"
    Path(report_path).write_text(html, encoding="utf-8")
    log(f"Final report saved: {report_path}")
    return True


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main():
    log("=" * 60)
    log("AAIF PIPELINE STARTED")
    log("=" * 60)
    overall_start = time.time()

    phases_state = {
        "ingestion_df": {"total": 0, "new": 0, "dups": 0},
        "validation": {},
        "training": {},
        "failures": [],
        "n_crops": 0, "n_rows_extracted": 0, "n_schema_cols": 0,
        "n_features": 0, "n_recommendations": 0, "n_reckoner_entries": 0,
        "n_registered": 0, "fuzzy_rules": 0, "fuzzy_inputs": 0,
    }

    # Phase 1
    try:
        ingestion_df = phase1_ingestion()
        phases_state["ingestion_df"] = {
            "total": len(ingestion_df),
            "new": int((ingestion_df["Status"] == "New").sum()),
            "dups": int((ingestion_df["Duplicate_Flag"] == "Yes").sum()),
        }
        phases_state["n_crops"] = ingestion_df["Crop"].nunique()
    except Exception as e:
        log(f"PHASE 1 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 1 (Ingestion): {str(e)}")

    # Phase 2-3
    extract_df = pd.DataFrame()
    rec_df = pd.DataFrame()
    try:
        if ingestion_df is not None and len(ingestion_df) > 0:
            extract_df = phase2_3_extraction(ingestion_df)
            if extract_df is not None and not extract_df.empty:
                phases_state["n_rows_extracted"] = len(extract_df)
                phases_state["n_schema_cols"] = len(extract_df.columns)
        else:
            log("Skipping Phase 2-3: No papers to process")
    except Exception as e:
        log(f"PHASE 2-3 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 2-3 (Extraction): {str(e)}")

    # Phase 4
    try:
        if extract_df is not None and not extract_df.empty:
            val_issues = phase4_validation(extract_df)
            phases_state["validation"] = val_issues
    except Exception as e:
        log(f"PHASE 4 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 4 (Validation): {str(e)}")

    # Phase 5
    feature_df = pd.DataFrame()
    try:
        if extract_df is not None and not extract_df.empty:
            feature_df = phase5_features(extract_df)
            if feature_df is not None and not feature_df.empty:
                phases_state["n_features"] = len(feature_df.columns) - len(extract_df.columns)
    except Exception as e:
        log(f"PHASE 5 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 5 (Features): {str(e)}")

    # Phase 6
    try:
        if feature_df is not None and not feature_df.empty:
            train_results = phase6_training(feature_df)
            phases_state["training"] = train_results
    except Exception as e:
        log(f"PHASE 6 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 6 (Training): {str(e)}")

    # Phase 7
    try:
        if feature_df is not None and not feature_df.empty:
            fuzzy_ok = phase7_fuzzy(feature_df)
            if fuzzy_ok:
                phases_state["fuzzy_rules"] = 14
                phases_state["fuzzy_inputs"] = 10
    except Exception as e:
        log(f"PHASE 7 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 7 (Fuzzy): {str(e)}")

    # Phase 8
    try:
        combined_df = feature_df if feature_df is not None and not feature_df.empty else extract_df
        if combined_df is not None and not combined_df.empty:
            rec_df = phase8_recommendations(combined_df)
            phases_state["n_recommendations"] = len(rec_df) if rec_df is not None else 0
    except Exception as e:
        log(f"PHASE 8 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 8 (Recommendations): {str(e)}")

    # Phase 9
    try:
        if rec_df is not None and not rec_df.empty:
            combined_df2 = feature_df if feature_df is not None and not feature_df.empty else extract_df
            reck_ok = phase9_ready_reckoner(combined_df2, rec_df)
            if reck_ok:
                phases_state["n_reckoner_entries"] = len(rec_df)
    except Exception as e:
        log(f"PHASE 9 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 9 (Ready Reckoner): {str(e)}")

    # Phase 10
    try:
        cl_ok = phase10_continuous_learning(ingestion_df)
        if cl_ok:
            phases_state["n_registered"] = len(ingestion_df)
    except Exception as e:
        log(f"PHASE 10 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 10 (Continuous Learning): {str(e)}")

    # Final Report
    try:
        generate_final_report(phases_state)
    except Exception as e:
        log(f"FINAL REPORT FAILED: {traceback.format_exc()}")

    elapsed = time.time() - overall_start
    log("\n" + "=" * 60)
    log(f"AAIF PIPELINE COMPLETE ({elapsed:.1f}s)")
    log("=" * 60)
    log(f"\n  Outputs directory: {OUTPUTS_DIR}")
    log(f"  Final report: {OUTPUTS_DIR / 'AAIF_Final_Report.html'}")
    log(f"  Failures: {len(phases_state['failures'])}")
    for f in phases_state["failures"]:
        log(f"     - {f}")


if __name__ == "__main__":
    main()
