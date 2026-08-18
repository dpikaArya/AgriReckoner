"""
AAIF Complete Pipeline Runner — LEGACY (frozen).

This 2000-line procedural monolith produced the validated committed outputs/ and models/
(git tag ``legacy-monolith-v1``). It is kept ONLY as the reference PDF-extraction path until
the agent pipeline is validated on real source PDFs. Do not extend it.

Prefer the maintained engine:  ``agriai run --file data.csv``  (agri_ai_agent.orchestrator).

Retirement criterion: delete this file once the agent pipeline reproduces the golden outputs.
"""

import warnings

warnings.warn(
    "run_aaf_pipeline.py is the frozen legacy runner; use `agriai run` (agri_ai_agent) instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json  # noqa: E402
import re  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "agri_ai_agent"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agri_ai_agent.config.schema import (  # noqa: E402
    NON_FEATURE_COLS,
    POST_HARVEST_VARIABLES,
    UAMS_COLUMNS,
)
from agri_ai_agent.config.settings import AgriAISettings  # noqa: E402
from agri_ai_agent.external_data.column_mapper import to_elemental_basis  # noqa: E402
from agri_ai_agent.extractors.nutrients import to_elemental  # noqa: E402

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
    "bell pepper",
    "capsicum annuum",
    "capsicum",
    "black wheat",
    "triticum aestivum",
    "carrot",
    "daucus carota",
    "cowpea",
    "vigna unguiculata",
    "spinach",
    "spinacia oleracea",
    "rice",
    "oryza sativa",
    "wheat",
    "triticum",
    "maize",
    "zea mays",
    "corn",
    "soybean",
    "glycine max",
    "potato",
    "solanum tuberosum",
    "tomato",
    "solanum lycopersicum",
    "chickpea",
    "cicer arietinum",
    "pigeonpea",
    "cajanus cajan",
    "groundnut",
    "arachis hypogaea",
    "mustard",
    "brassica juncea",
    "sunflower",
    "helianthus annuus",
    "sugarcane",
    "saccharum",
    "cotton",
    "gossypium",
    "onion",
    "allium cepa",
    "chilli",
    "capsicum frutescens",
    "gram",
    "pigeon pea",
    "black gram",
    "green gram",
    "red gram",
    "barley",
    "hordeum vulgare",
    "oat",
    "avena sativa",
    "pearl millet",
    "pennisetum glaucum",
    "finger millet",
    "eleusine coracana",
    "sorghum",
    "sorghum bicolor",
    "ragi",
    "safflower",
    "carthamus tinctorius",
    "linseed",
    "linum usitatissimum",
    "sesame",
    "sesamum indicum",
    "castor",
    "ricinus communis",
    "jute",
    "corchorus",
}

DESIGN_KEYWORDS = [
    "randomized block design",
    "rcbd",
    "randomized complete block design",
    "split plot",
    "split-plot",
    "crd",
    "completely randomized design",
    "latin square",
    "factorial",
    "nested design",
    "strip plot",
    "augmented design",
    "alpha lattice",
    "row column design",
]

MASTER_COLUMN_MAP = {
    "Paper_ID": "Paper_ID",
    "Crop": "Crop",
    "Variety": "Variety",
    "Treatment": "Treatment",
    "Fertilizer": "Fertilizer_Name",
    "Amendment": "Fertilizer_Name",
    "Dose_kg_acre": "Dose",
    "Location": "Location",
    "Country": "Country",
    "State": "State",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Season": "Season",
    "Design": "Design",
    "Experimental_Design": "Design",
    "Replications": "Replications",
    "Replicate": "Replications",
    "Plot_Size_m2": "Plot_Size",
    "Row_Spacing_cm": "Spacing_Row",
    "Plant_Spacing_cm": "Spacing_Plant",
    "Harvest_Stage": "Growth_Duration_Days",
    "Harvest_DAS": "Growth_Duration_Days",
    "Harvest_Days": "Growth_Duration_Days",
    "Rainfall_mm": "Rainfall",
    "Tmax_C": "Temperature_Max",
    "Tmin_C": "Temperature_Min",
    "Soil_pH": "Soil_pH",
    "pH": "Soil_pH",
    "EC": "EC",
    "EC_dS_m": "EC",
    "Organic_Carbon": "Organic_Carbon",
    "Organic_C": "Organic_Carbon",
    "Organic_Carbon_%": "Organic_Carbon",
    "Organic_Matter_%": "Organic_Matter",
    "Available_N": "Nitrogen",
    "Nitrogen_kg_ha": "Nitrogen",
    "Available_P": "Phosphorus",
    "P2O5_kg_ha": "Phosphorus",
    "Available_K": "Potassium",
    "K2O_kg_ha": "Potassium",
    "Sulphur_ppm": "Sulphur",
    "Zn_ppm": "Zinc",
    "Iron_ppm": "Iron",
    "Mn_ppm": "Manganese",
    "Cu_ppm": "Copper",
    "PlantHeight_30_cm": "Plant_Height_30_cm",
    "PlantHeight_60_cm": "Plant_Height_60_cm",
    "PlantHeight_90_cm": "Plant_Height_90_cm",
    "PlantHeight_120_cm": "Plant_Height_cm",
    "Plant_Height_cm": "Plant_Height_cm",
    "LeafArea_30_cm2": "Leaf_Area_30_cm2",
    "LeafArea_60_cm2": "Leaf_Area_60_cm2",
    "LeafArea_90_cm2": "Leaf_Area_90_cm2",
    "LeafArea_120_cm2": "Leaf_Area_cm2",
    "Leaf_Area_cm2": "Leaf_Area_cm2",
    "Branches_60": "Branches",
    "Branches_90": "Branches",
    "Flowers_60": "Flowers",
    "FruitWeight_90_g": "Fruit_Weight",
    "FruitWeight_120_g": "Fruit_Weight",
    "FruitDiameter_90_mm": "Fruit_Diameter_mm",
    "FruitDiameter_120_mm": "Fruit_Diameter_mm",
    "YieldPlot_90_g": "Yield_per_Plot",
    "YieldPlot_120_g": "Yield_per_Plot",
    "Fresh_Weight_g": "Yield_per_Plot",
    "Shoot_Biomass_g": "Shoot_Biomass_g",
    "Root_Biomass_g": "Root_Biomass_g",
    "Shoot_Length_cm": "Shoot_Length_cm",
    "Root_Length_cm": "Root_Length_cm",
    "No_Leaves": "Leaf_Number",
    "Root_Diameter_mm": "Root_Diameter_mm",
    "Chlorophyll_SPAD": "SPAD",
    "SPAD": "SPAD",
    "Tillers": "Tillers",
    "Spike_Length_cm": "Spike_Length",
    "Seeds_per_Spike": "Seeds_per_Spike",
    "Weight_100_Seeds_g": "100_Seed_Weight",
    "Yield_per_Plot_g": "Yield_per_Plot",
    "Dry_Matter_pct": "Dry_Matter",
    "Ash_pct": "Ash",
    "Iron_mgkg": "Iron",
}

MASTER_DATASETS_DIR = BASE_DIR / "data" / "master_datasets"


def load_registered_papers() -> set:
    """Load paper names already registered in the paper_registry.sqlite."""
    db_path = DB_DIR / "paper_registry.sqlite"
    registered = set()
    if db_path.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            try:
                for row in conn.execute("SELECT paper_name FROM paper_registry").fetchall():
                    registered.add(row[0])
            finally:
                conn.close()
        except Exception:
            pass
    return registered


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}")
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('utf-8', errors='replace').decode('utf-8')}")


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
        meta = result.get("metadata", {})
        return meta.get("full_text", "")

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
        return result.get("rows", [])

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
    for _, line in enumerate(lines[:8]):
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
# PHASE 0 — LOAD MASTER DATASETS
# =====================================================================
def phase0_load_master_datasets() -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 0: LOAD MASTER DATASETS")
    log("=" * 60)

    if not MASTER_DATASETS_DIR.exists():
        log(f"Master datasets directory not found: {MASTER_DATASETS_DIR}")
        return pd.DataFrame()

    xlsx_files = sorted(MASTER_DATASETS_DIR.glob("*.xlsx"))
    log(f"Found {len(xlsx_files)} master dataset files")

    all_dfs = []
    for xlsx_path in xlsx_files:
        try:
            df = pd.read_excel(xlsx_path, engine="openpyxl")
            log(f"  {xlsx_path.name}: {len(df)} rows, {len(df.columns)} cols")
            df["_source_file"] = xlsx_path.name
            all_dfs.append(df)
        except Exception as e:
            log(f"  ERROR reading {xlsx_path.name}: {e}")

    if not all_dfs:
        log("No master datasets loaded")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log(f"Combined master datasets: {len(combined)} rows, {len(combined.columns)} cols")

    rename_map = {c: MASTER_COLUMN_MAP[c] for c in combined.columns if c in MASTER_COLUMN_MAP}
    combined = to_elemental_basis(combined, rename_map)
    mapped = combined.rename(columns=rename_map, errors="ignore")
    mapped = mapped.loc[:, ~mapped.columns.duplicated()]

    for col in UAMS_COLUMNS:
        if col not in mapped.columns:
            mapped[col] = pd.NA

    ordered = [c for c in UAMS_COLUMNS if c in mapped.columns]
    extra = [c for c in mapped.columns if c not in UAMS_COLUMNS and c != "_source_file"]
    mapped = mapped[ordered + extra + ["_source_file"]]

    n_filled = sum(1 for c in ordered if mapped[c].notna().any())
    log(f"Schema columns with data: {n_filled}/{len(ordered)}")

    csv_path = OUTPUTS_DIR / "master_datasets_combined.csv"
    mapped.to_csv(csv_path, index=False)
    log(f"Combined master datasets saved: {csv_path}")

    return mapped


# =====================================================================
# PHASE 1 — INGESTION
# =====================================================================
def phase1_ingestion() -> pd.DataFrame:
    log("=" * 60)
    log("PHASE 1: INGESTION (Incremental)")
    log("=" * 60)

    registered_names = load_registered_papers()
    log(f"Previously registered papers in DB: {len(registered_names)}")

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    log(f"Found {len(pdf_files)} PDF files in {PAPERS_DIR}")

    records = []
    all_dois = {}
    all_titles = {}
    skipped_previously = 0

    for pdf_path in pdf_files:
        start = time.time()
        paper_id = generate_paper_id(pdf_path.name)

        if pdf_path.name in registered_names:
            records.append(
                {
                    "Paper_ID": paper_id,
                    "Paper_Name": pdf_path.name,
                    "Title": "(Previously processed)",
                    "Crop": "",
                    "DOI": "",
                    "Pages": 0,
                    "Words": 0,
                    "Status": "Previously Processed",
                    "Duplicate_Flag": "No",
                    "Duplicate_With": "",
                    "Processing_Time_s": 0,
                }
            )
            skipped_previously += 1
            continue

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
        records.append(
            {
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
            }
        )
        log(f"    Crop={crops[0]}, DOI={doi[:40] if doi else 'N/A'}, Status={status}, {duration}s")

    ingestion_df = pd.DataFrame(records)

    xlsx_path = OUTPUTS_DIR / "ingestion_report.xlsx"
    ingestion_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    log(f"Ingestion report saved: {xlsx_path}")

    csv_path = OUTPUTS_DIR / "ingestion_report.csv"
    ingestion_df.to_csv(csv_path, index=False)
    log(f"Ingestion report CSV saved: {csv_path}")

    new_papers = ingestion_df[ingestion_df["Status"] == "New"]["Paper_ID"].tolist()
    prev_processed = int((ingestion_df["Status"] == "Previously Processed").sum())
    log(f"New papers to extract: {len(new_papers)}")
    log(f"Previously processed (skipped extraction): {prev_processed}")
    log(f"Duplicate/intra-run: {len(ingestion_df) - len(new_papers) - prev_processed}")

    return ingestion_df


_UNIT_TAIL_CHARS = 24
_CLAUSE_BREAK = re.compile(r"[,;.]|\band\b|\bwith\b|\bplus\b", re.IGNORECASE)


def _unit_tail(text: str, end: int) -> str:
    """Return the unit written just after a number, stopped before the next clause.

    "phosphorus 60 kg P2O5 ha-1" states the basis in the unit rather than the term, so the
    unit has to be read too. The clause stop keeps a later "and K2O 40" out of this value.
    """
    tail = text[end : end + _UNIT_TAIL_CHARS]
    stop = _CLAUSE_BREAK.search(tail)
    return tail[: stop.start()] if stop else tail


def _elemental_values(pattern: str, text: str) -> list[float]:
    """Return every number ``pattern`` finds in ``text``, on the elemental nutrient basis.

    The pattern must capture the nutrient term first and the number second, so that
    "P2O5 60" is told apart from "phosphorus 60" instead of both landing in one column.
    A term whose basis cannot be read is skipped rather than assumed.
    """
    values: list[float] = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        term = f"{match.group(1)} {_unit_tail(text, match.end(2))}"
        value, ok, _ = to_elemental(float(match.group(2)), term)
        if ok:
            values.append(value)
    return values


# =====================================================================
# PHASE 2-3 — AI EXTRACTION + UNIVERSAL SCHEMA
# =====================================================================
def phase2_3_extraction(ingestion_df: pd.DataFrame) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 2-3: AI EXTRACTION + UNIVERSAL SCHEMA (Incremental)")
    log("=" * 60)

    cached_csv = OUTPUTS_DIR / "Universal_Agricultural_Schema.csv"
    existing_df = None
    if cached_csv.exists():
        try:
            existing_df = pd.read_csv(cached_csv)
            if len(existing_df) > 0:
                log(f"Loaded existing schema: {len(existing_df)} rows from cache")
        except Exception as e:
            log(f"Could not load cached schema: {e}")
            existing_df = None

    new_papers_mask = ingestion_df["Status"] == "New"
    new_papers_df = ingestion_df[new_papers_mask]
    log(f"Papers to extract: {len(new_papers_df)} new")

    if len(new_papers_df) == 0:
        log("No new papers to extract. Using cached schema.")
        if existing_df is not None:
            return existing_df
        return pd.DataFrame()

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    all_rows = []

    for pdf_path in pdf_files:
        paper_id = generate_paper_id(pdf_path.name)
        row = ingestion_df[ingestion_df["Paper_ID"] == paper_id]
        if len(row) == 0:
            continue
        status = row.iloc[0]["Status"]
        if status != "New":
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
                    "randomized block design": "RBD",
                    "rcbd": "RCBD",
                    "randomized complete block design": "RCBD",
                    "split plot": "Split Plot",
                    "split-plot": "Split Plot",
                    "crd": "CRD",
                    "completely randomized design": "CRD",
                    "latin square": "Latin Square",
                    "factorial": "Factorial",
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

        ph_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:soil\s*)?ph\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE
            )
        ]
        ec_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:ec|electrical\s*conductivity)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))",
                text,
                re.IGNORECASE,
            )
        ]
        oc_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:organic\s*carbon|oc)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*%?", text, re.IGNORECASE
            )
        ]
        n_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:available\s*nitrogen|total\s*nitrogen|soil\s*nitrogen|nitrogen\s*content)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?",
                text,
                re.IGNORECASE,
            )
        ]
        if not n_vals:
            n_vals = [
                float(m.group(1))
                for m in re.finditer(
                    r"(?:\bnitrogen)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)",
                    text,
                    re.IGNORECASE,
                )
            ]
        p_vals = _elemental_values(
            r"(available\s*phosphorus|phosphorus|phosphorous|p\s*2\s*o\s*5|p₂o₅)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?",
            text,
        )
        k_vals = _elemental_values(
            r"(available\s*potassium|potassium|k\s*2\s*o|k₂o)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?",
            text,
        )
        tmax_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:max\s*temp|tmax|temperature\s*max|t\.?\s*max)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))",
                text,
                re.IGNORECASE,
            )
        ]
        tmin_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:min\s*temp|tmin|temperature\s*min|t\.?\s*min)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))",
                text,
                re.IGNORECASE,
            )
        ]
        rain_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:rainfall|precipitation|annual\s*rainfall)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:mm)?",
                text,
                re.IGNORECASE,
            )
        ]
        yield_vals = []
        for pat in [
            r"(?:yield|grain\s*yield|seed\s*yield|fruit\s*yield|biological\s*yield|economic\s*yield)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha|q/ha|t|kg|g)",
            r"(?:yield|grain\s*yield)\s*(?:was|:|=)\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha)",
            r"(?:kg\s*ha-1|t\s*ha-1)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))",
            r"((?:\d+\.?\d*|\.\d+))\s*(?:kg\s*ha-1|t\s*ha-1)",
            r"(?:recorded|observed|measured|obtained)\s*(?:yield|grain\s*yield)\s*(?:of|:|=)\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|t/ha)?",
        ]:
            yield_vals.extend([float(m.group(1)) for m in re.finditer(pat, text, re.IGNORECASE)])
        height_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:plant\s*height|height)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*cm",
                text,
                re.IGNORECASE,
            )
        ]
        spad_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:spad|chlorophyll)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))", text, re.IGNORECASE
            )
        ]
        protein_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:protein|protein\s*content|crude\s*protein)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*%",
                text,
                re.IGNORECASE,
            )
        ]

        zn_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:zinc|zn)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:ppm|mg/kg)?", text, re.IGNORECASE
            )
        ]
        fe_vals = [
            float(m.group(1))
            for m in re.finditer(
                r"(?:iron|fe)\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:ppm|mg/kg)?", text, re.IGNORECASE
            )
        ]

        def first_or_none(vals):
            return vals[0] if vals else None

        base_row = {
            "Paper_ID": paper_id,
            "DOI": doi,
            "Crop": crop.split(",")[0].strip() if crop else "",
            "Design": design_match,
            "Replications": replications,
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
        fert_list = [
            "urea",
            "dap",
            "npk",
            "compost",
            "vermicompost",
            "farmyard manure",
            "fym",
            "potassium sulphate",
            "ammonium sulphate",
            "single super phosphate",
            "ssp",
            "muriate of potash",
            "mop",
            "biofertilizer",
            "rhizobium",
            "azotobacter",
            "psb",
            "pgpr",
        ]
        for fert in fert_list:
            pattern = re.compile(re.escape(fert), re.IGNORECASE)
            if pattern.search(text):
                fertilizers_detected.add(fert.title())

        treatment_matches = re.findall(r"(T\d+|T\d+[A-Z]?|Treatment\s*\d+)", text, re.IGNORECASE)
        treatment = ", ".join(sorted(set(treatment_matches))) if treatment_matches else "Control"

        base_row["Treatment"] = treatment
        base_row["Fertilizer_Name"] = ", ".join(sorted(fertilizers_detected))

        all_rows.append(base_row)
        log(
            f"    Extracted: pH={base_row['Soil_pH']}, N={base_row['Nitrogen']}, "
            f"Yield={base_row['Yield_per_Hectare']}, Fert={len(fertilizers_detected)}"
        )

    if not all_rows:
        log("WARNING: No data extracted from new papers")
        if existing_df is not None:
            return existing_df
        return pd.DataFrame()

    new_extract_df = pd.DataFrame(all_rows)

    log("\n  Attempting pdfplumber table extraction for new papers...")
    table_enrichment_rows = []
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        paper_id = generate_paper_id(pdf_path.name)
        row = ingestion_df[ingestion_df["Paper_ID"] == paper_id]
        if len(row) == 0 or row.iloc[0]["Status"] != "New":
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

        for col in [
            "Yield_per_Hectare",
            "Yield_per_Plot",
            "Plant_Height_cm",
            "SPAD",
            "Soil_pH",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Zinc",
            "Iron",
            "Protein",
            "Organic_Carbon",
            "EC",
            "Rainfall",
            "Temperature_Max",
            "Temperature_Min",
            "Fruit_Weight",
            "Fruit_Diameter_mm",
            "Spike_Length",
            "Seeds_per_Spike",
            "100_Seed_Weight",
            "Biomass_Yield",
            "Harvest_Index",
        ]:
            if col in table_df.columns:
                for paper_id in table_df["Paper_ID"].unique():
                    if paper_id in new_extract_df["Paper_ID"].values:
                        mask_reg = new_extract_df["Paper_ID"] == paper_id
                        mask_tbl = table_df["Paper_ID"] == paper_id
                        if col in new_extract_df.columns:
                            reg_val = (
                                new_extract_df.loc[mask_reg, col].iloc[0]
                                if mask_reg.any()
                                else None
                            )
                        else:
                            reg_val = None
                        tbl_vals = table_df.loc[mask_tbl, col].dropna()
                        if (pd.isna(reg_val) or reg_val is None) and len(tbl_vals) > 0:
                            if col in new_extract_df.columns:
                                new_extract_df.loc[mask_reg, col] = tbl_vals.mean()
                            log(
                                f"    Enriched {paper_id}.{col} = {tbl_vals.mean():.2f} (from table)"
                            )

        if "Treatment" in table_df.columns:
            for paper_id in table_df["Paper_ID"].unique():
                if paper_id in new_extract_df["Paper_ID"].values:
                    treatments = (
                        table_df.loc[table_df["Paper_ID"] == paper_id, "Treatment"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    mask = new_extract_df["Paper_ID"] == paper_id
                    if "Treatment" in new_extract_df.columns:
                        existing = (
                            str(new_extract_df.loc[mask, "Treatment"].iloc[0]) if mask.any() else ""
                        )
                        if not existing or existing == "Control" or existing == "nan":
                            new_extract_df.loc[mask, "Treatment"] = ", ".join(
                                sorted(treatments)[:5]
                            )

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_extract_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Paper_ID"], keep="last")
        log(
            f"Merged: {len(existing_df)} existing + {len(new_extract_df)} new = {len(combined)} total"
        )
    else:
        combined = new_extract_df
        log(f"No existing cache. Created schema with {len(combined)} rows")

    for col in UAMS_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    ordered = [c for c in UAMS_COLUMNS if c in combined.columns]
    extra = [c for c in combined.columns if c not in UAMS_COLUMNS]
    combined = combined[ordered + extra]

    xlsx_path = OUTPUTS_DIR / "Universal_Agricultural_Schema.xlsx"
    combined.to_excel(xlsx_path, index=False, engine="openpyxl")
    log(f"Universal Schema XLSX saved: {xlsx_path}")

    csv_path = OUTPUTS_DIR / "Universal_Agricultural_Schema.csv"
    combined.to_csv(csv_path, index=False)
    log(f"Universal Schema CSV saved: {csv_path}")

    log(f"Extraction complete: {len(combined)} rows, {len(combined.columns)} columns")
    return combined


# =====================================================================
# PHASE 4 — VALIDATION
# =====================================================================
def phase4_validation(df: pd.DataFrame) -> dict:
    log("\n" + "=" * 60)
    log("PHASE 4: VALIDATION")
    log("=" * 60)

    issues = {
        "missing_values": {},
        "duplicate_records": 0,
        "impossible_values": [],
        "outliers": [],
        "ocr_mistakes": [],
        "unit_inconsistencies": [],
        "column_consistency": [],
    }

    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = round(missing / len(df) * 100, 1) if len(df) > 0 else 0
        if pct > 0:
            issues["missing_values"][col] = f"{missing}/{len(df)} ({pct}%)"

    issues["duplicate_records"] = int(df.duplicated().sum())

    range_checks = {
        "Soil_pH": (3.0, 10.0),
        "EC": (0, 20),
        "Temperature_Max": (-20, 55),
        "Temperature_Min": (-30, 50),
        "Rainfall": (0, 10000),
        "Yield_per_Hectare": (0, 50000),
        "Plant_Height_cm": (0, 500),
        "SPAD": (0, 80),
        "Protein": (0, 60),
        "Organic_Carbon": (0, 10),
        "Nitrogen": (0, 1000),
        "Phosphorus": (0, 500),
        "Potassium": (0, 1000),
        "Zinc": (0, 100),
        "Iron": (0, 500),
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
            strange = (
                df[col]
                .dropna()
                .apply(lambda x: bool(re.search(r"[^a-zA-Z0-9\s.,;:\-()/%]", str(x))))
                .sum()
            )
            if strange > 0:
                issues["ocr_mistakes"].append(f"{col}: {strange} cells with special chars")

    if "Temperature_Max" in df.columns and "Temperature_Min" in df.columns:
        tmax = pd.to_numeric(df["Temperature_Max"], errors="coerce")
        tmin = pd.to_numeric(df["Temperature_Min"], errors="coerce")
        bad = int((tmax < tmin).sum())
        if bad > 0:
            issues["column_consistency"].append(f"Tmax<Tmin in {bad} rows")

    if "Yield_per_Hectare" in df.columns and "Plant_Height_cm" in df.columns:
        cols_present = [
            c for c in df.columns if c in ["Yield_per_Hectare", "Plant_Height_cm", "SPAD"]
        ]
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
        report_rows.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Non_Null": len(df) - n_missing,
                "Missing": n_missing,
                "Missing_Pct": pct_missing,
                "Unique": int(df[col].nunique()),
                "Issues": "; ".join(col_issues),
            }
        )

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
        fdf["Temp_squared"] = tmean**2
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

    if all(
        c in fdf.columns
        for c in ["Nitrogen", "Phosphorus", "Potassium", "Zinc", "Iron", "Organic_Carbon"]
    ):
        fdf["Soil_Fertility_Index"] = (
            pd.to_numeric(fdf["Nitrogen"], errors="coerce").fillna(0) / 100
            + pd.to_numeric(fdf["Phosphorus"], errors="coerce").fillna(0) / 50
            + pd.to_numeric(fdf["Potassium"], errors="coerce").fillna(0) / 200
            + pd.to_numeric(fdf["Zinc"], errors="coerce").fillna(0) / 5
            + pd.to_numeric(fdf["Iron"], errors="coerce").fillna(0) / 50
            + (pd.to_numeric(fdf["Organic_Carbon"], errors="coerce").fillna(0) * 2)
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
        tmean2 = (
            pd.to_numeric(fdf["Temperature_Max"], errors="coerce")
            + pd.to_numeric(
                fdf["Temperature_Min"]
                if "Temperature_Min" in fdf.columns
                else fdf["Temperature_Max"],
                errors="coerce",
            )
        ) / 2
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
# PHASE 6 — MODEL TRAINING (Enhanced with CV, tuning, multi-model)
# =====================================================================
def phase6_training(df: pd.DataFrame, master_df: pd.DataFrame = None) -> dict:
    log("\n" + "=" * 60)
    log("PHASE 6: MODEL TRAINING")
    log("=" * 60)

    if master_df is not None and not master_df.empty:
        combine_cols = [c for c in df.columns if c in master_df.columns or c in UAMS_COLUMNS]
        all_data = pd.concat([master_df[combine_cols], df[combine_cols]], ignore_index=True)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        log(f"Merged dataset: {len(all_data)} rows (master={len(master_df)}, pdf={len(df)})")
    else:
        all_data = df.copy()

    targets = ["Yield_per_Plot", "Yield_per_Hectare", "Plant_Height_cm", "SPAD", "Shoot_Biomass_g"]
    available_targets = [
        t for t in targets if t in all_data.columns and all_data[t].notna().sum() >= 5
    ]
    if not available_targets:
        for alt in ["Target_Yield", "Yield_per_Plot", "Yield_per_Hectare"]:
            if alt in all_data.columns and all_data[alt].notna().sum() >= 3:
                available_targets = [alt]
                break
    if not available_targets:
        log("WARNING: No target columns with sufficient data")
        return {"error": "No targets"}

    log(f"Training targets: {available_targets}")

    exclude = {
        "Paper_ID",
        "DOI",
        "Crop",
        "Treatment",
        "Fertilizer_Name",
        "Season",
        "Variety",
        "Location",
        "Site",
        "State",
        "Country",
        "Recommendation_Summary",
        "Fuzzy_Summary",
        "_source_file",
        "Source_File",
        "Paper_Name",
        "Title",
        "Status",
        "Duplicate_Flag",
        "Duplicate_With",
        "Processing_Time_s",
    }

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.svm import SVR

    try:
        from xgboost import XGBRegressor

        has_xgb = True
    except ImportError:
        has_xgb = False

    all_results = {}
    all_cv = {}
    all_feature_importance = {}
    all_models = {}

    for target in available_targets:
        log(f"\n  --- Target: {target} ---")
        numeric_df = all_data.select_dtypes(include=[np.number])
        feature_cols = [
            c
            for c in numeric_df.columns
            if c not in exclude
            and c != target
            and not c.startswith("Target_")
            and c not in POST_HARVEST_VARIABLES
            and c not in NON_FEATURE_COLS
        ]

        X = numeric_df[feature_cols].copy()
        y = all_data[target].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        for col in X.columns:
            if X[col].isna().sum() < len(X):
                X[col] = X[col].fillna(X[col].median())

        non_null = [c for c in X.columns if X[c].notna().any()]
        X = X[non_null]
        valid = y.notna()
        X, y = X[valid], y[valid]

        if len(X) < 5:
            log(f"  Insufficient samples ({len(X)}) for {target}")
            continue

        log(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

        if X.shape[1] > X.shape[0] // 2:
            from sklearn.feature_selection import SelectKBest, f_regression

            zero_var = [c for c in X.columns if X[c].std() == 0]
            if zero_var:
                X = X.drop(columns=zero_var, errors="ignore")
                log(f"  Dropped {len(zero_var)} zero-variance features")
            if X.shape[1] > 0:
                k = max(2, X.shape[0] // 3)
                selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
                try:
                    X_arr = selector.fit_transform(X.values, y.values)
                    selected_mask = selector.get_support()
                    selected_features = [
                        f for f, m in zip(X.columns, selected_mask, strict=True) if m
                    ]
                    X = pd.DataFrame(X_arr, columns=selected_features, index=X.index)
                    log(
                        f"  Feature selection: {len(selected_features)}/{len(non_null)} features retained"
                    )
                except Exception as e:
                    log(
                        f"  Feature selection failed ({e}), using top {min(k, X.shape[1])} features by variance"
                    )
                    variances = X.var().sort_values(ascending=False)
                    keep = variances.head(min(k, X.shape[1])).index.tolist()
                    X = X[keep]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models_dict = {}
        param_grids = {}

        models_dict["LinearRegression"] = LinearRegression()
        models_dict["Ridge"] = Ridge(alpha=1.0)
        models_dict["Lasso"] = Lasso(alpha=0.1, max_iter=5000)
        models_dict["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        models_dict["RandomForest"] = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        )
        models_dict["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )

        param_grids["Ridge"] = {"alpha": [0.01, 0.1, 1.0, 10.0]}
        param_grids["Lasso"] = {"alpha": [0.001, 0.01, 0.1, 1.0]}
        param_grids["ElasticNet"] = {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}
        param_grids["RandomForest"] = {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
        param_grids["GradientBoosting"] = {"n_estimators": [50, 100], "max_depth": [2, 3, 4]}

        if has_xgb:
            models_dict["XGBoost"] = XGBRegressor(
                n_estimators=100, max_depth=3, random_state=42, verbosity=0, n_jobs=1
            )
            param_grids["XGBoost"] = {"n_estimators": [50, 100, 200], "max_depth": [2, 3, 5]}

        if len(X) >= 10:
            try:
                models_dict["SVR"] = SVR(kernel="rbf", C=1.0)
                param_grids["SVR"] = {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1]}
            except Exception as e:
                log(f"  SVR not available: {e}")

        target_results = []
        target_cv = {}
        target_fi = {}
        target_models = {}

        for name, model in models_dict.items():
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                log(f"    {name} FIT FAILED: {e}")
                continue

            try:
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = r2_score(y_test, y_pred)
                mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)

                n_cv = min(5, len(X))
                cv_scores = cross_val_score(model, X, y, cv=n_cv, scoring="r2")
                cv_mean = round(float(cv_scores.mean()), 4)
                cv_std = round(float(cv_scores.std()), 4)

                if name in param_grids and len(X) >= 10:
                    try:
                        gs = GridSearchCV(
                            model,
                            param_grids[name],
                            cv=min(3, len(X_train)),
                            scoring="r2",
                            n_jobs=-1,
                            error_score="raise",
                        )
                        gs.fit(X_train, y_train)
                        best_params = gs.best_params_
                        best_score = round(float(gs.best_score_), 4)
                    except Exception as gs_e:
                        log(f"  GridSearchCV failed for {name}: {gs_e}")
                        best_params = {}
                        best_score = cv_mean
                else:
                    best_params = {}
                    best_score = cv_mean

                fi = {}
                if hasattr(model, "feature_importances_"):
                    fi = dict(
                        zip(
                            list(X.columns),
                            [round(float(x), 4) for x in model.feature_importances_],
                            strict=True,
                        )
                    )
                elif hasattr(model, "coef_"):
                    fi = dict(
                        zip(list(X.columns), [round(float(x), 4) for x in model.coef_], strict=True)
                    )

                target_results.append(
                    {
                        "Model": name,
                        "Target": target,
                        "MAE": round(mae, 4),
                        "RMSE": round(rmse, 4),
                        "R2": round(r2, 4),
                        "MAPE": round(mape, 2),
                        "CV_Mean_R2": cv_mean,
                        "CV_Std_R2": cv_std,
                        "Best_CV_R2": best_score,
                        "Best_Params": str(best_params),
                    }
                )
                target_cv[name] = {"mean_r2": cv_mean, "std_r2": cv_std}
                target_fi[name] = fi
                target_models[name] = model

                log(f"    {name}: R2={r2:.4f}, RMSE={rmse:.2f}, CV_R2={cv_mean:.4f}+/-{cv_std:.4f}")
            except Exception as e:
                log(f"    {name} FAILED: {e}")

        if target_results:

            def _best_score(r):
                cv = r.get("CV_Mean_R2", -999)
                return cv if cv == cv else r.get("R2", -999)  # NaN check

            best = max(target_results, key=_best_score)
            best_name = best["Model"]
            best_model = target_models.get(best_name)
            if best_model:
                best_model.fit(X, y)

            fi_df = pd.DataFrame(
                [
                    {"Feature": k, "Importance": v, "Target": target}
                    for k, v in sorted(
                        target_fi.get(best_name, {}).items(), key=lambda x: -abs(x[1])
                    )
                ]
            )
            fi_df.to_csv(OUTPUTS_DIR / f"feature_importance_{target}.csv", index=False)

            all_results[target] = target_results
            all_cv[target] = target_cv
            all_feature_importance[target] = target_fi.get(best_name, {})
            all_models[target] = target_models

    results_flat = []
    for _, results in all_results.items():
        results_flat.extend(results)

    if results_flat:
        results_df = pd.DataFrame(results_flat)
        r_xlsx = OUTPUTS_DIR / "model_metrics.xlsx"
        results_df.to_excel(r_xlsx, index=False, engine="openpyxl")
        log(f"Model metrics saved: {r_xlsx}")

        best_overall = max(results_flat, key=lambda r: r.get("CV_Mean_R2", 0))
        log(
            f"\n  BEST MODEL: {best_overall['Model']} on {best_overall['Target']} "
            f"(CV R2={best_overall['CV_Mean_R2']})"
        )

        import joblib

        for target, models in all_models.items():
            best_name = max(all_results[target], key=lambda r: r.get("CV_Mean_R2", 0))["Model"]
            if best_name in models:
                path = MODELS_DIR / f"best_model_{target}.pkl"
                joblib.dump(models[best_name], path)
                log(f"  Saved: {path}")

    return {
        "training_results": results_flat,
        "cv_scores": all_cv,
        "feature_importance": all_feature_importance,
        "targets_trained": available_targets,
        "n_samples": len(all_data),
        "models": {t: m for t, m in all_models.items()},
    }


# =====================================================================
# PHASE 7 — FUZZY LOGIC
# =====================================================================
def phase7_fuzzy(df: pd.DataFrame) -> bool:
    log("\n" + "=" * 60)
    log("PHASE 7: FUZZY LOGIC")
    log("=" * 60)

    from agri_ai_agent.rules.membership_functions import fuzzify

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
# PHASE 8 — RECOMMENDATION AGENT (Enhanced with model predictions)
# =====================================================================
def phase8_recommendations(df: pd.DataFrame, training_metrics: dict = None) -> pd.DataFrame:
    log("\n" + "=" * 60)
    log("PHASE 8: RECOMMENDATION AGENT")
    log("=" * 60)

    trained_models = {}
    if training_metrics and "models" in training_metrics:
        for target, model_dict in training_metrics["models"].items():
            target_results = [
                r for r in training_metrics.get("training_results", []) if r.get("Target") == target
            ]
            if target_results:
                best_name = max(
                    target_results,
                    key=lambda r: (
                        r.get("CV_Mean_R2", -999)
                        if not np.isnan(r.get("CV_Mean_R2", -999))
                        else r.get("R2", -999)
                    ),
                )["Model"]
                if best_name in model_dict:
                    trained_models[target] = model_dict[best_name]

    rdf = df.copy()
    recs = []

    for idx in rdf.index:
        row = rdf.loc[idx]

        n = (
            pd.to_numeric(row.get("Nitrogen"), errors="coerce")
            if pd.notna(row.get("Nitrogen"))
            else None
        )
        p = (
            pd.to_numeric(row.get("Phosphorus"), errors="coerce")
            if pd.notna(row.get("Phosphorus"))
            else None
        )
        k = (
            pd.to_numeric(row.get("Potassium"), errors="coerce")
            if pd.notna(row.get("Potassium"))
            else None
        )
        ph = (
            pd.to_numeric(row.get("Soil_pH"), errors="coerce")
            if pd.notna(row.get("Soil_pH"))
            else None
        )
        rainfall = (
            pd.to_numeric(row.get("Rainfall"), errors="coerce")
            if pd.notna(row.get("Rainfall"))
            else None
        )
        yield_val = (
            pd.to_numeric(row.get("Yield_per_Hectare"), errors="coerce")
            if pd.notna(row.get("Yield_per_Hectare"))
            else None
        )
        oc = (
            pd.to_numeric(row.get("Organic_Carbon"), errors="coerce")
            if pd.notna(row.get("Organic_Carbon"))
            else None
        )
        zn = pd.to_numeric(row.get("Zinc"), errors="coerce") if pd.notna(row.get("Zinc")) else None
        crop = str(row.get("Crop", "Unknown"))

        predicted_yield = None
        if trained_models:
            for _, model in trained_models.items():
                try:
                    feature_names = (
                        model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
                    )
                    if feature_names:
                        feat_vals = []
                        for fn in feature_names:
                            val = row.get(fn)
                            feat_vals.append(
                                pd.to_numeric(val, errors="coerce") if pd.notna(val) else 0
                            )
                        if any(v != 0 for v in feat_vals):
                            predicted_yield = round(float(model.predict([feat_vals])[0]), 2)
                            break
                except Exception as e:
                    log(f"  Model predict failed: {e}")

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

        # Only a model prediction can support an expected yield. Inflating the observed
        # yield by a flat 15% invents both the figure and the gain: it is arithmetic on
        # the input, not evidence about the fertiliser, and it was previously reported to
        # users as "High confidence".
        expected_yield = predicted_yield if predicted_yield else None
        expected_increase = (
            round(expected_yield - yield_val, 2)
            if expected_yield is not None and yield_val
            else None
        )

        # Confidence must fall as evidence thins. The previous ladder started at 0.75 and
        # only ever lowered it, so a row with no soil data at all kept 0.75 ("High") while
        # a row with partial data dropped to 0.65 ("Medium") — less data read as more
        # confident.
        if predicted_yield and n is not None and p is not None and k is not None:
            conf_score = 0.90
        elif predicted_yield:
            conf_score = 0.70
        elif n is not None and p is not None and k is not None and ph is not None:
            conf_score = 0.45
        elif n is not None or p is not None or k is not None:
            conf_score = 0.30
        else:
            conf_score = 0.15

        conf_label = "High" if conf_score >= 0.7 else "Medium" if conf_score >= 0.4 else "Low"

        alt_treatments = []
        if best_fert != "Balanced NPK (15-15-15)":
            alt_treatments.append("Balanced NPK (15-15-15)")
        if "Urea" in best_fert:
            alt_treatments.append("DAP + MOP")
        if "DAP" in best_fert:
            alt_treatments.append("SSP + Urea")
        alt_treatments.append("Organic compost")

        recs.append(
            {
                "Crop": crop,
                "Paper_ID": row.get("Paper_ID", ""),
                "Treatment": row.get("Treatment", ""),
                "Best_Treatment": best_fert,
                "Predicted_Yield": predicted_yield,
                "Expected_Yield_kg_ha": expected_yield,
                "Yield_Increase_kg_ha": expected_increase,
                "Confidence_Score": conf_score,
                "Confidence_Label": conf_label,
                "Alternative_Treatments": "; ".join(alt_treatments[:3]),
                "Recommendation": n_rec,
                "Application_Interval": interval,
            }
        )

    rec_df = pd.DataFrame(recs)
    rec_path = OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.xlsx"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_excel(rec_path, index=False, engine="openpyxl")
    log(f"Recommendations saved: {rec_path}")

    rec_csv = OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.csv"
    rec_df.to_csv(rec_csv, index=False)
    log(f"Recommendations CSV saved: {rec_csv}")

    best_by_crop = rec_df.groupby("Crop").first().reset_index()
    for _, brow in best_by_crop.iterrows():
        log(f"  {brow['Crop']}: {brow['Best_Treatment']} (Conf: {brow['Confidence_Label']})")

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
            pr.get("Yield_per_Hectare")
        else:
            ph = n = p = k = tmax = tmin = rainfall = oc = None

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
            "Urea": "Soil application (band placement)",
            "DAP": "Soil application at sowing",
            "MOP": "Soil application",
            "NPK": "Broadcast + incorporation",
            "Compost": "Broadcast + incorporation",
            "Zinc": "Foliar spray",
            "SSP": "Soil application at sowing",
            "Ammonium": "Soil application",
            "Dolomite": "Soil application (broadcast)",
        }
        app_method = "Soil application"
        for kw, method in method_map.items():
            if kw.lower() in str(rec.get("Best_Treatment", "")).lower():
                app_method = method
                break

        reckoner_rows.append(
            {
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
            }
        )

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
        tbody += (
            "<tr>"
            + "".join(
                f"<td>{v}</td>" if not isinstance(v, float) else f"<td>{v:.2f}</td>" for v in row
            )
            + "</tr>"
        )
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
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {len(df)} entries</p>
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
                (row["Duplicate_Flag"], pid),
            )
            skip_count += 1
            continue

        cursor.execute(
            """
            INSERT INTO paper_registry
            (paper_id, paper_name, title, crop, doi, status, processed_at, pipeline_version, duplicate_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pid,
                row["Paper_Name"],
                row["Title"],
                row["Crop"],
                row["DOI"],
                row["Status"],
                datetime.now().isoformat(),
                "1.0.0",
                row["Duplicate_Flag"],
            ),
        )
        new_count += 1

    conn.commit()

    total = cursor.execute("SELECT COUNT(*) FROM paper_registry").fetchone()[0]
    processed = cursor.execute("SELECT COUNT(*) FROM paper_registry WHERE status='New'").fetchone()[
        0
    ]
    duplicates = cursor.execute(
        "SELECT COUNT(*) FROM paper_registry WHERE duplicate_flag='Yes'"
    ).fetchone()[0]
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
<p>Final Execution Report — Incremental Mode — Generated: {now}</p>
</div>
<div class="container">

<div class="stats">
<div class="stat"><div class="value">{p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else len(p.get("ingestion_df", []))}</div><div class="label">Total PDFs</div></div>
<div class="stat"><div class="value">{p.get("n_newly_extracted", 0)}</div><div class="label">New Papers Extracted</div></div>
<div class="stat"><div class="value">{p.get("n_previously_processed", 0)}</div><div class="label">Previously Processed (Skipped)</div></div>
<div class="stat"><div class="value">{p.get("n_rows_extracted", 0)}</div><div class="label">Total Schema Rows</div></div>
<div class="stat"><div class="value">{p.get("n_features", 0)}</div><div class="label">Features Created</div></div>
<div class="stat"><div class="value">{n_models}</div><div class="label">Models Trained</div></div>
</div>

<div class="card">
<h2>Phase 1 — Ingestion (Incremental)</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total PDFs found</td><td>{p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else len(p.get("ingestion_df", []))}</td></tr>
<tr><td>New papers (to extract)</td><td>{p["ingestion_df"]["new"] if isinstance(p.get("ingestion_df"), dict) else 0}</td></tr>
<tr><td>Previously processed (skipped)</td><td>{p["ingestion_df"].get("prev_processed", 0) if isinstance(p.get("ingestion_df"), dict) else 0}</td></tr>
<tr><td>Duplicates (intra-run)</td><td>{p["ingestion_df"]["dups"] if isinstance(p.get("ingestion_df"), dict) else 0}</td></tr>
<tr><td>Ingestion report</td><td>outputs/ingestion_report.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 2-3 — AI Extraction &amp; Universal Schema (Incremental)</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>New rows extracted</td><td>{p.get("n_newly_extracted", 0)}</td></tr>
<tr><td>Total schema rows (merged)</td><td>{p.get("n_rows_extracted", 0)}</td></tr>
<tr><td>Schema columns</td><td>{p.get("n_schema_cols", 0)}</td></tr>
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
""".format(p.get("n_features", 0))

    for tr in p.get("training", {}).get("training_results", []):
        html += f"<tr><td>{tr.get('Model', '')}</td><td>{tr.get('R2', '')}</td><td>{tr.get('RMSE', '')}</td><td>{tr.get('MAE', '')}</td><td>{tr.get('MAPE', '')}%</td></tr>\n"

    html += f"""</table>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Best model</td><td>{best_r2}</td></tr>
<tr><td>Samples trained</td><td>{p.get("training", {}).get("n_samples", 0)}</td></tr>
<tr><td>Features used</td><td>{p.get("training", {}).get("n_features", 0)}</td></tr>
<tr><td>XGBoost model</td><td>models/xgboost_model.pkl</td></tr>
<tr><td>Regression model</td><td>models/regression_model.pkl</td></tr>
<tr><td>Model metrics</td><td>outputs/model_metrics.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 7 — Fuzzy Logic</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Rules</td><td>{p.get("fuzzy_rules", 0)}</td></tr>
<tr><td>Input variables</td><td>{p.get("fuzzy_inputs", 0)}</td></tr>
<tr><td>Rules file</td><td>fuzzy_logic/fertilizer_rules.yaml</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 8 — Recommendations</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Recommendations generated</td><td>{p.get("n_recommendations", 0)}</td></tr>
<tr><td>Recommendations file</td><td>outputs/recommendations/fertilizer_recommendations.xlsx</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 9 — Ready Reckoner</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Entries</td><td>{p.get("n_reckoner_entries", 0)}</td></tr>
<tr><td>Ready Reckoner XLSX</td><td>outputs/Ready_Reckoner.xlsx</td></tr>
<tr><td>Ready Reckoner PDF</td><td>outputs/Ready_Reckoner.pdf</td></tr>
</table>
</div>

<div class="card">
<h2>Phase 10 — Continuous Learning</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Database</td><td>database/paper_registry.sqlite</td></tr>
<tr><td>Papers registered</td><td>{p.get("n_registered", 0)}</td></tr>
</table>
</div>

<div class="card" style="border-left: 4px solid #2980b9;">
<h2>Efficiency Improvement Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total PDFs in folder</td><td>{p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else 0}</td></tr>
<tr><td>Previously processed (skipped)</td><td>{p.get("n_previously_processed", 0)}</td></tr>
<tr><td>New papers extracted this run</td><td>{p.get("n_newly_extracted", 0)}</td></tr>
<tr><td>Skipped ratio</td><td>{round(p.get("n_previously_processed", 0) / max(1, (p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else 1)) * 100, 1)}%</td></tr>
<tr><td>PDF extraction avoided</td><td>{p.get("n_previously_processed", 0)} PDFs x ~30s avg = ~{round(p.get("n_previously_processed", 0) * 30 / 60, 1)} min saved</td></tr>
<tr><td>Time efficiency gain</td><td>Only {p.get("n_newly_extracted", 0)} of {p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else 0} PDFs required extraction ({round(p.get("n_newly_extracted", 0) / max(1, (p["ingestion_df"]["total"] if isinstance(p.get("ingestion_df"), dict) else 1)) * 100, 1)}% of total)</td></tr>
<tr><td>Schema growth</td><td>{p.get("n_rows_extracted", 0)} total rows (incremental merge)</td></tr>
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
        html += (
            '<p><span class="badge badge-success">All stages completed successfully</span></p>\n'
        )

    html += """
</div>
</div>
<div class="footer">Agentic Agricultural Intelligence Framework (AAIF) v2.0 — Incremental Processing Enabled</div>
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
    log("AAIF PIPELINE STARTED (Incremental Mode)")
    log("=" * 60)
    overall_start = time.time()

    load_registered_papers()

    phases_state = {
        "ingestion_df": {"total": 0, "new": 0, "dups": 0, "prev_processed": 0},
        "validation": {},
        "training": {},
        "failures": [],
        "n_crops": 0,
        "n_rows_extracted": 0,
        "n_schema_cols": 0,
        "n_features": 0,
        "n_recommendations": 0,
        "n_reckoner_entries": 0,
        "n_registered": 0,
        "fuzzy_rules": 0,
        "fuzzy_inputs": 0,
        "n_previously_processed": 0,
        "n_newly_extracted": 0,
    }

    master_df = pd.DataFrame()
    try:
        master_df = phase0_load_master_datasets()
        if not master_df.empty:
            phases_state["n_master_rows"] = len(master_df)
    except Exception as e:
        log(f"PHASE 0 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 0 (Master Datasets): {str(e)}")

    ingestion_df = pd.DataFrame()
    try:
        ingestion_df = phase1_ingestion()
        prev_proc = int((ingestion_df["Status"] == "Previously Processed").sum())
        new_count = int((ingestion_df["Status"] == "New").sum())
        phases_state["ingestion_df"] = {
            "total": len(ingestion_df),
            "new": new_count,
            "dups": int((ingestion_df["Duplicate_Flag"] == "Yes").sum()),
            "prev_processed": prev_proc,
        }
        phases_state["n_previously_processed"] = prev_proc
        phases_state["n_newly_extracted"] = new_count
        phases_state["n_crops"] = ingestion_df["Crop"].nunique()
    except Exception as e:
        log(f"PHASE 1 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 1 (Ingestion): {str(e)}")

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

    try:
        if extract_df is not None and not extract_df.empty:
            val_issues = phase4_validation(extract_df)
            phases_state["validation"] = val_issues
    except Exception as e:
        log(f"PHASE 4 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 4 (Validation): {str(e)}")

    try:
        validated_data = []
        if not master_df.empty:
            validated_data.extend(master_df.to_dict(orient="records"))
        if extract_df is not None and not extract_df.empty:
            validated_data.extend(extract_df.to_dict(orient="records"))
        val_path = OUTPUTS_DIR / "Validated_Extractions.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validated_data, f, indent=2, default=str)
        log(f"Validated extractions saved: {val_path} ({len(validated_data)} records)")
    except Exception as e:
        log(f"WARNING: Could not save Validated_Extractions.json: {e}")

    feature_df = pd.DataFrame()
    try:
        if extract_df is not None and not extract_df.empty:
            feature_df = phase5_features(extract_df)
            if feature_df is not None and not feature_df.empty:
                phases_state["n_features"] = len(feature_df.columns) - len(extract_df.columns)
    except Exception as e:
        log(f"PHASE 5 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 5 (Features): {str(e)}")

    training_metrics = {}
    try:
        train_data = feature_df if feature_df is not None and not feature_df.empty else extract_df
        if train_data is not None and not train_data.empty:
            training_metrics = phase6_training(train_data, master_df)
            phases_state["training"] = training_metrics
    except Exception as e:
        log(f"PHASE 6 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 6 (Training): {str(e)}")

    try:
        if feature_df is not None and not feature_df.empty:
            fuzzy_ok = phase7_fuzzy(feature_df)
            if fuzzy_ok:
                phases_state["fuzzy_rules"] = 14
                phases_state["fuzzy_inputs"] = 10
    except Exception as e:
        log(f"PHASE 7 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 7 (Fuzzy): {str(e)}")

    try:
        combined_df = feature_df if feature_df is not None and not feature_df.empty else extract_df
        if not master_df.empty:
            combine_cols = [c for c in combined_df.columns if c in master_df.columns]
            if combine_cols:
                combined_df = pd.concat(
                    [master_df[combine_cols], combined_df[combine_cols]], ignore_index=True
                )
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        if combined_df is not None and not combined_df.empty:
            rec_df = phase8_recommendations(combined_df, training_metrics)
            phases_state["n_recommendations"] = len(rec_df) if rec_df is not None else 0
    except Exception as e:
        log(f"PHASE 8 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 8 (Recommendations): {str(e)}")

    try:
        if rec_df is not None and not rec_df.empty:
            combined_df2 = (
                feature_df if feature_df is not None and not feature_df.empty else extract_df
            )
            reck_ok = phase9_ready_reckoner(combined_df2, rec_df)
            if reck_ok:
                phases_state["n_reckoner_entries"] = len(rec_df)
    except Exception as e:
        log(f"PHASE 9 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 9 (Ready Reckoner): {str(e)}")

    try:
        cl_ok = phase10_continuous_learning(ingestion_df)
        if cl_ok:
            phases_state["n_registered"] = len(ingestion_df)
    except Exception as e:
        log(f"PHASE 10 FAILED: {traceback.format_exc()}")
        phases_state["failures"].append(f"Phase 10 (Continuous Learning): {str(e)}")

    try:
        generate_final_report(phases_state)
    except Exception:
        log(f"FINAL REPORT FAILED: {traceback.format_exc()}")

    elapsed = time.time() - overall_start
    prev = phases_state.get("n_previously_processed", 0)
    new_ext = phases_state.get("n_newly_extracted", 0)
    total_pdfs = (
        phases_state["ingestion_df"]["total"]
        if isinstance(phases_state.get("ingestion_df"), dict)
        else 0
    )
    skip_pct = round(prev / max(1, total_pdfs) * 100, 1)

    log("\n" + "=" * 60)
    log(f"AAIF PIPELINE COMPLETE ({elapsed:.1f}s)")
    log("=" * 60)
    log("\n  EFFICIENCY SUMMARY:")
    log(f"    Total PDFs scanned:     {total_pdfs}")
    log(f"    Previously processed:   {prev} (skipped extraction)")
    log(f"    New papers extracted:   {new_ext}")
    log(f"    Skipped ratio:          {skip_pct}%")
    log(f"    Time saved (est.):      ~{prev * 30 / 60:.1f} min (PDF extraction avoided)")
    log(f"    Schema rows (merged):   {phases_state.get('n_rows_extracted', 0)}")
    log(f"\n  Outputs directory: {OUTPUTS_DIR}")
    log(f"  Final report: {OUTPUTS_DIR / 'AAIF_Final_Report.html'}")
    log(f"  Failures: {len(phases_state['failures'])}")
    for f in phases_state["failures"]:
        log(f"     - {f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AAIF Complete Pipeline Runner")
    parser.add_argument(
        "--papers",
        "-p",
        type=str,
        default=None,
        help="Directory containing research papers (PDFs). Default: Data ADES",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-extraction by removing cached outputs"
    )
    args, _ = parser.parse_known_args()

    if args.papers:
        _cli_papers_dir = Path(args.papers).resolve()
        if not _cli_papers_dir.exists():
            log(f"ERROR: Papers directory not found: {_cli_papers_dir}")
            sys.exit(1)
        PAPERS_DIR = _cli_papers_dir
        log(f"Using papers directory: {PAPERS_DIR}")

    if args.force:
        for f in [
            "Universal_Agricultural_Schema.csv",
            "Universal_Agricultural_Schema.xlsx",
            "Validated_Extractions.json",
            "features_dataset.csv",
            "model_metrics.xlsx",
            "feature_importance_Yield_per_Plot.csv",
        ]:
            p = OUTPUTS_DIR / f
            if p.exists():
                p.unlink()
                log(f"Removed cached: {p.name}")
    main()
