"""
Shared utilities for the evaluation framework.
"""

import json
import os
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import numpy as np


def _find_poppler_pdftotext() -> Optional[str]:
    """Locate pdftotext binary from common install locations."""
    import shutil
    found = shutil.which("pdftotext")
    if found:
        return found
    for d in [
        r"C:\poppler\poppler-24.08.0\Library\bin",
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        r"C:\Program Files\poppler\Library\bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/homebrew/bin",
    ]:
        for name in ("pdftotext", "pdftotext.exe"):
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate
    return None


BASE_DIR = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = BASE_DIR / "outputs"
EVAL_DIR = BASE_DIR / "evaluation"
REPORTS_DIR = EVAL_DIR / "reports"

PAPERS_DIR = Path("/Users/deepika/Desktop/Ready_Reckoner Table/RPa")

PIPELINE_AGENTS = [
    "ingestion", "schema_mapping", "ontology_mapping",
    "unit_harmonization", "quality_assurance", "feature_engineering",
    "leakage_detection", "encoding", "statistical_diagnostics",
    "model_readiness", "documentation", "export",
]

PAPER_FILES = [
    "Bell pepper.pdf",
    "Black wheat.pdf",
    "Carrot.pdf",
    "Cowpea paper publish.pdf",
    "Spinach.pdf",
]

CROP_FROM_PAPER = {
    "Bell pepper.pdf": "Bell Pepper",
    "Black wheat.pdf": "Black Wheat",
    "Carrot.pdf": "Carrot",
    "Cowpea paper publish.pdf": "Gram",
    "Spinach.pdf": "Spinach",
}

MODEL_LIST = [
    "Multiple Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Support Vector Regression",
    "Decision Tree",
]


def ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_provenance() -> dict:
    path = OUTPUT_DIR / "Pipeline_Provenance.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_dataframe(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    elif p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, engine="openpyxl")
    elif p.suffix == ".sqlite":
        import sqlite3
        conn = sqlite3.connect(p)
        df = pd.read_sql("SELECT * FROM data", conn)
        conn.close()
        return df
    return None


def get_master_df() -> Optional[pd.DataFrame]:
    for f in ["Universal_Agricultural_ML_Master.csv",
              "Universal_Agricultural_ML_Master.parquet",
              "MachineLearning_Dataset.csv"]:
        df = load_dataframe(str(OUTPUT_DIR / f))
        if df is not None:
            return df
    return None


def get_uams_cols() -> list:
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from agri_ai_agent.config.schema import UAMS_COLUMNS
        return UAMS_COLUMNS
    except ImportError:
        return []


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass
    try:
        import pdfminer
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(pdf_path)
    except ImportError:
        pass
    # Try Poppler (pdftotext)
    try:
        poppler_bin = _find_poppler_pdftotext()
        if poppler_bin:
            result = subprocess.run(
                [poppler_bin, "-layout", pdf_path, "-"],
                capture_output=True, text=True, timeout=30,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
    except Exception:
        pass
    try:
        import PyPDF2
        text = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    except ImportError:
        pass
    return ""


def extract_tables_from_pdf(pdf_path: str) -> list:
    tables = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tbls = page.extract_tables()
                for t in tbls:
                    if t and len(t) > 1:
                        tables.append(t)
    except ImportError:
        pass
    return tables


def get_memory_usage() -> dict:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return {
            "rss_mb": proc.memory_info().rss / 1e6,
            "vms_mb": proc.memory_info().vms / 1e6,
            "percent": proc.memory_percent(),
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def get_cpu_usage() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).cpu_percent(interval=0.1)
    except ImportError:
        return 0.0


def write_report(filename: str, content: str):
    ensure_reports_dir()
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


def write_csv(filename: str, data: list[dict]):
    ensure_reports_dir()
    path = REPORTS_DIR / filename
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    return f"{s/60:.2f}m"


def safe_mean(vals: list) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
