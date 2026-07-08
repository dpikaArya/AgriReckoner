"""
File I/O utilities for dataset ingestion and export.
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import numpy as np


def detect_encoding(filepath: Path, n_bytes: int = 10000) -> str:
    try:
        import chardet
        raw = filepath.read_bytes()[:n_bytes]
        result = chardet.detect(raw)
        return result.get("encoding", "utf-8")
    except ImportError:
        return "utf-8"


def detect_delimiter(filepath: Path, n_lines: int = 5) -> str:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            sample = "".join(f.readline() for _ in range(n_lines))
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def detect_worksheet(filepath: Path) -> Optional[str]:
    if filepath.suffix.lower() in (".xlsx", ".xls"):
        xls = pd.ExcelFile(filepath, engine="openpyxl")
        sheets = xls.sheet_names
        if len(sheets) == 1:
            return sheets[0]
        for kw in ["data", "sheet1", "master", "dataset"]:
            for s in sheets:
                if kw in s.lower():
                    return s
        return sheets[0]
    return None


def read_dataset(
    filepath: Union[str, Path],
    sheet_name: Optional[str] = None,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        enc = encoding or detect_encoding(filepath)
        sep = delimiter or detect_delimiter(filepath)
        return pd.read_csv(filepath, encoding=enc, sep=sep, low_memory=False)

    elif suffix == ".tsv":
        return pd.read_csv(filepath, sep="\t", encoding=encoding or "utf-8", low_memory=False)

    elif suffix in (".xlsx", ".xls"):
        sn = sheet_name or detect_worksheet(filepath) or 0
        return pd.read_excel(filepath, sheet_name=sn, engine="openpyxl")

    elif suffix == ".parquet":
        return pd.read_parquet(filepath)

    elif suffix == ".json":
        return pd.read_json(filepath)

    elif suffix == ".db":
        import sqlite3
        conn = sqlite3.connect(str(filepath))
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if len(tables):
            return pd.read_sql(f"SELECT * FROM [{tables.iloc[0, 0]}]", conn)
        conn.close()
        return pd.DataFrame()

    elif suffix == ".duckdb":
        try:
            import duckdb
            con = duckdb.connect(str(filepath))
            tables = con.execute("SELECT table_name FROM information_schema.tables").fetchdf()
            if len(tables):
                return con.execute(f"SELECT * FROM \"{tables.iloc[0, 0]}\"").fetchdf()
            con.close()
            return pd.DataFrame()
        except ImportError:
            raise RuntimeError("duckdb package required to read .duckdb files")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def write_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> Path:
    path = Path(path)
    format = format or path.suffix.lower().lstrip(".")

    if not path.suffix:
        path = path.with_suffix(f".{format}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig", **kwargs)
    elif format == "tsv":
        df.to_csv(path, index=False, sep="\t", encoding="utf-8-sig", **kwargs)
    elif format in ("xlsx", "xls"):
        df.to_excel(path, index=False, engine="openpyxl", **kwargs)
    elif format == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif format == "json":
        df.to_json(path, orient="records", **kwargs)
    elif format == "sqlite":
        import sqlite3
        conn = sqlite3.connect(str(path))
        df.to_sql("data", conn, if_exists="replace", index=False)
        conn.close()
    elif format == "duckdb":
        try:
            import duckdb
            con = duckdb.connect(str(path))
            con.execute("CREATE TABLE data AS SELECT * FROM df")
            con.close()
        except ImportError:
            raise RuntimeError("duckdb package required to write .duckdb files")
    else:
        raise ValueError(f"Unsupported export format: {format}")

    return path.resolve()
