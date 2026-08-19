"""Persistent storage and output writing for the literature pipeline.

``BibliographyDB`` owns the normalized SQLite database (papers, authors,
sources, references, citations, PDFs, crawl log).  The remaining writers
produce the parquet / csv / xlsx / docx / html deliverables.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agri_ai_agent.literature.models import LiteratureRecord, SyncResult

logger = logging.getLogger(__name__)

# Fixed timestamps used when writing XLSX so that identical inputs always
# produce byte-identical files (reproducible builds).  Without this, openpyxl
# stamps docProps/core.xml and every zip entry with the current wall-clock time.
_XLSX_CREATOR = "AAIF"
_XLSX_STAMP = datetime(2000, 1, 1, tzinfo=timezone.utc)
_XLSX_STAMP_ISO = b"2000-01-01T00:00:00Z"
_XLSX_ZIP_DATE = (2000, 1, 1, 0, 0, 0)
_CORE_TS_RE = re.compile(
    rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)"
)


# ---------------------------------------------------------------------- #
# SQLite bibliography database
# ---------------------------------------------------------------------- #
class BibliographyDB:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id            TEXT PRIMARY KEY,
                    doi                 TEXT,
                    title               TEXT,
                    abstract            TEXT,
                    year                INTEGER,
                    journal             TEXT,
                    publisher           TEXT,
                    publication_type    TEXT,
                    language            TEXT,
                    citations_count     INTEGER,
                    license             TEXT,
                    landing_page        TEXT,
                    is_original_study   INTEGER,
                    quality_score       REAL,
                    valid_doi           INTEGER,
                    experimental_design TEXT,
                    study_variables     TEXT,
                    crop_terms          TEXT,
                    country             TEXT,
                    created_at          TEXT,
                    updated_at          TEXT
                );
                CREATE TABLE IF NOT EXISTS authors (
                    author_id   TEXT PRIMARY KEY,
                    full_name   TEXT,
                    given_name  TEXT,
                    family_name TEXT,
                    orcid       TEXT,
                    affiliation TEXT
                );
                CREATE TABLE IF NOT EXISTS paper_authors (
                    paper_id    TEXT NOT NULL,
                    author_id   TEXT NOT NULL,
                    author_order INTEGER,
                    PRIMARY KEY (paper_id, author_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_sources (
                    source     TEXT NOT NULL,
                    source_id  TEXT NOT NULL,
                    paper_id   TEXT NOT NULL,
                    raw_json   TEXT,
                    fetched_at TEXT,
                    PRIMARY KEY (source, source_id),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS references_ (
                    paper_id     TEXT NOT NULL,
                    ref_doi      TEXT,
                    ref_paper_id TEXT,
                    source       TEXT,
                    relation     TEXT DEFAULT 'cites',
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS citations (
                    paper_id       TEXT NOT NULL,
                    citing_doi     TEXT,
                    citing_paper_id TEXT,
                    source         TEXT,
                    relation       TEXT DEFAULT 'cited-by',
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS pdfs (
                    paper_id      TEXT NOT NULL,
                    url           TEXT NOT NULL,
                    source_kind   TEXT,
                    license       TEXT,
                    content_type  TEXT,
                    verified      INTEGER DEFAULT 0,
                    source_repository TEXT,
                    added_at      TEXT,
                    PRIMARY KEY (paper_id, url),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE TABLE IF NOT EXISTS crawl_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    connector     TEXT,
                    started_at    TEXT,
                    completed_at  TEXT,
                    fetched       INTEGER DEFAULT 0,
                    new_records   INTEGER DEFAULT 0,
                    skipped       INTEGER DEFAULT 0,
                    errors        TEXT
                );
                """
            )

    # ------------------------------------------------------------------ #
    # papers
    # ------------------------------------------------------------------ #
    def upsert_paper(self, record: LiteratureRecord) -> None:
        now = record.fetched_at
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    paper_id, doi, title, abstract, year, journal, publisher,
                    publication_type, language, citations_count, license,
                    landing_page, is_original_study, quality_score, valid_doi,
                    experimental_design, study_variables, crop_terms, country,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    doi = excluded.doi,
                    title = excluded.title,
                    abstract = COALESCE(excluded.abstract, papers.abstract),
                    year = excluded.year,
                    journal = excluded.journal,
                    publisher = excluded.publisher,
                    publication_type = excluded.publication_type,
                    language = excluded.language,
                    citations_count = excluded.citations_count,
                    license = excluded.license,
                    landing_page = excluded.landing_page,
                    is_original_study = excluded.is_original_study,
                    quality_score = excluded.quality_score,
                    valid_doi = excluded.valid_doi,
                    experimental_design = excluded.experimental_design,
                    study_variables = excluded.study_variables,
                    crop_terms = excluded.crop_terms,
                    country = excluded.country,
                    updated_at = excluded.updated_at
                """,
                (
                    record.paper_id,
                    record.canonical_doi or record.doi,
                    record.title,
                    record.abstract,
                    record.year,
                    record.journal,
                    record.publisher,
                    record.publication_type,
                    record.language,
                    record.citations_count,
                    record.license,
                    record.landing_page,
                    1 if record.is_original_study else 0,
                    record.quality_score,
                    1 if record.valid_doi else 0,
                    ";".join(record.experimental_design or []),
                    ";".join(record.study_variables or []),
                    ";".join(record.crop_terms or []),
                    record.country,
                    now,
                    now,
                ),
            )
            # authors
            for order, author in enumerate(record.authors):
                author_id = _author_id(
                    author.full_name or f"{author.given_name} {author.family_name}"
                )
                conn.execute(
                    """
                    INSERT INTO authors (author_id, full_name, given_name, family_name, orcid, affiliation)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(author_id) DO UPDATE SET
                        full_name = excluded.full_name,
                        given_name = COALESCE(excluded.given_name, authors.given_name),
                        family_name = COALESCE(excluded.family_name, authors.family_name),
                        orcid = COALESCE(excluded.orcid, authors.orcid),
                        affiliation = COALESCE(excluded.affiliation, authors.affiliation)
                    """,
                    (
                        author_id,
                        author.full_name,
                        author.given_name,
                        author.family_name,
                        author.orcid,
                        author.affiliation,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO paper_authors (paper_id, author_id, author_order)
                    VALUES (?, ?, ?)
                    """,
                    (record.paper_id, author_id, order),
                )
            # source mapping
            conn.execute(
                """
                INSERT INTO paper_sources (source, source_id, paper_id, raw_json, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, source_id) DO UPDATE SET
                    paper_id = excluded.paper_id,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    record.source,
                    record.source_id,
                    record.paper_id,
                    json.dumps(record.raw, default=str),
                    record.fetched_at,
                ),
            )

    def add_references(self, paper_id: str, refs: list[str], source: str) -> None:
        with self._connect() as conn:
            for ref in refs:
                if not ref or not ref.strip():
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO references_ (paper_id, ref_doi, source, relation)
                    VALUES (?, ?, ?, 'cites')
                    """,
                    (paper_id, ref.strip(), source),
                )

    def add_citations(
        self, paper_id: str, citing: list[str], source: str, relation: str = "cited-by"
    ) -> None:
        with self._connect() as conn:
            for item in citing:
                if not item or not item.strip():
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO citations (paper_id, citing_doi, source, relation)
                    VALUES (?, ?, ?, ?)
                    """,
                    (paper_id, item.strip(), source, relation),
                )

    def add_pdf(
        self,
        paper_id: str,
        url: str,
        source_kind: str,
        license: str | None,
        content_type: str | None,
        verified: bool,
        source_repository: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO pdfs (
                    paper_id, url, source_kind, license, content_type, verified,
                    source_repository, added_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    url,
                    source_kind,
                    license,
                    content_type,
                    1 if verified else 0,
                    source_repository,
                    pd.Timestamp.utcnow().isoformat(),
                ),
            )

    def log_crawl(self, result: SyncResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crawl_log (
                    connector, started_at, completed_at, fetched, new_records, skipped, errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.connector,
                    result.started_at,
                    result.completed_at,
                    result.fetched_records,
                    result.new_records,
                    result.skipped_records,
                    "; ".join(result.errors),
                ),
            )

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def all_papers(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM papers ORDER BY year", conn)

    def all_paper_sources(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM paper_sources", conn)

    def all_references(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT paper_id, ref_doi, ref_paper_id, source, relation FROM references_",
                conn,
            )

    def all_citations(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT paper_id, citing_doi, citing_paper_id, source, relation FROM citations",
                conn,
            )

    def all_pdfs(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM pdfs", conn)

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            counts = {
                "papers": conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"],
                "sources": conn.execute("SELECT COUNT(*) AS n FROM paper_sources").fetchone()["n"],
                "references": conn.execute("SELECT COUNT(*) AS n FROM references_").fetchone()["n"],
                "citations": conn.execute("SELECT COUNT(*) AS n FROM citations").fetchone()["n"],
                "pdfs": conn.execute("SELECT COUNT(*) AS n FROM pdfs").fetchone()["n"],
                "verified": conn.execute(
                    "SELECT COUNT(*) AS n FROM papers WHERE is_original_study = 1"
                ).fetchone()["n"],
            }
        return counts


def _author_id(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------- #
# Tabular output writers
# ---------------------------------------------------------------------- #
def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))
    return path


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s (%d rows)", path, len(df))
    return path


def write_excel(sheets: dict[str, pd.DataFrame], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            frame = df if df is not None else pd.DataFrame()
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        _set_xlsx_core_properties(writer.book.properties)
    path.write_bytes(_rewrite_xlsx_deterministic(buf.getvalue()))
    logger.info("Wrote %s (%d sheets)", path, len(sheets))
    return path


def _set_xlsx_core_properties(props: Any) -> None:
    """Pin openpyxl core-properties metadata to a fixed, reproducible value."""
    try:
        props.creator = _XLSX_CREATOR
        props.lastModifiedBy = _XLSX_CREATOR
        props.created = _XLSX_STAMP
        props.modified = _XLSX_STAMP
    except Exception:  # pragma: no cover - property names vary across openpyxl
        logger.debug("openpyxl core-property pinning unavailable", exc_info=True)


def _rewrite_xlsx_deterministic(data: bytes) -> bytes:
    """Re-pack an XLSX (zip) with fixed member timestamps and a pinned core.xml.

    ``_set_xlsx_core_properties`` pins ``created``, but openpyxl overwrites
    ``modified`` with the current time on save, so the timestamp text inside
    ``docProps/core.xml`` is patched too (namespace attributes are preserved).
    Re-writing the archive then gives every zip member a constant date, making
    the output byte-reproducible for identical input frames.
    """
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data), "r") as src,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            raw = src.read(info.filename)
            if info.filename == "docProps/core.xml":
                raw = _CORE_TS_RE.sub(
                    rb"\g<1>" + _XLSX_STAMP_ISO + rb"\g<2>",
                    raw,
                )
            new_info = zipfile.ZipInfo(info.filename, _XLSX_ZIP_DATE)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            dst.writestr(new_info, raw)
    return out.getvalue()


# ---------------------------------------------------------------------- #
# Reports
# ---------------------------------------------------------------------- #
def records_to_dataframe(records: list[LiteratureRecord]) -> pd.DataFrame:
    rows = [r.to_dict() for r in records]
    return pd.DataFrame(rows)


def write_html_report(
    path: str | Path,
    title: str,
    sections: list[tuple[str, Any]],
) -> Path:
    """Render a simple, dependency-free HTML report.

    ``sections`` is a list of (heading, content) where content is a DataFrame,
    a list[dict], or a plain string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body: list[str] = []
    for heading, content in sections:
        body.append(f"<h2>{heading}</h2>")
        if isinstance(content, pd.DataFrame):
            body.append(content.to_html(index=False, escape=False))
        elif isinstance(content, (list, tuple)) and content and isinstance(content[0], dict):
            body.append(pd.DataFrame(content).to_html(index=False, escape=False))
        elif isinstance(content, dict):
            body.append(pd.DataFrame([content]).to_html(index=False, escape=False))
        else:
            body.append(f"<p>{content}</p>")

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1100px;"
        "color:#1a1a1a;background:#fafafa;} "
        "h1{border-bottom:3px solid #2e7d32;padding-bottom:.4rem;} h2{color:#2e7d32;margin-top:2rem;}"
        "table{border-collapse:collapse;width:100%;background:#fff;margin:.6rem 0;} "
        "th,td{border:1px solid #d0d0d0;padding:.35rem .5rem;font-size:.85rem;text-align:left;}"
        "th{background:#e8f5e9;}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        + "".join(body)
        + "<footer><p>Generated by AAIF Literature Intelligence Module.</p></footer>"
        "</body></html>"
    )
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def write_docx_report(
    path: str | Path,
    title: str,
    sections: list[tuple[str, Any]],
) -> Path:
    """Render a .docx report via python-docx (requires the ``ml-full`` extra)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        logger.warning("python-docx not installed; skipping .docx report (%s)", exc)
        # fall back to a plain-text .docx-compatible file
        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n")
            for heading, content in sections:
                f.write("\n" + heading + "\n" + ("=" * len(heading)) + "\n")
                if isinstance(content, pd.DataFrame):
                    f.write(content.to_csv(index=False))
                elif isinstance(content, str):
                    f.write(content)
                else:
                    f.write(str(content))
        return path

    doc = Document()
    doc.add_heading(title, level=0)
    for heading, content in sections:
        doc.add_heading(heading, level=1)
        if isinstance(content, pd.DataFrame):
            if content.empty:
                doc.add_paragraph("(no rows)")
                continue
            table = doc.add_table(rows=1, cols=len(content.columns))
            table.style = "Light Grid Accent 1"
            hdr = table.rows[0].cells
            for i, col in enumerate(content.columns):
                hdr[i].text = str(col)
            for _, row in content.head(200).iterrows():
                cells = table.add_row().cells
                for i, value in enumerate(row):
                    cells[i].text = "" if pd.isna(value) else str(value)
        elif isinstance(content, str):
            doc.add_paragraph(content)
        else:
            doc.add_paragraph(str(content))
    doc.save(str(path))
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------- #
# Dependency-free PDF report writer
# ---------------------------------------------------------------------- #
def write_pdf_report(
    path: str | Path,
    title: str,
    sections: list[tuple[str, Any]],
) -> Path:
    """Render a minimal, dependency-free PDF report.

    The report mirrors ``write_html_report``'s structure: a title, a section
    heading, then tables / key-value lists / paragraphs.  PDF-unsafe characters
    are escaped; tables are truncated to a sane row budget.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _PdfDocument()
    doc.add_line(title, size=20, gap=16)
    doc.add_line(f"Generated: {datetime.now(timezone.utc).isoformat()}", size=8)
    doc.add_line("")
    for heading, content in sections:
        doc.add_line(heading, size=14, gap=12)
        if isinstance(content, pd.DataFrame):
            doc.add_dataframe(content)
        elif isinstance(content, (list, tuple)):
            if content and isinstance(content[0], dict):
                doc.add_dataframe(pd.DataFrame(content))
            else:
                doc.add_text("\n".join(str(x) for x in content))
        elif isinstance(content, dict):
            doc.add_text("\n".join(f"{k}: {v}" for k, v in content.items()))
        else:
            doc.add_text(str(content))
    path.write_bytes(doc.render())
    logger.info("Wrote %s", path)
    return path


class _PdfDocument:
    """Tiny PDF (1.4) builder: one page per ``A4`` sheet, Helvetica text only."""

    PAGE_WIDTH = 595.27
    PAGE_HEIGHT = 841.89
    MARGIN = 46.0
    MAX_LINES_PER_PAGE = 58
    MAX_TABLE_ROWS = 60

    def __init__(self) -> None:
        self._pages: list[list[tuple[str, int, int]]] = [[]]
        self._line_no = 0

    def _ensure_line(self) -> None:
        if self._line_no >= self.MAX_LINES_PER_PAGE:
            self._pages.append([])
            self._line_no = 0

    def _push(self, text: str, size: int, gap: int) -> None:
        self._ensure_line()
        self._pages[-1].append((text, size, gap))
        self._line_no += 1

    def add_line(self, text: str, size: int = 9, gap: int = 10) -> None:
        for wrapped in self._wrap(text, size):
            self._push(wrapped, size, gap)

    def add_text(self, text: str) -> None:
        for chunk in str(text).splitlines() or [""]:
            self.add_line(chunk, size=9, gap=9)

    def add_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.add_line("(no rows)", size=9)
            return
        head = df.head(self.MAX_TABLE_ROWS)
        widths = self._column_widths(head)
        header = "  |  ".join(f"{_truncate(c, w)}" for c, w in zip(df.columns, widths, strict=True))
        self.add_line(header, size=8, gap=10)
        for _, row in head.iterrows():
            cells = ["" if pd.isna(v) else str(v) for v in row.values]
            line = "  |  ".join(f"{_truncate(c, w)}" for c, w in zip(cells, widths, strict=True))
            self.add_line(line, size=8, gap=7)
        if len(df) > self.MAX_TABLE_ROWS:
            self.add_line(f"... {len(df) - self.MAX_TABLE_ROWS} more rows omitted", size=8)

    @staticmethod
    def _column_widths(df: pd.DataFrame) -> list[int]:
        available = (  # usable width in characters at 8pt Helvetica (~4.5pt/char)
            _PdfDocument.PAGE_WIDTH - 2 * _PdfDocument.MARGIN
        ) / 4.5
        ncols = max(len(df.columns), 1)
        base = int(available / ncols)
        widths: list[int] = []
        for col in df.columns:
            maxlen = max(
                [len(str(col))] + [len(str(v)) for v in df[col].head(40) if not pd.isna(v)],
                default=base,
            )
            widths.append(max(min(maxlen + 1, int(base * 2.2)), 6))
        # normalize to available budget
        total = sum(widths)
        if total > available:
            scale = available / total
            widths = [max(int(w * scale), 4) for w in widths]
        return widths

    def _wrap(self, text: str, size: int) -> list[str]:
        text = str(text)
        max_chars = int((self.PAGE_WIDTH - 2 * self.MARGIN) / (0.5 * size))
        if len(text) <= max_chars or max_chars <= 1:
            return [text]
        out: list[str] = []
        for raw_line in text.split("\n") or [""]:
            line = raw_line
            while len(line) > max_chars:
                out.append(line[:max_chars])
                line = line[max_chars:]
            out.append(line)
        return out or [""]

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render(self) -> bytes:
        # Fixed object layout:
        #   1 catalog, 2 pages, 3 Helvetica, 4 Helvetica-Bold,
        #   5..4+N content streams, 5+N..4+2N page objects.
        n_pages = len(self._pages)
        content_objs: list[bytes] = [self._content_stream(items) for items in self._pages]
        page_objs: list[bytes] = []
        for i in range(n_pages):
            content_obj = 5 + i
            page_objs.append(self._page_object(content_obj))
        pages_obj = 2
        catalog_obj = 1

        objects = [
            self._catalog_object(pages_obj),  # 1
            self._pages_object(page_objs),  # 2
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",  # 3
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",  # 4
            *content_objs,  # 5..4+N
            *page_objs,  # 5+N..4+2N
        ]

        out = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode()
            out += obj + b"\nendobj\n"
        xref_pos = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n"
            "startxref\n"
            f"{xref_pos}\n"
            "%%EOF\n"
        ).encode()
        return bytes(out)

    def _content_stream(self, items: list[tuple[str, int, int]]) -> bytes:
        lines: list[str] = []
        y = self.PAGE_HEIGHT - self.MARGIN - 14
        for text, size, gap in items:
            font = "/F2" if size >= 14 else "/F1"
            lines.append(
                f"BT {font} {size} Tf 1 0 0 1 {self.MARGIN} {y:.2f} Tm ({_pdf_escape(text)}) Tj ET"
            )
            y -= size + gap
            if y < self.MARGIN:
                y = self.PAGE_HEIGHT - self.MARGIN - 14
        stream = "\n".join(lines)
        return (
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream".encode()
        )

    def _page_object(self, content_obj: int) -> bytes:
        return (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.PAGE_WIDTH:.1f} {self.PAGE_HEIGHT:.1f}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_obj} 0 R >>"
        ).encode()

    def _pages_object(self, pages: list[bytes]) -> bytes:
        # page objects occupy 5+N .. 4+2N
        first = 5 + len(pages)
        kids = " ".join(f"{first + i} 0 R" for i in range(len(pages)))
        return f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()

    @staticmethod
    def _catalog_object(pages_obj: int) -> bytes:
        return f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()


def _pdf_escape(text: str) -> str:
    escaped = (
        str(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    # Helvetica WinAnsi (latin-1) only: drop any character it cannot encode.
    return escaped.encode("latin-1", "replace").decode("latin-1")


def _truncate(text: str, width: int) -> str:
    text = str(text)
    if len(text) <= width:
        return text
    return text[: max(width - 1, 1)] + "\u2026"
