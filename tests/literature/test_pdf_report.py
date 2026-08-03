"""Tests for the dependency-free PDF report writer."""

from __future__ import annotations

import pandas as pd

from agri_ai_agent.literature.storage import _pdf_escape, _truncate, write_pdf_report


def test_pdf_report_is_valid_pdf_with_content(tmp_path):
    sections = [
        ("Executive Summary", {"run_id": "RUN-1", "papers": 24}),
        ("Verified Papers", pd.DataFrame({"doi": ["10.1/x"], "title": ["A paper"]})),
        ("Errors", ["(none)"]),
    ]
    path = write_pdf_report(
        tmp_path / "report.pdf",
        "AAIF Literature Intelligence Framework Report",
        sections,
    )
    data = path.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert b"/Type /Catalog" in data
    assert b"/Type /Pages" in data
    assert b"/Type /Page" in data
    assert b"AAIF Literature Intelligence Framework Report" in data
    assert b"RUN-1" in data
    assert b"(no errors)" in data or b"none" in data


def test_pdf_report_handles_multi_page_and_special_chars(tmp_path):
    long_text = "sample field experiment row " * 200
    sections = [
        ("Long Text", long_text),
        ("Special", "parens (a) and \\ backslash"),
    ]
    path = write_pdf_report(tmp_path / "multi.pdf", "Multi Page Report", sections)
    data = path.read_bytes()
    assert data.count(b"/Type /Page") >= 1
    assert b"parens \\(a\\) and \\\\ backslash" in data


def test_pdf_escape_and_truncate():
    assert _pdf_escape("a(b)") == "a\\(b\\)"
    assert _pdf_escape("back\\slash") == "back\\\\slash"
    assert _pdf_escape("line\nbreak") == "line break"
    assert _truncate("abcdefgh", 4) == "abc\u2026"
    assert _truncate("abc", 5) == "abc"
