"""Offline tests for the PDF reader and the `agriai extract` CLI path."""

import sys

from agri_ai_agent.extractors.pdf_reader import read_papers_dir


def test_read_papers_dir_empty(tmp_path):
    assert read_papers_dir(tmp_path) == {}


def test_read_papers_dir_ignores_non_pdf(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")
    assert read_papers_dir(tmp_path) == {}


def test_cli_extract_empty_dir_does_not_crash(tmp_path, monkeypatch):
    from agri_ai_agent import cli

    monkeypatch.setattr(sys, "argv", ["agriai", "extract", "--papers", str(tmp_path)])
    cli.main()  # warns "no readable PDFs" and returns; no exception


def test_cli_extract_offline_no_key(tmp_path, monkeypatch):
    """With papers but no OpenAI key, extract degrades to a no-op without crashing."""
    from agri_ai_agent import cli
    from agri_ai_agent.config.settings import AgriAISettings

    monkeypatch.setattr(
        "agri_ai_agent.extractors.pdf_reader.read_papers_dir",
        lambda _d: {"p1": "Soil pH was 6.8."},
    )
    monkeypatch.setattr(AgriAISettings, "OPENAI_API_KEY", "", raising=False)
    monkeypatch.setattr(sys, "argv", ["agriai", "extract", "--papers", str(tmp_path)])
    cli.main()  # LLMExtractionAgent has no key/extractor -> no-op, no exception


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
