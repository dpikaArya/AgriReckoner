"""Unit tests for deduplication, DOI validation and quality scoring."""

from __future__ import annotations

from agri_ai_agent.literature.dedupe import (
    compute_quality_score,
    deduplicate_records,
    title_similarity,
    validate_doi_format,
)
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


def _rec(source: str, source_id: str, doi: str | None = None, title: str | None = None, **kw):
    return LiteratureRecord(
        source=source, source_id=source_id, doi=doi, title=title, **kw
    )


# ---------------------------------------------------------------------- #
# similarity
# ---------------------------------------------------------------------- #
def test_title_similarity_identical():
    assert title_similarity("Wheat yield in India", "wheat yield in india") == 1.0


def test_title_similarity_punctuation():
    a = title_similarity("Field experiments, 2020", "Field experiments 2020")
    assert a >= 0.95


# ---------------------------------------------------------------------- #
# dedupe by DOI
# ---------------------------------------------------------------------- #
def test_dedupe_by_doi_merges():
    primary = _rec(
        "OpenAlex", "W1", doi="10.1000/wheat",
        title="Wheat yield response to nitrogen",
        authors=[Author(full_name="A. Kumar")],
        abstract="A long abstract " + "x" * 200,
        pdf_locations=[PdfLocation(url="https://example.com/paper.pdf", source_kind="open_access")],
        references=["10.1000/ref"],
    )
    duplicate = _rec(
        "Crossref", "cr-1", doi="HTTP://DOI.ORG/10.1000/wheat",
        title="Wheat yield response to nitrogen",
        citations_count=42,
    )
    canonical, mapping = deduplicate_records([primary, duplicate])

    assert len(canonical) == 1
    paper = canonical[0]
    assert paper.canonical_doi == "10.1000/wheat"
    assert len(paper.pdf_locations) == 1
    assert "10.1000/ref" in paper.references
    assert paper.citations_count == 42
    assert paper.duplicate_of == "cr-1"
    assert mapping[("OpenAlex", "W1")] == paper.paper_id or paper.source_id
    assert mapping[("Crossref", "cr-1")] == paper.paper_id or paper.source_id


def test_dedupe_by_title_when_no_doi():
    a = _rec("A", "a1", title="Rice transplanting density effects on yield")
    b = _rec("B", "b1", title="Rice transplanting density effects on yield")
    canonical, _ = deduplicate_records([a, b])
    assert len(canonical) == 1


def test_dedupe_distinct_papers_kept():
    a = _rec("A", "a1", doi="10.1000/a", title="Wheat nitrogen study")
    b = _rec("B", "b1", doi="10.1000/b", title="Rice water study")
    canonical, _ = deduplicate_records([a, b])
    assert len(canonical) == 2


# ---------------------------------------------------------------------- #
# quality scoring
# ---------------------------------------------------------------------- #
def test_quality_score_bounds():
    rich = _rec(
        "A", "a1", doi="10.1000/x",
        title="A sufficiently long title about field experiments",
        year=2020, journal="Field Crops Research",
        authors=[Author(full_name="A")],
        abstract="An abstract that is definitely longer than one hundred characters in total length here.",
        pdf_locations=[PdfLocation(url="https://x/y.pdf", source_kind="publisher")],
        references=["10.1000/z"],
    )
    score = compute_quality_score(rich)
    assert 0.0 <= score <= 1.0
    assert score > 0.7


def test_quality_score_floor():
    sparse = _rec("A", source_id="a1", title=None)
    assert compute_quality_score(sparse) == 0.0


# ---------------------------------------------------------------------- #
# DOI format validation
# ---------------------------------------------------------------------- #
def test_validate_doi_format():
    assert validate_doi_format("10.1234/abc") is True
    assert validate_doi_format("https://doi.org/10.1234/abc") is True
    assert validate_doi_format("nope") is False
    assert validate_doi_format(None) is False
