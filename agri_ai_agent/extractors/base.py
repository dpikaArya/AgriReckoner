"""Shared types for extraction strategies."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExtractedField:
    """One value pulled from a paper, with its provenance and trust status.

    ``status`` is one of: ``reported`` (verbatim from the source and grounded),
    ``unverified`` (LLM-proposed but not yet grounded), ``rejected`` (failed grounding).
    """

    column: str
    value: float | None = None
    unit_as_reported: str | None = None
    source_quote: str | None = None
    page: int | None = None
    model_confidence: float | None = None
    status: str = "unverified"
    reject_reason: str | None = None
    value_canonical: float | None = None
    canonical_unit: str | None = None


@dataclass
class ExtractionResult:
    """All fields extracted from a single paper, plus paper-level metadata."""

    paper_id: str
    crop: str | None = None
    doi: str | None = None
    year: int | None = None
    fields: list[ExtractedField] = field(default_factory=list)
    method: str = "unknown"


@runtime_checkable
class Extractor(Protocol):
    """A strategy that extracts UAMS fields from a paper's text."""

    def extract(self, text: str, paper_id: str) -> ExtractionResult:
        ...


if __name__ == "__main__":
    f = ExtractedField(column="Soil_pH", value=6.8, unit_as_reported=None, source_quote="Soil pH was 6.8")
    r = ExtractionResult(paper_id="p1", crop="Wheat", fields=[f], method="test")
    assert r.fields[0].column == "Soil_pH" and r.fields[0].status == "unverified"
    print("base smoke OK ->", r.paper_id, r.fields[0].column)
