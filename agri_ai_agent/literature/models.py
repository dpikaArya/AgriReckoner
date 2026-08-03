"""Normalized metadata schema for the AAIF Literature Intelligence Module.

Every connector maps its native API response onto :class:`LiteratureRecord`
and :class:`AgriculturalRecord` so downstream consumers (bibliography DB,
parquet outputs, citation/author/experiment graphs, UAMS updates) work on a
single, source-independent representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Author:
    """A single author of a scholarly work."""

    full_name: str = ""
    given_name: str | None = None
    family_name: str | None = None
    orcid: str | None = None
    affiliation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "orcid": self.orcid,
            "affiliation": self.affiliation,
        }


@dataclass
class PdfLocation:
    """A legally-accessible full-text location recorded from official metadata."""

    url: str
    source_kind: str  # "publisher" | "repository" | "open_access"
    license: str | None = None
    content_type: str | None = None
    verified: bool = False
    doi: str | None = None
    source_repository: str | None = None


@dataclass
class LiteratureRecord:
    """Canonical, source-normalized record of a scholarly work.

    ``source`` / ``source_id`` identify the originating connector and its
    native identifier. ``doi`` is the cross-source deduplication key when
    present; otherwise a normalized title is used.
    """

    source: str
    source_id: str
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    journal: str | None = None
    publisher: str | None = None
    authors: list[Author] = field(default_factory=list)
    publication_type: str | None = None  # journal-article | review | ... (source taxonomy)
    language: str | None = None
    subjects: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # DOIs / native ids of cited works
    citations_count: int | None = None
    related_ids: list[str] = field(default_factory=list)
    pdf_locations: list[PdfLocation] = field(default_factory=list)
    license: str | None = None
    landing_page: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = field(default_factory=utcnow_iso)

    # ---- pipeline-derived fields (filled by dedupe/classify/quality) ----
    paper_id: str | None = None
    canonical_title: str | None = None
    canonical_doi: str | None = None
    quality_score: float | None = None
    is_original_study: bool | None = None
    valid_doi: bool | None = None
    duplicate_of: str | None = None
    experimental_design: list[str] = field(default_factory=list)
    study_variables: list[str] = field(default_factory=list)
    crop_terms: list[str] = field(default_factory=list)
    experimental_keywords: list[str] = field(default_factory=list)
    country: str | None = None

    @property
    def source_title(self) -> str:
        return self.title or self.source_id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiteratureRecord:
        """Rehydrate a record from ``to_dict()`` (persisted in the cache)."""
        authors = [
            Author(
                full_name=a.get("full_name", "") or "",
                given_name=a.get("given_name"),
                family_name=a.get("family_name"),
                orcid=a.get("orcid"),
                affiliation=a.get("affiliation"),
            )
            for a in data.get("authors", []) or []
        ]
        pdfs = [
            PdfLocation(
                url=p.get("url", ""),
                source_kind=p.get("source_kind", "publisher"),
                license=p.get("license"),
                content_type=p.get("content_type"),
                verified=bool(p.get("verified", False)),
                doi=p.get("doi"),
                source_repository=p.get("source_repository"),
            )
            for p in data.get("pdf_locations", []) or []
        ]
        return cls(
            source=data.get("source", ""),
            source_id=data.get("source_id", ""),
            doi=data.get("doi"),
            title=data.get("title"),
            abstract=data.get("abstract"),
            year=data.get("year"),
            journal=data.get("journal"),
            publisher=data.get("publisher"),
            authors=authors,
            publication_type=data.get("publication_type"),
            language=data.get("language"),
            subjects=list(data.get("subjects", []) or []),
            keywords=list(data.get("keywords", []) or []),
            references=list(data.get("references", []) or []),
            citations_count=data.get("citations_count"),
            related_ids=list(data.get("related_ids", []) or []),
            pdf_locations=pdfs,
            license=data.get("license"),
            landing_page=data.get("landing_page"),
            raw=data.get("raw", {}),
            fetched_at=data.get("fetched_at", ""),
            paper_id=data.get("paper_id"),
            canonical_title=data.get("canonical_title"),
            canonical_doi=data.get("canonical_doi"),
            quality_score=data.get("quality_score"),
            is_original_study=data.get("is_original_study"),
            valid_doi=data.get("valid_doi"),
            duplicate_of=data.get("duplicate_of"),
            experimental_design=list(data.get("experimental_design", []) or []),
            study_variables=list(data.get("study_variables", []) or []),
            crop_terms=list(data.get("crop_terms", []) or []),
            experimental_keywords=list(data.get("experimental_keywords", []) or []),
            country=data.get("country"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "journal": self.journal,
            "publisher": self.publisher,
            "authors": [a.to_dict() for a in self.authors],
            "publication_type": self.publication_type,
            "language": self.language,
            "subjects": self.subjects,
            "keywords": self.keywords,
            "references": self.references,
            "citations_count": self.citations_count,
            "related_ids": self.related_ids,
            "pdf_locations": [
                {
                    "url": p.url,
                    "source_kind": p.source_kind,
                    "license": p.license,
                    "content_type": p.content_type,
                    "verified": p.verified,
                    "doi": p.doi,
                    "source_repository": p.source_repository,
                }
                for p in self.pdf_locations
            ],
            "license": self.license,
            "landing_page": self.landing_page,
            "fetched_at": self.fetched_at,
            "paper_id": self.paper_id,
            "canonical_title": self.canonical_title,
            "canonical_doi": self.canonical_doi,
            "quality_score": self.quality_score,
            "is_original_study": self.is_original_study,
            "valid_doi": self.valid_doi,
            "duplicate_of": self.duplicate_of,
            "experimental_design": self.experimental_design,
            "study_variables": self.study_variables,
            "crop_terms": self.crop_terms,
            "experimental_keywords": self.experimental_keywords,
            "country": self.country,
        }


@dataclass
class AgriculturalRecord:
    """Normalized record produced by agricultural data connectors.

    ``variable`` is the Universal-Agricultural-Schema canonical variable name,
    ``value`` the numeric observation, ``unit`` a normalized unit string, and
    ``provenance`` the exact official URL / dataset identifier the value was
    read from.  ``extra`` carries connector-specific metadata.
    """

    source: str
    dataset_id: str
    variable: str
    value: float | None
    unit: str | None = None
    location_lat: float | None = None
    location_lon: float | None = None
    year: int | None = None
    crop: str | None = None
    country: str | None = None
    provenance: str | None = None
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset_id": self.dataset_id,
            "variable": self.variable,
            "value": self.value,
            "unit": self.unit,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "year": self.year,
            "crop": self.crop,
            "country": self.country,
            "provenance": self.provenance,
            "license": self.license,
            "extra": self.extra,
            "fetched_at": self.fetched_at,
        }


@dataclass
class SyncResult:
    """Outcome of an incremental synchronization pass."""

    connector: str
    started_at: str = field(default_factory=utcnow_iso)
    completed_at: str | None = None
    fetched_records: int = 0
    new_records: int = 0
    updated_records: int = 0
    skipped_records: int = 0
    errors: list[str] = field(default_factory=list)
    cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "fetched_records": self.fetched_records,
            "new_records": self.new_records,
            "updated_records": self.updated_records,
            "skipped_records": self.skipped_records,
            "errors": self.errors,
            "cursor": self.cursor,
        }
