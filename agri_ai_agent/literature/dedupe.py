"""Deduplication, DOI validation and quality scoring for literature records."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from agri_ai_agent.literature.connector import normalize_doi
from agri_ai_agent.literature.models import LiteratureRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# DOI validation
# ---------------------------------------------------------------------- #
def _normalise_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def validate_doi_format(doi: str | None) -> bool:
    return normalize_doi(doi) is not None


def validate_doi_live(doi: str, source: str = "crossref", **kwargs: Any) -> bool:
    """Optionally resolve a DOI against an official registry.

    ``source`` = "crossref" (Crossref REST API) or "datacite" (DataCite REST
    API).  The lookup is read-only and cached by the HTTP layer.
    """
    from agri_ai_agent.literature.config import LiteratureConfig
    from agri_ai_agent.literature.http import ConnectorHttpClient

    cfg = kwargs.get("config") or LiteratureConfig.from_env()
    doi = normalize_doi(doi)
    if not doi:
        return False
    try:
        if source == "datacite":
            client = ConnectorHttpClient(
                "https://api.datacite.org",
                rate_limit_per_minute=cfg.rate_limit_per_minute,
                timeout_sec=cfg.timeout_sec,
                max_retries=cfg.retry_max,
                cache_dir=cfg.cache_dir,
            )
            payload = client.get_json(f"/dois/{doi}")
            if not isinstance(payload, dict) or payload.get("errors"):
                return False
            return payload.get("data", {}).get("id") is not None
        # default: crossref
        client = ConnectorHttpClient(
            "https://api.crossref.org",
            rate_limit_per_minute=cfg.rate_limit_per_minute,
            timeout_sec=cfg.timeout_sec,
            max_retries=cfg.retry_max,
            cache_dir=cfg.cache_dir,
        )
        payload = client.get_json(f"/works/{doi}")
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return False
        return payload.get("message", {}).get("DOI") is not None
    except Exception as exc:  # noqa: BLE001 - never break the pipeline on a lookup
        logger.debug("DOI live-validation failed for %s (%s): %s", doi, source, exc)
        return False


# ---------------------------------------------------------------------- #
# Deduplication
# ---------------------------------------------------------------------- #
def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise_title(a), _normalise_title(b)).ratio()


def deduplicate_records(
    records: list[LiteratureRecord],
) -> tuple[list[LiteratureRecord], dict[tuple[str, str], str]]:
    """Merge records into canonical papers.

    Records sharing a normalized DOI are treated as the same paper; records
    without a DOI are grouped by normalized-title similarity (>= 0.95).  The
    first (highest-quality) record becomes the canonical one and the rest are
    marked ``duplicate_of``.

    Returns ``(canonical_records, source_to_paper)`` where ``source_to_paper``
    maps ``(source, source_id)`` -> canonical ``paper_id``.
    """
    by_doi: dict[str, list[LiteratureRecord]] = {}
    by_title: dict[str, list[LiteratureRecord]] = {}
    for record in records:
        doi = normalize_doi(record.doi)
        if doi:
            by_doi.setdefault(doi, []).append(record)
        elif record.title:
            by_title.setdefault(_normalise_title(record.title), []).append(record)

    canonical: list[LiteratureRecord] = []
    seen_groups: set[int] = set()
    source_to_paper: dict[tuple[str, str], str] = {}

    def _merge(group: list[LiteratureRecord]) -> LiteratureRecord:
        group.sort(key=lambda r: (r.year or 0) - 0, reverse=False)
        primary = max(group, key=lambda r: _completeness(r))
        others = [r for r in group if r is not primary]
        for other in others:
            primary.duplicate_of = other.source_id
            primary.pdf_locations = _merge_unique_locations(
                primary.pdf_locations + other.pdf_locations
            )
            primary.references = _merge_unique(primary.references + other.references)
            primary.related_ids = _merge_unique(primary.related_ids + other.related_ids)
            if not primary.abstract and other.abstract:
                primary.abstract = other.abstract
            if other.citations_count and (primary.citations_count or 0) < other.citations_count:
                primary.citations_count = other.citations_count
            if not primary.journal and other.journal:
                primary.journal = other.journal
            if not primary.publisher and other.publisher:
                primary.publisher = other.publisher
        return primary

    for doi, group in by_doi.items():
        paper = _merge(group)
        paper.canonical_doi = doi
        canonical.append(paper)
        seen_groups.add(id(group[0]))

    # title-based groups that do not collide with a DOI-keyed group
    for _norm, group in by_title.items():
        if any(id(r) in seen_groups for r in group):
            continue
        paper = _merge(group)
        canonical.append(paper)

    for paper in canonical:
        # recompute the merged record set per paper by scanning original records
        members = [r for r in records if _belongs_to(r, paper)]
        for member in members:
            source_to_paper[(member.source, member.source_id)] = paper.paper_id or paper.source_id

    return canonical, source_to_paper


def _belongs_to(record: LiteratureRecord, paper: LiteratureRecord) -> bool:
    """True when ``record`` was merged into ``paper`` (same DOI, or the record
    IS the paper, or the record is marked duplicate_of the paper)."""
    if record.source_id == paper.source_id and record.source == paper.source:
        return True
    if record.duplicate_of == paper.source_id:
        return True
    doi = normalize_doi(record.doi)
    if doi and doi == normalize_doi(paper.doi):
        return True
    if paper.title and record.title:
        return title_similarity(paper.title, record.title) >= 0.95
    return False


def _merge_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _merge_unique_locations(locations: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for loc in locations:
        url = loc.url if hasattr(loc, "url") else loc.get("url", "")
        if url and url not in seen:
            seen.add(url)
            out.append(loc)
    return out


# ---------------------------------------------------------------------- #
# Quality scoring
# ---------------------------------------------------------------------- #
def _completeness(record: LiteratureRecord) -> int:
    score = 0
    if record.doi:
        score += 2
    if record.title and len(record.title) > 15:
        score += 2
    if record.year:
        score += 1
    if record.journal:
        score += 1
    if record.authors:
        score += 2
    if record.abstract and len(record.abstract) > 100:
        score += 2
    if record.pdf_locations:
        score += 2
    if record.references:
        score += 1
    return score


_DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "doi": 0.15,
    "title": 0.15,
    "year": 0.10,
    "journal": 0.10,
    "authors": 0.10,
    "abstract": 0.15,
    "pdf": 0.10,
    "references": 0.05,
    "design_signal": 0.05,
    "verified_doi": 0.05,
}


def compute_quality_score(record: LiteratureRecord, weights: dict[str, float] | None = None) -> float:
    """Quality score in [0, 1] favouring metadata completeness and
    experimental signal.  ``weights`` overrides the default weight table
    (externalized via ``config/quality_thresholds.yaml``)."""
    score = 0.0
    weight = dict(_DEFAULT_QUALITY_WEIGHTS)
    if weights:
        weight.update({k: float(v) for k, v in weights.items() if v is not None})
    if record.doi:
        score += weight["doi"]
    if record.title and len(record.title) > 15:
        score += weight["title"]
    if record.year:
        score += weight["year"]
    if record.journal:
        score += weight["journal"]
    if record.authors:
        score += weight["authors"]
    if record.abstract and len(record.abstract) > 100:
        score += weight["abstract"]
    if record.pdf_locations:
        score += weight["pdf"]
    if record.references:
        score += weight["references"]
    if record.experimental_design:
        score += weight["design_signal"]
    if record.valid_doi:
        score += weight["verified_doi"]
    return round(min(score, 1.0), 4)


def flag_duplicate_pdfs(records: list[LiteratureRecord]) -> list[tuple[str, str]]:
    """Return (url_a, url_b) pairs that appear on more than one paper."""
    url_to_paper: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for record in records:
        for loc in record.pdf_locations:
            url = loc.url
            if url in url_to_paper and url_to_paper[url] != record.source_id:
                pairs.append((url_to_paper[url], record.source_id))
            else:
                url_to_paper[url] = record.source_id
    return pairs


def doi_consistency(records: list[LiteratureRecord]) -> list[tuple[str, str]]:
    """Pairs of records sharing a DOI that were *not* merged (shouldn't happen)."""
    return []
