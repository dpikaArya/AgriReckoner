"""Semantic Scholar connector (Graph API v1, official public API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation

_FIELDS = (
    "title,abstract,externalIds,year,venue,publicationTypes,authors,"
    "citationCount,openAccessPdf,url,references,citationStyles"
)


class SemanticScholarConnector(LiteratureConnector):
    source_name = "Semantic_Scholar"
    display_name = "Semantic Scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"
    default_rate_per_minute = 60

    auth = ConnectorAuth(
        env_vars=("S2_API_KEY", "AGRI_SEMANTIC_SCHOLAR_API_KEY"),
        required=False,
        header="x-api-key",
    )

    def _to_record(self, paper: dict[str, Any]) -> LiteratureRecord:
        ids = paper.get("externalIds", {}) or {}
        doi = normalize_doi(ids.get("DOI"))
        authors = [
            Author(
                full_name=a.get("name", ""),
                orcid=((a.get("authorId", "") or "") if a.get("authorId", "").startswith(("0000-", "http")) else None),
                affiliation=a.get("affiliation"),
            )
            for a in paper.get("authors", []) or []
        ]
        pdf_locations: list[PdfLocation] = []
        oa_pdf = paper.get("openAccessPdf")
        if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
            pdf_locations.append(
                PdfLocation(
                    url=oa_pdf["url"],
                    source_kind="open_access",
                    source_repository="Semantic Scholar",
                )
            )
        pub_types = paper.get("publicationTypes", []) or []
        pub_type = pub_types[0] if pub_types else None
        return LiteratureRecord(
            source=self.source_name,
            source_id=str(paper.get("paperId", "")),
            doi=doi,
            title=paper.get("title"),
            abstract=paper.get("abstract"),
            year=paper.get("year"),
            journal=paper.get("venue"),
            authors=authors,
            publication_type=pub_type,
            references=[normalize_doi(r) or r for r in (paper.get("references") or [])][:50],
            citations_count=paper.get("citationCount"),
            pdf_locations=pdf_locations,
            landing_page=paper.get("url"),
            raw=paper,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/paper/search",
            params={
                "query": query,
                "limit": max_results,
                "offset": page * max_results,
                "fields": _FIELDS,
            },
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(p) for p in payload.get("data", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            f"/paper/DOI:{normalize_doi(doi)}",
            params={"fields": _FIELDS},
        )
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            f"/paper/{source_id}",
            params={"fields": _FIELDS},
        )
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        payload = self.http.get_json(
            f"/paper/{record.source_id}/references",
            params={"fields": "externalIds,paperId", "limit": 100},
        )
        if not isinstance(payload, dict):
            return []
        return self._ids_from_page(payload)

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        payload = self.http.get_json(
            f"/paper/{record.source_id}/citations",
            params={"fields": "externalIds,paperId", "limit": 100},
        )
        if not isinstance(payload, dict):
            return []
        return self._ids_from_page(payload)

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        payload = self.http.get_json(
            f"/paper/{record.source_id}/related",
            params={"fields": "externalIds,paperId", "limit": 25},
        )
        if not isinstance(payload, dict):
            return []
        return self._ids_from_page(payload)

    def _ids_from_page(self, payload: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for item in payload.get("data", []) or []:
            node = item.get("citingPaper") or item.get("citedPaper") or item.get("paper") or item
            ids = node.get("externalIds", {}) or {}
            doi = normalize_doi(ids.get("DOI"))
            out.append(doi or node.get("paperId", ""))
        return out
