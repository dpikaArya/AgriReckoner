"""DOAJ connector (https://doaj.org/api, official public API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class DoajConnector(LiteratureConnector):
    source_name = "DOAJ"
    display_name = "DOAJ"
    base_url = "https://doaj.org/api"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=("DOAJ_API_KEY", "AGRI_DOAJ_API_KEY"), required=False)

    def _to_record(self, article: dict[str, Any]) -> LiteratureRecord:
        bib = article.get("bibjson", {}) or {}
        doi = None
        for ident in bib.get("identifier", []) or []:
            if ident.get("type") == "doi":
                doi = normalize_doi(ident.get("id"))
                break
        authors = [
            Author(
                full_name=a.get("name", ""),
                orcid=((a.get("orcid_id") or "") or None),
                affiliation=a.get("affiliation"),
            )
            for a in bib.get("author", []) or []
        ]
        pdf_locations: list[PdfLocation] = []
        for link in bib.get("link", []) or []:
            content_type = (link.get("content_type") or "").lower()
            if "pdf" in content_type and link.get("url"):
                pdf_locations.append(
                    PdfLocation(
                        url=link["url"],
                        source_kind="open_access",
                        content_type="application/pdf",
                        source_repository="DOAJ",
                    )
                )
        return LiteratureRecord(
            source=self.source_name,
            source_id=str(article.get("id", "")),
            doi=doi,
            title=bib.get("title"),
            abstract=bib.get("abstract"),
            year=_year(bib.get("year")),
            journal=bib.get("journal", {}).get("title"),
            publisher=bib.get("journal", {}).get("publisher"),
            authors=authors,
            publication_type=_pub_type(bib.get("doaj_type", []) or []),
            language=_first_lang(bib.get("keywords", []) or []),
            keywords=list(dict.fromkeys(kw for kw in (bib.get("keywords") or []) if kw))[:20],
            pdf_locations=pdf_locations,
            license=bib.get("license"),
            landing_page=bib.get("link") and bib["link"][0].get("url"),
            raw=article,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/search/articles/",
            params={"query": query, "pageSize": max_results, "page": page + 1, "source": "docs"},
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(a) for a in payload.get("results", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "/search/articles/",
            params={
                "query": f"doi:{normalize_doi(doi)}",
                "pageSize": 1,
                "page": 1,
                "source": "docs",
            },
        )
        if not isinstance(payload, dict):
            return None
        results = payload.get("results", []) or []
        return self._to_record(results[0]) if results else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/articles/{source_id}")
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []  # DOAJ does not index references

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []


def _year(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _pub_type(types: list[str]) -> str | None:
    return types[0].lower() if types else None


def _first_lang(keywords: list[str]) -> str | None:
    return None
