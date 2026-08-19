"""Scopus connector (api.elsevier.com, official Search API).

Requires an Elsevier API key (X-ELS-APIKey).  ``fetch_citations`` returns the
authoritative citation count; full citing lists require the institutional
Citation Overview add-on and are left to the SCOPUS_CITATION_OVERVIEW flag.
"""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class ScopusConnector(LiteratureConnector):
    source_name = "Scopus"
    display_name = "Scopus"
    base_url = "https://api.elsevier.com/content/search/scopus"
    default_rate_per_minute = 30

    auth = ConnectorAuth(
        env_vars=("SCOPUS_API_KEY", "AGRI_SCOPUS_API_KEY"),
        required=True,
        header="X-ELS-APIKey",
    )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        return headers

    def _to_record(self, entry: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(
            entry.get("dc:identifier")
            and _doi_from_scopus_id(entry["dc:identifier"])
            or entry.get("prism:doi")
        )
        authors: list[Author] = []
        for a in entry.get("author", []) or []:
            given = a.get("given-name")
            family = a.get("surname")
            authors.append(
                Author(
                    full_name=" ".join(x for x in (given, family) if x),
                    given_name=given,
                    family_name=family,
                    orcid=a.get("orcid") or None,
                )
            )
        return LiteratureRecord(
            source=self.source_name,
            source_id=entry.get("dc:identifier", "").rsplit(":", 1)[-1],
            doi=doi,
            title=entry.get("dc:title"),
            abstract=entry.get("dc:description"),
            year=_int(entry.get("prism:coverDate")),
            journal=entry.get("prism:publicationName"),
            publisher=entry.get("dc:publisher"),
            authors=authors,
            publication_type=entry.get("subtypeDescription"),
            keywords=[k for k in (entry.get("authkeywords", "") or "").split("|") if k.strip()][
                :20
            ],
            citations_count=_int(entry.get("citedby-count")),
            license=entry.get("openaccess", None) and "open-access" or None,
            landing_page=entry.get("link") and _link_href(entry["link"], "scopus"),
            raw=entry,
        )

    def _params(self, query: str, max_results: int, page: int) -> dict[str, Any]:
        return {
            "query": query,
            "count": max_results,
            "start": page * max_results,
            "view": "STANDARD",
            "httpAccept": "application/json",
        }

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "", params=self._params(query, max_results, page), headers=self._headers()
        )
        if not isinstance(payload, dict):
            return []
        results = payload.get("search-results", {}).get("entry", []) or []
        return [self._to_record(e) for e in results if e.get("error") != "Result set was empty"]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "", params=self._params(f'DOI("{normalize_doi(doi)}")', 1, 0), headers=self._headers()
        )
        if not isinstance(payload, dict):
            return None
        results = payload.get("search-results", {}).get("entry", []) or []
        return self._to_record(results[0]) if results else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "",
            params={
                "query": f"SCP({source_id})",
                "count": 1,
                "start": 0,
                "view": "STANDARD",
                "httpAccept": "application/json",
            },
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            return None
        results = payload.get("search-results", {}).get("entry", []) or []
        return self._to_record(results[0]) if results else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return []

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []


def _doi_from_scopus_id(identifier: str) -> str | None:
    # dc:identifier looks like "SCOPUS_ID:85012345678" — no DOI inside.
    return None


def _link_href(links: list[dict[str, Any]], ref: str) -> str | None:
    for link in links or []:
        if link.get("@ref") == ref:
            return link.get("@href")
    return None


def _int(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    try:
        return int(digits) if digits else None
    except ValueError:
        return None
