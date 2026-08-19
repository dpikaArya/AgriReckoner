"""Web of Science Starter API connector (api.clarivate.com, official API).

Requires an institutional Web of Science Starter API key (X-ApiKey header).
"""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class WosConnector(LiteratureConnector):
    source_name = "Web_of_Science"
    display_name = "Web of Science (Starter API)"
    base_url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
    default_rate_per_minute = 30

    auth = ConnectorAuth(
        env_vars=("WOS_API_KEY", "AGRI_WOS_API_KEY"),
        required=True,
        header="X-ApiKey",
    )

    def _to_record(self, record: dict[str, Any]) -> LiteratureRecord:
        identifiers = record.get("identifiers", {}) or {}
        doi = normalize_doi(identifiers.get("doi"))
        authors = [
            Author(
                full_name=" ".join(x for x in (a.get("firstName", ""), a.get("lastName", "")) if x)
                or a.get("fullName", ""),
                given_name=a.get("firstName"),
                family_name=a.get("lastName"),
                orcid=(a.get("orcid", "") or None),
                affiliation=(a.get("addresses") or [{}])[0].get("country")
                if a.get("addresses")
                else None,
            )
            for a in record.get("authors", []) or []
        ]
        source = record.get("source", {}) or {}
        return LiteratureRecord(
            source=self.source_name,
            source_id=record.get("uid", ""),
            doi=doi,
            title=record.get("title"),
            abstract=record.get("abstract"),
            year=record.get("publicationYear"),
            journal=source.get("title"),
            publisher=source.get("publisher"),
            authors=authors,
            publication_type=record.get("sourceTypes") and record["sourceTypes"][0],
            language=record.get("language"),
            keywords=[kw for kw in (record.get("keywords") or []) if kw][:20],
            citations_count=record.get("citations"),
            landing_page=f"https://doi.org/{doi}" if doi else None,
            raw=record,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "",
            params={"q": query, "limit": max_results, "page": page + 1},
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(r) for r in payload.get("hits", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "",
            params={"q": f"DO:{normalize_doi(doi)}", "limit": 1, "page": 1},
        )
        if not isinstance(payload, dict):
            return None
        hits = payload.get("hits", []) or []
        return self._to_record(hits[0]) if hits else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json("", params={"q": f"UT:{source_id}", "limit": 1, "page": 1})
        if not isinstance(payload, dict):
            return None
        hits = payload.get("hits", []) or []
        return self._to_record(hits[0]) if hits else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        # WoS metadata does not include full-text URLs.
        return []

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []
