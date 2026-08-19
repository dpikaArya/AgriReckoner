"""Dimensions connector (api.dimensions.ai/v1/slate, official GraphQL API).

Requires a Dimensions bearer token (DIMENSIONS_TOKEN).
"""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation

_SEARCH_QUERY = """
query search($query: String!, $limit: Int!, $skip: Int!) {
  publications(
    search: $query
    limit: $limit
    skip: $skip
    order: [relevance DESC]
  ) {
    publications {
      id
      doi
      title
      year
      type
      journal { title }
      publisher
      authors { first_name last_name orcid }
      abstract { text }
      fields { name }
      times_cited
    }
  }
}
"""


class DimensionsConnector(LiteratureConnector):
    source_name = "Dimensions"
    display_name = "Dimensions"
    base_url = "https://api.dimensions.ai/v1/slate"
    default_rate_per_minute = 30

    auth = ConnectorAuth(
        env_vars=("DIMENSIONS_TOKEN", "AGRI_DIMENSIONS_TOKEN"),
        required=True,
    )

    def _headers(self) -> dict[str, str]:
        token = self.auth.resolve()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _run_query(self, query: str, limit: int, skip: int) -> list[dict[str, Any]]:
        payload = self.http.post_json(
            "",
            payload={
                "query": _SEARCH_QUERY,
                "variables": {"query": query, "limit": limit, "skip": skip},
            },
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            return []
        errors = payload.get("errors")
        if errors:
            self.logger.debug("Dimensions GraphQL errors: %s", errors)
            return []
        data = payload.get("data", {}).get("publications", {}).get("publications", []) or []
        return list(data)

    def _to_record(self, pub: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(pub.get("doi"))
        authors = [
            Author(
                full_name=" ".join(
                    x for x in (a.get("first_name", ""), a.get("last_name", "")) if x
                ),
                given_name=a.get("first_name"),
                family_name=a.get("last_name"),
                orcid=a.get("orcid"),
            )
            for a in pub.get("authors", []) or []
        ]
        return LiteratureRecord(
            source=self.source_name,
            source_id=pub.get("id", ""),
            doi=doi,
            title=pub.get("title"),
            abstract=(pub.get("abstract") or {}).get("text"),
            year=pub.get("year"),
            journal=(pub.get("journal") or {}).get("title"),
            publisher=pub.get("publisher"),
            authors=authors,
            publication_type=pub.get("type"),
            subjects=[f.get("name") for f in pub.get("fields", []) or [] if f.get("name")],
            citations_count=pub.get("times_cited"),
            landing_page=f"https://doi.org/{doi}" if doi else None,
            raw=pub,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        pubs = self._run_query(query, max_results, skip=page * max_results)
        return [self._to_record(p) for p in pubs]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        pubs = self._run_query(f'doi:"{normalize_doi(doi)}"', 1, 0)
        return self._to_record(pubs[0]) if pubs else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        pubs = self._run_query(f'id:"{source_id}"', 1, 0)
        return self._to_record(pubs[0]) if pubs else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return []

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []
