"""Europe PMC connector (https://www.ebi.ac.uk/europepmc/webservices/rest)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class EuropePmcConnector(LiteratureConnector):
    source_name = "Europe_PMC"
    display_name = "Europe PMC"
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=("EUROPEPMC_EMAIL",), required=False)

    def _to_record(self, result: dict[str, Any]) -> LiteratureRecord:
        source = result.get("source", "MED")
        source_id = f"{source}:{result.get('id', '')}"
        doi = normalize_doi(result.get("doi"))
        authors = [
            Author(
                full_name=a.get("fullName", ""),
                orcid=(a.get("authorId", "") or "").replace("ORCID:", "") or None,
                affiliation=a.get("affiliation"),
            )
            for a in result.get("authorList", {}).get("author", []) or []
        ]
        pdf_locations: list[PdfLocation] = []
        for item in (result.get("fullTextUrlList", {}).get("fullTextUrl", []) or []):
            if item.get("documentStyle") == "pdf" and item.get("url"):
                pdf_locations.append(
                    PdfLocation(
                        url=item["url"],
                        source_kind="open_access"
                        if item.get("availability") in ("Open access", "OA")
                        else "repository",
                        license=result.get("license"),
                        content_type="application/pdf",
                        source_repository=item.get("repository", "Europe PMC"),
                    )
                )
        if result.get("pmcid"):
            pdf_locations.append(
                PdfLocation(
                    url=f"https://pmc.ncbi.nlm.nih.gov/articles/{result['pmcid']}/pdf/",
                    source_kind="repository",
                    source_repository="PubMed Central",
                )
            )
        return LiteratureRecord(
            source=self.source_name,
            source_id=source_id,
            doi=doi,
            title=result.get("title"),
            abstract=result.get("abstractText"),
            year=_year(result.get("pubYear")),
            journal=result.get("journalInfo", {}).get("journal", {}).get("title"),
            publisher=None,
            authors=authors,
            publication_type=result.get("pubType"),
            language=result.get("language"),
            keywords=[kw.get("keyword") for kw in (result.get("keywordList", {}).get("keyword", []) or []) if kw.get("keyword")][:20],
            citations_count=result.get("citedByCount"),
            pdf_locations=pdf_locations,
            license=result.get("license"),
            landing_page=result.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
            or result.get("uri"),
            raw=result,
        )

    def _search_params(self, query: str, max_results: int, page: int) -> dict[str, Any]:
        return {"query": query, "pageSize": max_results, "page": page + 1, "format": "json"}

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json("/search", params=self._search_params(query, max_results, page))
        if not isinstance(payload, dict):
            return []
        return [self._to_record(r) for r in payload.get("resultList", {}).get("result", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "/search",
            params={"query": f'DOI:"{normalize_doi(doi)}"', "format": "json", "pageSize": 1},
        )
        if not isinstance(payload, dict):
            return None
        results = payload.get("resultList", {}).get("result", []) or []
        return self._to_record(results[0]) if results else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        if ":" in source_id:
            source, _id = source_id.split(":", 1)
        else:
            source, _id = "MED", source_id
        payload = self.http.get_json(
            "/search",
            params={"query": f'SRC:{source} AND EXT_ID:{_id}', "format": "json", "pageSize": 1},
        )
        if not isinstance(payload, dict):
            return None
        results = payload.get("resultList", {}).get("result", []) or []
        return self._to_record(results[0]) if results else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return self._ids_via_endpoint(record, "references")

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return self._ids_via_endpoint(record, "citations")

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return self._ids_via_endpoint(record, "similarity")

    def _ids_via_endpoint(self, record: LiteratureRecord, endpoint: str) -> list[str]:
        source, _id = record.source_id.split(":", 1)
        payload = self.http.get_json(
            f"/{source}/{_id}/{endpoint}",
            params={"format": "json", "pageSize": 100},
        )
        if not isinstance(payload, dict):
            return []
        out: list[str] = []
        for item in payload.get("citationList", {}).get("citation", []) or []:
            doi = normalize_doi(item.get("doi"))
            out.append(doi or f"{item.get('source')}:{item.get('id')}")
        return out


def _year(pub_year: str | None) -> int | None:
    try:
        return int(pub_year) if pub_year else None
    except (TypeError, ValueError):
        return None
