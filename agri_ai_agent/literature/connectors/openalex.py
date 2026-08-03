"""OpenAlex connector (https://api.openalex.org, official REST API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import (
    Author,
    LiteratureRecord,
    PdfLocation,
)

_OA_LANDING = "https://openalex.org"


class OpenAlexConnector(LiteratureConnector):
    source_name = "OpenAlex"
    display_name = "OpenAlex"
    base_url = "https://api.openalex.org"
    contact_email = "mailto:openalex@ourresearch.org"
    default_rate_per_minute = 100

    auth = ConnectorAuth(env_vars=("OPENALEX_API_KEY", "AGRI_OPENALEX_API_KEY"), required=False)

    # ------------------------------------------------------------------ #
    # record mapping
    # ------------------------------------------------------------------ #
    def _to_record(self, work: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(work.get("doi"))
        authors: list[Author] = []
        for a in work.get("authorships", []) or []:
            author = a.get("author") or {}
            raw_name = author.get("display_name") or author.get("id") or ""
            orcid = (author.get("orcid") or "").replace("https://orcid.org/", "") or None
            affiliation = ""
            if a.get("institutions"):
                affiliation = "; ".join(
                    inst.get("display_name", "") for inst in a["institutions"] if inst.get("display_name")
                )
            authors.append(
                Author(full_name=raw_name, orcid=orcid, affiliation=affiliation or None)
            )

        subjects = [
            (c.get("display_name") or c.get("name") or "")
            for c in work.get("topics", []) or []
            if c.get("display_name")
        ]
        keywords = [
            kw.get("display_name", "")
            for kw in work.get("keywords", []) or []
            if kw.get("display_name")
        ]

        open_access = work.get("open_access") or {}
        oa_url = open_access.get("oa_url")
        best_location = work.get("best_oa_location") or {}
        pdf_url = best_location.get("pdf_url") or (oa_url if oa_url and ".pdf" in (oa_url or "").lower() else None)

        pdf_locations: list[PdfLocation] = []
        if pdf_url:
            pdf_locations.append(
                PdfLocation(
                    url=pdf_url,
                    source_kind="open_access",
                    license=(best_location.get("license") or open_access.get("license_id")),
                    source_repository="OpenAlex",
                )
            )
        # primary location as landing page
        primary = work.get("primary_location") or {}
        landing_page = (
            primary.get("landing_page_url")
            or (oa_url or "")
            or f"{_OA_LANDING}/works/{work.get('id', '').rsplit('/', 1)[-1]}"
        )

        return LiteratureRecord(
            source=self.source_name,
            source_id=str(work.get("id", "").rsplit("/", 1)[-1]),
            doi=doi,
            title=work.get("title"),
            abstract=self._abstract_from_inverted(work.get("abstract_inverted_index")),
            year=(work.get("publication_year") or None),
            journal=(primary.get("source") or {}).get("display_name"),
            publisher=(primary.get("source") or {}).get("host_organization_name")
            or ((primary.get("source") or {}).get("publisher") or ""),
            authors=authors,
            publication_type=_openalex_type(work.get("type")),
            language=work.get("language"),
            subjects=subjects,
            keywords=keywords,
            references=[normalize_doi(r) or r for r in (work.get("referenced_works") or [])][:50],
            citations_count=work.get("cited_by_count"),
            related_ids=[str(r).rsplit("/", 1)[-1] for r in (work.get("related_works") or [])][:50],
            pdf_locations=pdf_locations,
            license=open_access.get("oa_status"),
            landing_page=landing_page,
            raw=work,
        )

    @staticmethod
    def _abstract_from_inverted(inverted: dict[str, Any] | None) -> str | None:
        if not inverted:
            return None
        words: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            for pos in positions:
                words.append((pos, word))
        words.sort(key=lambda t: t[0])
        return " ".join(w for _, w in words)

    def _search_params(self, query: str, max_results: int, page: int) -> dict[str, Any]:
        return {
            "search": query,
            "per-page": max_results,
            "page": page + 1,
            "mailto": self.config.contact_email or "aaif@agriculture.intelligence",
        }

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/works", params=self._search_params(query, max_results, page)
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(w) for w in payload.get("results", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/works/doi/{normalize_doi(doi)}")
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/works/{source_id}")
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return record.references or []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        # OpenAlex: works that cite this work via the cites filter.
        oa_id = f"https://openalex.org/W{record.source_id}" if not record.source_id.startswith("W") else f"https://openalex.org/{record.source_id}"
        payload = self.http.get_json(
            "/works", params={"filter": f"cites:{oa_id}", "per-page": 25}
        )
        if not isinstance(payload, dict):
            return []
        return [str(w.get("doi", "")).replace("https://doi.org/", "") or w.get("id") for w in payload.get("results", []) or []]

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return record.related_ids or []


def _openalex_type(work_type: str | None) -> str | None:
    if not work_type:
        return None
    return {
        "article": "journal-article",
        "review": "review",
        "editorial": "editorial",
        "letter": "letter",
        "preprint": "preprint",
        "book-chapter": "book-chapter",
        "book": "book",
        "paratext": "other",
        "other": "other",
    }.get(work_type, work_type)
