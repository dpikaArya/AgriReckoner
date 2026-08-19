"""Figshare connector (https://api.figshare.com/v2, official public API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class FigshareConnector(LiteratureConnector):
    source_name = "Figshare"
    display_name = "Figshare"
    base_url = "https://api.figshare.com/v2"
    default_rate_per_minute = 60

    auth = ConnectorAuth(env_vars=("FIGSHARE_TOKEN", "AGRI_FIGSHARE_TOKEN"), required=False)

    def _to_record(self, article: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(article.get("doi"))
        authors = [
            Author(
                full_name=a.get("full_name", "") if isinstance(a, dict) else str(a),
                orcid=((a.get("orcid_id") or "") or None) if isinstance(a, dict) else None,
            )
            for a in article.get("authors", []) or []
        ]
        pdf_locations: list[PdfLocation] = []
        for file_ in article.get("files", []) or []:
            if (file_.get("name") or "").lower().endswith(".pdf") and file_.get("download_url"):
                pdf_locations.append(
                    PdfLocation(
                        url=file_["download_url"],
                        source_kind="repository",
                        content_type="application/pdf",
                        doi=doi,
                        source_repository="Figshare",
                    )
                )
        subjects = [
            s.get("name") if isinstance(s, dict) else str(s)
            for s in article.get("subjects", []) or []
        ]
        keywords = [
            k if isinstance(k, str) else (k.get("name") if isinstance(k, dict) else str(k))
            for k in article.get("keywords", []) or []
        ]
        return LiteratureRecord(
            source=self.source_name,
            source_id=str(article.get("id", "")),
            doi=doi,
            title=article.get("title"),
            abstract=article.get("description"),
            year=_year(article.get("published_date")),
            publisher="Figshare",
            authors=authors,
            publication_type=article.get("defined_type_name"),
            subjects=[s for s in subjects if s],
            keywords=[k for k in keywords if k][:20],
            pdf_locations=pdf_locations,
            landing_page=article.get("url_public_html"),
            raw=article,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.post_json(
            "/articles/search",
            payload={
                "search_for": query,
                "page_size": max_results,
                "page": page + 1,
            },
        )
        if not isinstance(payload, list):
            return []
        records: list[LiteratureRecord] = []
        for item in payload:
            record = self._fetch_detail(item.get("id"))
            if record:
                records.append(record)
            if len(records) >= max_results:
                break
        return records

    def _fetch_detail(self, article_id: Any) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/articles/{article_id}")
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.post_json(
            "/articles/search",
            payload={"doi": normalize_doi(doi), "page_size": 1, "page": 1},
        )
        if not isinstance(payload, list) or not payload:
            return None
        return self._fetch_detail(payload[0].get("id"))

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        return self._fetch_detail(source_id)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        # Figshare search API has no reference list; links in the description are
        # the only provider-supplied signal.
        from agri_ai_agent.literature.connector import extract_dois

        return extract_dois(record.abstract) or []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []


def _year(published_date: str | None) -> int | None:
    if not published_date:
        return None
    try:
        return int(published_date[:4])
    except (TypeError, ValueError):
        return None
