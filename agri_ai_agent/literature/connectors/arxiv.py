"""arXiv connector (export.arxiv.org, official public API)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation

_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(LiteratureConnector):
    source_name = "arXiv"
    display_name = "arXiv"
    base_url = "https://export.arxiv.org/api"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    def _query(self, query: str, max_results: int, start: int) -> list[LiteratureRecord]:
        xml = self.http.get_text(
            "/query",
            params={
                "search_query": query,
                "max_results": max_results,
                "start": start,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        if not xml:
            return []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []
        records: list[LiteratureRecord] = []
        for entry in root.iter(f"{_ATOM}entry"):
            records.append(self._parse_entry(entry))
        return records

    def _parse_entry(self, entry: ET.Element) -> LiteratureRecord:
        arxiv_id = (entry.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1].split("v", 1)[0]
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
        published = entry.findtext(f"{_ATOM}published")
        authors = [
            Author(full_name=" ".join((a.findtext(f"{_ATOM}name") or "").split()))
            for a in entry.findall(f"{_ATOM}author")
        ]
        subjects = [
            (c.get("term") or "")
            for c in entry.findall(f"{_ATOM}category")
            if c.get("term")
        ]
        doi = None
        pdf_url = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("rel") == "related" and link.get("href"):
                doi = normalize_doi(link.get("href"))
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_locations = (
            [
                PdfLocation(
                    url=pdf_url,
                    source_kind="repository",
                    content_type="application/pdf",
                    source_repository="arXiv",
                )
            ]
            if pdf_url
            else []
        )
        return LiteratureRecord(
            source=self.source_name,
            source_id=arxiv_id,
            doi=doi,
            title=title or None,
            abstract=summary or None,
            year=_year(published),
            journal="arXiv",
            authors=authors,
            publication_type="preprint",
            subjects=subjects,
            pdf_locations=pdf_locations,
            landing_page=f"https://arxiv.org/abs/{arxiv_id}",
            raw={"entry_id": entry.findtext(f"{_ATOM}id")},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        return self._query(query, max_results, start=page * max_results)

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        records = self._query(f'doi:"{normalize_doi(doi)}"', 1, 0)
        return records[0] if records else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        records = self._query(f'id:{source_id}', 1, 0)
        return records[0] if records else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []  # arXiv API does not expose a reference list

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []


def _year(published: str | None) -> int | None:
    if not published:
        return None
    try:
        return int(published[:4])
    except (TypeError, ValueError):
        return None
