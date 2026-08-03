"""Zenodo connector (https://zenodo.org/api, official public API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class ZenodoConnector(LiteratureConnector):
    source_name = "Zenodo"
    display_name = "Zenodo"
    base_url = "https://zenodo.org/api"
    default_rate_per_minute = 60

    auth = ConnectorAuth(env_vars=("ZENODO_TOKEN", "AGRI_ZENODO_TOKEN"), required=False)

    def _to_record(self, hit: dict[str, Any]) -> LiteratureRecord:
        metadata = hit.get("metadata", {}) or {}
        doi = normalize_doi(metadata.get("doi"))
        creators = metadata.get("creators", []) or []
        authors = [
            Author(
                full_name=" ".join(
                    x for x in (c.get("given_name", ""), c.get("family_name", "")) if x
                )
                or c.get("name", ""),
                orcid=(c.get("orcid", "") or None),
                affiliation=c.get("affiliation"),
            )
            for c in creators
        ]
        pdf_locations: list[PdfLocation] = []
        for file_ in hit.get("files", []) or []:
            if (file_.get("key") or "").lower().endswith(".pdf") and file_.get("links", {}).get("self"):
                pdf_locations.append(
                    PdfLocation(
                        url=file_["links"]["self"],
                        source_kind="repository",
                        license=metadata.get("license", {}).get("id"),
                        content_type="application/pdf",
                        doi=doi,
                        source_repository="Zenodo",
                    )
                )
        related_ids: list[str] = []
        for rel in metadata.get("related_identifiers", []) or []:
            if rel.get("identifier"):
                related_ids.append(normalize_doi(rel["identifier"]) or rel["identifier"])
        return LiteratureRecord(
            source=self.source_name,
            source_id=str(hit.get("id", "")),
            doi=doi,
            title=metadata.get("title"),
            abstract=metadata.get("description"),
            year=_year(metadata.get("publication_date")),
            authors=authors,
            publication_type=metadata.get("resource_type", {}).get("type"),
            language=metadata.get("language"),
            subjects=[(s.get("term") or "") for s in metadata.get("subjects", []) or [] if s.get("term")],
            keywords=metadata.get("keywords", []) or [],
            related_ids=related_ids,
            pdf_locations=pdf_locations,
            license=metadata.get("license", {}).get("id"),
            landing_page=metadata.get("links", {}).get("html") or hit.get("links", {}).get("html"),
            raw=hit,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/records",
            params={"q": query, "size": max_results, "page": page + 1},
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(hit) for hit in payload.get("hits", {}).get("hits", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json("/records", params={"q": f'doi:"{normalize_doi(doi)}"', "size": 1})
        if not isinstance(payload, dict):
            return None
        hits = payload.get("hits", {}).get("hits", []) or []
        return self._to_record(hits[0]) if hits else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/records/{source_id}")
        if not isinstance(payload, dict):
            return None
        return self._to_record(payload)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        # related identifiers with relation "references" / "cites"
        metadata = record.raw.get("metadata", {}) or {}
        out: list[str] = []
        for rel in metadata.get("related_identifiers", []) or []:
            if rel.get("relation") in ("references", "cites", "isCitedBy", "isReferencedBy"):
                out.append(normalize_doi(rel["identifier"]) or rel["identifier"])
        return out

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        metadata = record.raw.get("metadata", {}) or {}
        out: list[str] = []
        for rel in metadata.get("related_identifiers", []) or []:
            if rel.get("relation") in ("isCitedBy", "isReferencedBy"):
                out.append(normalize_doi(rel["identifier"]) or rel["identifier"])
        return out

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return record.related_ids or []


def _year(publication_date: str | None) -> int | None:
    if not publication_date:
        return None
    try:
        return int(publication_date[:4])
    except (TypeError, ValueError):
        return None
