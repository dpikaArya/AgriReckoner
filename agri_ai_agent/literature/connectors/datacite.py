"""DataCite connector (https://api.datacite.org, official REST API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class DataCiteConnector(LiteratureConnector):
    source_name = "DataCite"
    display_name = "DataCite"
    base_url = "https://api.datacite.org"
    default_rate_per_minute = 60

    auth = ConnectorAuth(env_vars=("DATACITE_TOKEN", "AGRI_DATACITE_TOKEN"), required=False)

    def _to_record(self, item: dict[str, Any]) -> LiteratureRecord:
        attrs = item.get("attributes", {}) or {}
        doi = normalize_doi(attrs.get("doi"))
        types = attrs.get("types", {}) or {}
        creators = attrs.get("creators", []) or []
        authors = [
            Author(
                full_name=" ".join(
                    x for x in (c.get("givenName", ""), c.get("familyName", "")) if x
                )
                or c.get("name", ""),
                given_name=c.get("givenName"),
                family_name=c.get("familyName"),
                orcid=(c.get("nameIdentifiers") or [{}])[0].get("nameIdentifier", "").replace("https://orcid.org/", "")
                if c.get("nameIdentifiers") else None,
                affiliation=(c.get("affiliation") or [{}])[0].get("name") if c.get("affiliation") else None,
            )
            for c in creators
        ]
        related: list[str] = []
        for rel in attrs.get("relatedIdentifiers", []) or []:
            if rel.get("relatedIdentifier"):
                related.append(
                    normalize_doi(rel["relatedIdentifier"])
                    or rel["relatedIdentifier"]
                )
        url = attrs.get("url")
        pdf_locations: list[PdfLocation] = []
        if url and ".pdf" in (url or "").lower():
            pdf_locations.append(
                PdfLocation(
                    url=url,
                    source_kind="repository",
                    license=attrs.get("rightsList") and attrs["rightsList"][0].get("rightsUri"),
                    content_type="application/pdf",
                )
            )
        return LiteratureRecord(
            source=self.source_name,
            source_id=doi or item.get("id", ""),
            doi=doi,
            title=attrs.get("titles") and attrs["titles"][0].get("title"),
            abstract=attrs.get("descriptions") and _first_description(attrs["descriptions"]),
            year=_int(attrs.get("publicationYear")),
            publisher=attrs.get("publisher"),
            authors=authors,
            publication_type=types.get("resourceTypeGeneral"),
            subjects=[(s.get("subject") or "") for s in attrs.get("subjects", []) or [] if s.get("subject")],
            references=[r for r in related if r],
            related_ids=[r for r in related if r],
            pdf_locations=pdf_locations,
            license=(attrs.get("rightsList") or [{}])[0].get("rights") if attrs.get("rightsList") else None,
            landing_page=url,
            raw=item,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/works",
            params={
                "query": query,
                "page[size]": max_results,
                "page[number]": page + 1,
            },
        )
        if not isinstance(payload, dict):
            return []
        return [self._to_record(item) for item in payload.get("data", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/dois/{normalize_doi(doi)}")
        if not isinstance(payload, dict) or payload.get("errors"):
            return None
        return self._to_record(payload.get("data", {}))

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        return self.lookup_doi(source_id)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return record.references or []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        # DataCite: related identifiers with relation type IsCitedBy / References-reversed.
        attrs = record.raw.get("attributes", {}) or {}
        out: list[str] = []
        for rel in attrs.get("relatedIdentifiers", []) or []:
            if rel.get("relationType") in ("IsCitedBy", "IsReferencedBy"):
                rid = rel.get("relatedIdentifier")
                if rid:
                    out.append(normalize_doi(rid) or rid)
        return out

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        attrs = record.raw.get("attributes", {}) or {}
        out: list[str] = []
        for rel in attrs.get("relatedIdentifiers", []) or []:
            rtype = rel.get("relationType")
            if rtype in ("IsRelatedTo", "IsSupplementedBy", "HasPart", "IsPartOf", "IsVersionOf"):
                rid = rel.get("relatedIdentifier")
                if rid:
                    out.append(normalize_doi(rid) or rid)
        return out


def _first_description(descriptions: list[dict[str, Any]]) -> str | None:
    for d in descriptions:
        if d.get("descriptionType") in ("Abstract", "abstract"):
            return d.get("description")
    return descriptions[0].get("description") if descriptions else None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
