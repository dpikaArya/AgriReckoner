"""Crossref connector (https://api.crossref.org, official REST API)."""

from __future__ import annotations

from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class CrossrefConnector(LiteratureConnector):
    source_name = "Crossref"
    display_name = "Crossref"
    base_url = "https://api.crossref.org"
    contact_email = "mailto:aaif@agriculture.intelligence"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=("CROSSREF_MAILTO",), required=False)

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.config.contact_email
            or "AAIF/1.0 (mailto:aaif@agriculture.intelligence)"
        }

    # ------------------------------------------------------------------ #
    # record mapping
    # ------------------------------------------------------------------ #
    def _to_record(self, work: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(work.get("DOI"))
        authors: list[Author] = []
        for a in work.get("author", []) or []:
            authors.append(
                Author(
                    full_name=" ".join(x for x in (a.get("given", ""), a.get("family", "")) if x),
                    given_name=a.get("given"),
                    family_name=a.get("family"),
                    orcid=(a.get("ORCID", "") or "")
                    .replace("http://orcid.org/", "")
                    .replace("https://orcid.org/", "")
                    or None,
                    affiliation=(a.get("affiliation") or [{}])[0].get("name"),
                )
            )

        refs = [
            (r.get("DOI") or r.get("unstructured") or "") for r in work.get("reference", []) or []
        ]
        refs = [normalize_doi(r) or r for r in refs if r]

        pdf_locations: list[PdfLocation] = []
        for link in work.get("link", []) or []:
            if link.get("content-type") in ("application/pdf", "pdf") and link.get("URL"):
                pdf_locations.append(
                    PdfLocation(
                        url=link["URL"],
                        source_kind="publisher",
                        license=_license_from_work(work),
                        content_type="application/pdf",
                        doi=doi,
                        source_repository=work.get("container-title")
                        and work["container-title"][0]
                        or "Crossref",
                    )
                )
        license_url = None
        for lic in work.get("license", []) or []:
            if lic.get("URL") and not lic.get("content-version") == "tdm":
                license_url = lic["URL"]
                break

        return LiteratureRecord(
            source=self.source_name,
            source_id=doi or f"crossref-{work.get('_id', '')}",
            doi=doi,
            title=_first(work.get("title")),
            abstract=work.get("abstract"),
            year=_year(work),
            journal=_first(work.get("container-title")),
            publisher=work.get("publisher"),
            authors=authors,
            publication_type=_first(work.get("type")) if work.get("type") else None,
            language=work.get("language"),
            subjects=[(s.get("name") or "") for s in work.get("subject", []) or [] if s],
            keywords=[kw for kw in (work.get("keyword") or "").split(";") if kw.strip()][:20],
            references=refs[:50],
            citations_count=work.get("is-referenced-by-count"),
            pdf_locations=pdf_locations,
            license=license_url,
            landing_page=f"https://doi.org/{doi}" if doi else None,
            raw=work,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        params = {
            "query.bibliographic": query,
            "rows": max_results,
            "offset": page * max_results,
            "select": "DOI,title,author,issued,container-title,publisher,type,is-referenced-by-count",
        }
        if self.config.contact_email:
            params["mailto"] = self.config.contact_email
        payload = self.http.get_json("/works", params=params, headers=self._headers())
        if not isinstance(payload, dict):
            return []
        return [self._to_record(w) for w in payload.get("message", {}).get("items", []) or []]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/works/{normalize_doi(doi)}", headers=self._headers())
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return None
        return self._to_record(payload.get("message", {}))

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        doi = normalize_doi(source_id)
        if not doi:
            return None
        return self.lookup_doi(doi)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return record.references or []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        # Crossref does not expose citing works through the public API.
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        # Crossref 'relation' field can list related works (rarely populated).
        related: list[str] = []
        relations = record.raw.get("relation", {}) or {}
        for _relation_type, items in relations.items():
            for item in items or []:
                related.append(normalize_doi(item.get("id")) or item.get("id", ""))
        return [r for r in related if r]


def _first(values: Any) -> str | None:
    if isinstance(values, str):
        return values or None
    if isinstance(values, list) and values:
        return str(values[0])
    return None


def _year(work: dict[str, Any]) -> int | None:
    issued = work.get("issued") or {}
    parts = issued.get("date-parts") or [[None]]
    try:
        return int(parts[0][0])
    except (TypeError, ValueError, IndexError):
        return None


def _license_from_work(work: dict[str, Any]) -> str | None:
    for lic in work.get("license", []) or []:
        url = lic.get("URL")
        if url and "creativecommons" in url.lower():
            # https://creativecommons.org/licenses/by/4.0 -> "CC BY 4.0"
            parts = url.rstrip("/").split("/")
            try:
                idx = parts.index("licenses")
                family, version = parts[idx + 1], parts[idx + 2]
                return f"CC {family.upper()} {version}"
            except (ValueError, IndexError):
                return url
    return None
