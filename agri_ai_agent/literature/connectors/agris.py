"""AGRIS connector (FAO) via the official OAI-PMH endpoint.

The AGRIS search REST API returns HTTP 403 to non-browser clients, so this
connector harvests the official OAI-PMH endpoint (``https://agris.fao.org/oai``)
instead.  ``search()`` performs a local keyword filter over harvested Dublin
Core records; ``incremental_sync`` uses a resumption-token cursor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, extract_dois
from agri_ai_agent.literature.models import (
    Author,
    LiteratureRecord,
    PdfLocation,
    SyncResult,
)

_DC_NS = "{http://purl.org/dc/elements/1.1/}"
_OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"


class AgrisConnector(LiteratureConnector):
    source_name = "AGRIS"
    display_name = "AGRIS (FAO)"
    base_url = "https://agris.fao.org"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    # ------------------------------------------------------------------ #
    # OAI helpers
    # ------------------------------------------------------------------ #
    def _list_records_page(
        self,
        resumption_token: str | None = None,
        *,
        from_date: str | None = None,
    ) -> tuple[list[LiteratureRecord], str | None]:
        params: dict[str, str] = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        if resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": resumption_token}
        elif from_date:
            params["from"] = from_date
        xml = self.http.get_text(
            "/oai", params=params, use_cache=False,
        )
        if not xml:
            return [], None
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return [], None
        records: list[LiteratureRecord] = []
        for record in root.iter(f"{_OAI_NS}record"):
            header = record.find(f"{_OAI_NS}header")
            metadata = record.find(f"{_OAI_NS}metadata")
            if header is None or metadata is None:
                continue
            ident = header.findtext(f"{_OAI_NS}identifier") or ""
            status = header.get("status")
            if status == "deleted":
                continue
            dc = metadata.find(f"{_DC_NS}dc") or metadata
            records.append(self._parse_dc(ident, dc))
        token = root.findtext(f".//{_OAI_NS}resumptionToken")
        return records, token or None

    def _parse_dc(self, ident: str, dc: ET.Element) -> LiteratureRecord:
        def values(tag: str) -> list[str]:
            return [
                (el.text or "").strip()
                for el in dc.findall(f"{_DC_NS}{tag}")
                if el.text and el.text.strip()
            ]

        titles = values("title")
        title = titles[0] if titles else None
        abstract = "\n".join(values("description")) or None
        year = _parse_year(values("date"))
        authors = [Author(full_name=a) for a in values("creator")]
        do_is = extract_dois("\n".join(values("identifier")))
        doi = do_is[0] if do_is else None
        pdf_locations: list[PdfLocation] = []
        for ident in values("identifier"):
            if ".pdf" in ident.lower():
                pdf_locations.append(
                    PdfLocation(
                        url=ident,
                        source_kind="open_access",
                        content_type="application/pdf",
                        source_repository="AGRIS",
                    )
                )
        journal = _find_journal(values("source"))
        language = values("language")
        return LiteratureRecord(
            source=self.source_name,
            source_id=ident,
            doi=doi,
            title=title,
            abstract=abstract,
            year=year,
            journal=journal,
            authors=authors,
            publication_type=None,
            language=language[0] if language else None,
            subjects=values("subject"),
            keywords=values("subject"),
            pdf_locations=pdf_locations,
            landing_page=f"https://agris.fao.org/agris-search/search.do?recordID={ident.rsplit('/', 1)[-1]}",
            raw={"oai_identifier": ident},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        tokens = query.lower().split()
        records, _ = self._list_records_page()
        hits: list[LiteratureRecord] = []
        for record in records:
            haystack = " ".join(
                filter(None, [record.title, record.abstract] + record.subjects)
            ).lower()
            if all(t in haystack for t in tokens):
                hits.append(record)
            if len(hits) >= max_results:
                break
        return hits

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        # OAI-PMH has no DOI index; scan a recent window and match locally.
        from agri_ai_agent.literature.connector import normalize_doi

        target = normalize_doi(doi)
        records, token = self._list_records_page()
        while records and token:
            for record in records:
                if record.doi and normalize_doi(record.doi) == target:
                    return record
            records, token = self._list_records_page(token)
        return None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        xml = self.http.get_text(
            "/oai",
            params={"verb": "GetRecord", "identifier": source_id, "metadataPrefix": "oai_dc"},
            use_cache=False,
        )
        if not xml:
            return None
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return None
        record = root.find(f".//{_OAI_NS}record")
        if record is None:
            return None
        dc = record.find(f"{_OAI_NS}metadata/{_DC_NS}dc") or record.find(f"{_OAI_NS}metadata")
        return self._parse_dc(source_id, dc)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []

    # ------------------------------------------------------------------ #
    # incremental sync: resumption-token based harvest
    # ------------------------------------------------------------------ #
    def incremental_sync(
        self,
        max_records: int | None = None,
        terms: tuple[str, ...] | None = None,
    ) -> SyncResult:
        result = SyncResult(connector=self.source_name)
        budget = max_records or (self.config.max_results_per_query * self.config.max_queries_per_connector)
        from_date = self.get_cursor("last_from_date")
        current_from = from_date or "1995-01-01"
        token = self.get_cursor("resumption_token")
        try:
            while budget > 0:
                records, next_token = self._list_records_page(
                    token if token else None, from_date=None if token else current_from
                )
                if not records:
                    break
                for record in records:
                    if budget <= 0:
                        break
                    result.fetched_records += 1
                    if self.state.cache_has(self.source_name, record.source_id):
                        result.skipped_records += 1
                    else:
                        self.state.cache_put(self.source_name, record.source_id, record.to_dict())
                        result.new_records += 1
                    budget -= 1
                if not next_token:
                    break
                token = next_token
                self.set_cursor("resumption_token", token)
            self.set_cursor("last_from_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            result.cursor = current_from
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("AGRIS incremental_sync failed")
            result.errors.append(str(exc))
        result.completed_at = result.started_at
        return result


def _parse_year(dates: list[str]) -> int | None:
    for value in dates:
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) >= 4:
            try:
                return int(digits[:4])
            except ValueError:
                continue
    return None


def _find_journal(sources: list[str]) -> str | None:
    for value in sources:
        if "[" in value and "]" in value:
            return value.strip("[]")
    return sources[0] if sources else None
