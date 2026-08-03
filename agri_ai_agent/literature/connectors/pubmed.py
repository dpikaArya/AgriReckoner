"""PubMed connector (NCBI E-utilities, official public API).

esearch -> PMIDs; efetch/esummary -> metadata; elink -> related records.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

NS = {
    "m": "http://www.w3.org/1999/xhtml",
    "pubmed": "http://pubmed.org/",
}
_ARTICLE = "{http://pubmed.org/}PubmedArticle"


class PubMedConnector(LiteratureConnector):
    source_name = "PubMed"
    display_name = "PubMed"
    base_url = _BASE
    default_rate_per_minute = 3  # NCBI: 3 req/s without API key

    auth = ConnectorAuth(
        env_vars=("NCBI_API_KEY", "AGRI_NCBI_API_KEY"),
        required=False,
    )

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.auth.available:
            self.default_rate_per_minute = 10  # 10 req/s with API key

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        key = self.auth.resolve()
        if key:
            params["api_key"] = key
        if self.config.contact_email:
            params["tool"] = "AAIF"
            params["email"] = self.config.contact_email
        return params

    def _esearch(self, term: str, max_results: int, start: int) -> list[str]:
        payload = self.http.get_json(
            "/esearch.fcgi",
            params={
                **self._base_params(),
                "db": "pubmed",
                "term": term,
                "retmax": max_results,
                "retstart": start,
                "retmode": "json",
            },
        )
        if not isinstance(payload, dict):
            return []
        return list(payload.get("esearchresult", {}).get("idlist", []) or [])

    def _efetch(self, pmids: list[str]) -> list[dict[str, Any]]:
        if not pmids:
            return []
        xml = self.http.get_text(
            "/efetch.fcgi",
            params={
                **self._base_params(),
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
        if not xml:
            return []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return []
        return [self._parse_article(el) for el in root.iter(_ARTICLE)]

    def _parse_article(self, article: ET.Element) -> dict[str, Any]:
        pmid = article.findtext("MedlineCitation/PMID") or ""
        article_el = article.find("MedlineCitation/Article")
        title = article_el.findtext("ArticleTitle") if article_el is not None else None
        abstract = None
        if article_el is not None:
            abstract_parts = [
                (a.text or "") for a in article_el.findall("Abstract/AbstractText")
            ]
            abstract = " ".join(p.strip() for p in abstract_parts if p.strip()) or None
        year = None
        if article_el is not None:
            date_el = article_el.find("Journal/JournalIssue/PubDate")
            if date_el is not None:
                year = _int(date_el.findtext("Year") or date_el.findtext("MedlineDate") or "")
        journal = article_el.findtext("Journal/Title") if article_el is not None else None
        pub_type = article_el.findtext("PublicationTypeList/PublicationType") if article_el is not None else None

        authors: list[Author] = []
        if article_el is not None:
            for a in article_el.findall("AuthorList/Author"):
                collective = a.findtext("CollectiveName")
                if collective:
                    authors.append(Author(full_name=collective))
                    continue
                given = a.findtext("ForeName")
                family = a.findtext("LastName")
                initials = a.findtext("Initials")
                full = " ".join(x for x in (given, family) if x) or initials or ""
                authors.append(Author(full_name=full, given_name=given, family_name=family))

        refs: list[str] = []
        for ref in article.findall("PubmedData/ReferenceList/Reference"):
            cited = ref.findtext("Citation")
            if cited:
                refs.append(cited)
        article_ids: dict[str, str] = {}
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            idtype = aid.get("IdType", "")
            if idtype and aid.text:
                article_ids[idtype] = aid.text

        doi = normalize_doi(article_ids.get("doi"))
        pdf_locations: list[PdfLocation] = []
        if article_ids.get("pmc"):
            pdf_locations.append(
                PdfLocation(
                    url=f"https://pmc.ncbi.nlm.nih.gov/articles/{article_ids['pmc']}/pdf/",
                    source_kind="repository",
                    source_repository="PubMed Central",
                )
            )
        return {
            "source_id": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "authors": authors,
            "publication_type": pub_type,
            "pdf_locations": pdf_locations,
            "pmcid": article_ids.get("pmc"),
            "references": refs[:50],
            "landing_page": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    def _to_record(self, data: dict[str, Any]) -> LiteratureRecord:
        return LiteratureRecord(
            source=self.source_name,
            source_id=data["source_id"],
            doi=data["doi"],
            title=data["title"],
            abstract=data["abstract"],
            year=data["year"],
            journal=data["journal"],
            authors=data["authors"],
            publication_type=data["publication_type"],
            references=data["references"],
            pdf_locations=data["pdf_locations"],
            landing_page=data["landing_page"],
            raw=data,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        pmids = self._esearch(query, max_results, start=page * max_results)
        return [self._to_record(r) for r in self._efetch(pmids)]

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        pmids = self._esearch(f'"{normalize_doi(doi)}"[doi]', max_results=1, start=0)
        if not pmids:
            return None
        records = self._efetch(pmids[:1])
        return self._to_record(records[0]) if records else None

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        records = self._efetch([source_id])
        return self._to_record(records[0]) if records else None

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return record.references or []

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        # E-utilities has no public 'cited-in' search.
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        payload = self.http.get_json(
            "/elink.fcgi",
            params={
                **self._base_params(),
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": record.source_id,
                "linkname": "pubmed_pubmed",
                "retmode": "json",
            },
        )
        if not isinstance(payload, dict):
            return []
        try:
            linksets = payload["linksets"][0]
            return [
                item["id"]
                for item in linksets.get("linksetdbs", [])[0].get("links", []) or []
            ]
        except (KeyError, IndexError):
            return []


def _int(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    try:
        return int(digits[:4]) if digits else None
    except ValueError:
        return None
