"""Offline unit tests for connector record parsing and base behaviour.

No network access: fixtures are parsed through the connectors' ``_to_record`` /
``_parse_entry`` mapping methods.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.literature.connector import LiteratureConnector
from agri_ai_agent.literature.connectors.arxiv import ArxivConnector
from agri_ai_agent.literature.connectors.crossref import CrossrefConnector
from agri_ai_agent.literature.connectors.figshare import FigshareConnector
from agri_ai_agent.literature.models import LiteratureRecord
from agri_ai_agent.literature.state import ConnectorStateStore


def _make_connector(connector_cls, tmp_path):
    cfg = LiteratureConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        state_db=tmp_path / "state.sqlite",
        cache_dir=tmp_path / "cache",
        search_terms=("wheat field experiment",),
        max_results_per_query=5,
        max_queries_per_connector=2,
    )
    state = ConnectorStateStore(cfg.state_db)
    return connector_cls(cfg, state)


# ---------------------------------------------------------------------- #
# arXiv
# ---------------------------------------------------------------------- #
_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2210.12345v2</id>
    <title>Nitrogen response of wheat under contrasting sowing dates</title>
    <summary>A field experiment was conducted in RCBD.</summary>
    <published>2022-10-20T12:00:00Z</published>
    <author><name>Sharma, R.</name></author>
    <author><name>Kumar, A.</name></author>
    <category term="q-bio.QM" />
    <link href="http://arxiv.org/abs/2210.12345v2" rel="alternate" type="text/html" />
    <link href="https://arxiv.org/pdf/2210.12345v2" rel="related" title="pdf" />
    <link href="https://doi.org/10.1000/arxiv-paper" rel="related" />
  </entry>
</feed>
"""


def test_arxiv_parse_entry(tmp_path):
    connector = _make_connector(ArxivConnector, tmp_path)
    root = ET.fromstring(_ARXIV_ATOM)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    record = connector._parse_entry(entry)

    assert record.source == "arXiv"
    assert record.source_id == "2210.12345"
    assert record.doi == "10.1000/arxiv-paper"
    assert "nitrogen" in record.title.lower()
    assert record.year == 2022
    assert len(record.authors) == 2
    assert record.authors[0].full_name == "Sharma, R."
    assert record.publication_type == "preprint"
    assert record.pdf_locations[0].url.endswith("2210.12345v2")
    assert record.pdf_locations[0].source_kind == "repository"
    assert record.landing_page == "https://arxiv.org/abs/2210.12345"


# ---------------------------------------------------------------------- #
# Crossref
# ---------------------------------------------------------------------- #
_CROSSREF_WORK = {
    "DOI": "10.1000/example.123",
    "title": ["Fertilizer response of maize in eastern India"],
    "abstract": "<jats:p>Replicated field trials...</jats:p>",
    "author": [
        {
            "given": "Ravi",
            "family": "Singh",
            "ORCID": "https://orcid.org/0000-0001-2345-6789",
            "affiliation": [{"name": "IARI"}],
        }
    ],
    "issued": {"date-parts": [[2021, 3, 15]]},
    "container-title": ["Field Crops Research"],
    "publisher": "Elsevier",
    "type": "journal-article",
    "is-referenced-by-count": 7,
    "reference": [{"DOI": "10.1000/ref1"}, {"unstructured": "Some reference"}],
    "link": [{"URL": "https://doi.org/10.1000/example.123.pdf", "content-type": "application/pdf"}],
    "license": [{"URL": "https://creativecommons.org/licenses/by/4.0", "content-version": "vor"}],
}


def test_crossref_to_record(tmp_path):
    connector = _make_connector(CrossrefConnector, tmp_path)
    record = connector._to_record(_CROSSREF_WORK)

    assert record.source == "Crossref"
    assert record.doi == "10.1000/example.123"
    assert record.title == "Fertilizer response of maize in eastern India"
    assert record.year == 2021
    assert record.journal == "Field Crops Research"
    assert record.authors[0].full_name == "Ravi Singh"
    assert record.authors[0].orcid == "0000-0001-2345-6789"
    assert record.citations_count == 7
    assert "10.1000/ref1" in record.references
    assert len(record.pdf_locations) == 1
    assert record.pdf_locations[0].source_kind == "publisher"
    assert record.pdf_locations[0].license == "CC BY 4.0"
    assert record.landing_page == "https://doi.org/10.1000/example.123"


def test_crossref_no_doi_uses_fallback_id(tmp_path):
    connector = _make_connector(CrossrefConnector, tmp_path)
    work = dict(_CROSSREF_WORK)
    work.pop("DOI")
    record = connector._to_record(work)
    assert record.source_id.startswith("crossref-")


# ---------------------------------------------------------------------- #
# Figshare (keywords are plain strings in the API response)
# ---------------------------------------------------------------------- #
def test_figshare_to_record_with_string_keywords(tmp_path):
    connector = _make_connector(FigshareConnector, tmp_path)
    article = {
        "id": 12345,
        "title": "Wheat experiment dataset",
        "description": "Data from a field experiment.",
        "published_date": "2020-05-01T00:00:00Z",
        "authors": [{"full_name": "Jane Doe", "orcid_id": "0000-0002-0000-0000"}],
        "subjects": [{"name": "Agriculture"}, {"name": "Agronomy"}],
        "keywords": ["wheat", "nitrogen"],
        "doi": "10.6084/m9.figshare.12345",
        "defined_type_name": "dataset",
        "url_public_html": "https://figshare.com/articles/12345",
        "files": [
            {"name": "data.pdf", "download_url": "https://figshare.com/ndownloader/files/1.pdf"}
        ],
    }
    record = connector._to_record(article)

    assert record.source == "Figshare"
    assert record.source_id == "12345"
    assert record.doi == "10.6084/m9.figshare.12345"
    assert record.keywords == ["wheat", "nitrogen"]
    assert record.subjects == ["Agriculture", "Agronomy"]
    assert len(record.pdf_locations) == 1
    assert record.publication_type == "dataset"


# ---------------------------------------------------------------------- #
# base connector: validation + incremental sync caching
# ---------------------------------------------------------------------- #
class _StubConnector(LiteratureConnector):
    source_name = "Stub"
    base_url = "https://example.invalid"

    def __init__(self, config, state, records):
        self._records = list(records)
        super().__init__(config, state)

    def search(self, query, max_results=25, **kwargs):
        return self._records[:max_results]

    def lookup_doi(self, doi):
        return None

    def fetch_metadata(self, source_id):
        return None

    def fetch_pdf_location(self, record):
        return []

    def fetch_references(self, record):
        return []

    def fetch_citations(self, record):
        return []

    def fetch_related_articles(self, record):
        return []


def test_validate_record(tmp_path):
    cfg = LiteratureConfig(
        data_dir=tmp_path / "data",
        state_db=tmp_path / "state.sqlite",
        cache_dir=tmp_path / "cache",
    )
    connector = _StubConnector(cfg, ConnectorStateStore(cfg.state_db), [])
    good = LiteratureRecord(source="S", source_id="1", doi="10.1000/x", title="A title")
    assert connector.validate_record(good) == []

    bad = LiteratureRecord(source="S", source_id="", doi=None, title=None)
    errors = connector.validate_record(bad)
    assert len(errors) >= 2

    weird_year = LiteratureRecord(source="S", source_id="1", title="T", year=99)
    assert any("year" in e for e in connector.validate_record(weird_year))


def test_incremental_sync_caches_and_dedupes(tmp_path):
    cfg = LiteratureConfig(
        data_dir=tmp_path / "data",
        state_db=tmp_path / "state.sqlite",
        cache_dir=tmp_path / "cache",
        search_terms=("wheat",),
        max_results_per_query=3,
        max_queries_per_connector=1,
    )
    state = ConnectorStateStore(cfg.state_db)
    records = [
        LiteratureRecord(source="Stub", source_id=f"id-{i}", title=f"Paper {i}") for i in range(3)
    ]
    connector = _StubConnector(cfg, state, records)

    first = connector.incremental_sync()
    assert first.new_records == 3
    assert first.fetched_records == 3
    assert state.cache_count("Stub") == 3

    second = connector.incremental_sync()
    assert second.new_records == 0
    assert second.skipped_records == 3
    assert connector.get_cursor("last_sync_at") is not None
