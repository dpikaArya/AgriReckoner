"""Unit tests for query construction and study classification heuristics."""

from __future__ import annotations

from agri_ai_agent.literature import connector as cconnector
from agri_ai_agent.literature.classify import (
    classify_publication_type,
    detect_crops,
    detect_experimental_designs,
    detect_study_variables,
    enrich_record,
    is_original_experimental_study,
)
from agri_ai_agent.literature.models import LiteratureRecord
from agri_ai_agent.literature.query import (
    build_boolean_queries,
    build_simple_queries,
    default_search_terms,
)


# ---------------------------------------------------------------------- #
# DOI helpers
# ---------------------------------------------------------------------- #
def test_normalize_doi_variants():
    assert cconnector.normalize_doi("10.1234/ABCD.EF") == "10.1234/abcd.ef"
    assert cconnector.normalize_doi("https://doi.org/10.1234/x") == "10.1234/x"
    assert cconnector.normalize_doi("http://doi.org/10.1234/x") == "10.1234/x"
    assert cconnector.normalize_doi("doi:10.1234/x") == "10.1234/x"
    assert cconnector.normalize_doi(" 10.1234/x ") == "10.1234/x"
    assert cconnector.normalize_doi("not-a-doi") is None
    assert cconnector.normalize_doi("") is None
    assert cconnector.normalize_doi(None) is None


def test_extract_dois():
    text = "See 10.1234/first and https://doi.org/10.5678/second and junk."
    dois = cconnector.extract_dois(text)
    assert dois == ["10.1234/first", "10.5678/second"]
    assert cconnector.extract_dois("") == []
    assert cconnector.extract_dois(None) == []


def test_validate_doi_syntax():
    assert cconnector.validate_doi_syntax("10.1000/xyz") is True
    assert cconnector.validate_doi_syntax("10.1000") is False


# ---------------------------------------------------------------------- #
# query construction
# ---------------------------------------------------------------------- #
def test_build_boolean_queries_count():
    queries = build_boolean_queries()
    assert len(queries) == 9  # one per study-design term
    assert all("AND" in q for q in queries)


def test_build_simple_queries_count():
    queries = build_simple_queries()
    assert len(queries) == len(("field experiments", "field trials", "RCBD", "randomized complete block", "split plot", "split-split plot", "factorial experiments", "strip plot", "latin square")) * 6
    assert all("AND" in q for q in queries)


def test_default_search_terms_prefers_config():
    assert default_search_terms(("wheat",)) == ("wheat",)
    terms = default_search_terms()
    assert len(terms) > 0


# ---------------------------------------------------------------------- #
# classification
# ---------------------------------------------------------------------- #
def test_classify_publication_type():
    assert classify_publication_type("journal article") == "journal-article"
    assert classify_publication_type("Review") == "review"
    assert classify_publication_type("Editorial Material") == "editorial"
    assert classify_publication_type("Meeting Abstract") == "conference-abstract"
    assert classify_publication_type("Patent") == "patent"
    assert classify_publication_type("Corrigendum") == "erratum"
    assert classify_publication_type("preprint") == "preprint"
    assert classify_publication_type(None) is None


def _record(abstract: str = "", ptype: str | None = None) -> LiteratureRecord:
    return LiteratureRecord(
        source="test",
        source_id="1",
        title="A randomized complete block field experiment on wheat",
        abstract=abstract,
        publication_type=ptype,
    )


def test_review_excluded():
    rec = LiteratureRecord(
        source="test", source_id="2", title="This review of wheat yields",
        abstract="We review the literature.", publication_type="journal-article",
    )
    assert is_original_experimental_study(rec) is False


def test_conference_abstract_excluded():
    assert is_original_experimental_study(_record(ptype="conference-abstract")) is False


def test_patent_excluded():
    assert is_original_experimental_study(_record(ptype="patent")) is False


def test_simulation_only_excluded():
    rec = LiteratureRecord(
        source="test", source_id="3", title="A computational model of wheat growth",
        abstract="We simulated yield under climate scenarios using APSIM.",
        publication_type="journal-article",
    )
    assert is_original_experimental_study(rec) is False


def test_simulation_with_experiment_accepted():
    rec = LiteratureRecord(
        source="test", source_id="4", title="Field trial and modelling of wheat",
        abstract="We calibrated a model with data from replicated field experiments.",
        publication_type="journal-article",
    )
    assert is_original_experimental_study(rec) is True


def test_field_experiment_accepted():
    rec = _record(
        abstract="A split plot trial with three nitrogen rates in RCBD was conducted.",
        ptype="journal-article",
    )
    assert is_original_experimental_study(rec) is True


def test_detect_designs_variables_crops():
    rec = LiteratureRecord(
        source="test", source_id="5",
        title="Effects of nitrogen and irrigation on rice yield in a split plot design",
        abstract="",
    )
    assert detect_experimental_designs(rec) == ["Split Plot"]
    variables = detect_study_variables(rec)
    assert "Nitrogen" in variables
    assert "Irrigation" in variables
    assert detect_crops(rec) == ["Rice"]


def test_enrich_record():
    rec = LiteratureRecord(
        source="test", source_id="6",
        title="Nitrogen response of maize in randomized complete block field trials",
        abstract="",
        publication_type="journal article",
    )
    enriched = enrich_record(rec)
    assert enriched.publication_type == "journal-article"
    assert enriched.is_original_study is True
    assert "Randomized Complete Block Design" in enriched.experimental_design
    assert "Maize" in enriched.crop_terms
    assert enriched.study_variables == ["Nitrogen"]
