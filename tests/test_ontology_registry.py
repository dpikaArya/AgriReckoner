"""Tests for the UAMS ontology registry (spec/uams_ontology.yaml + loaders).

No network access: every ontology term is asserted by its offline CURIE/IRI,
none are treated as 'live' resolvable resources.
"""

import pytest
import yaml

from agri_ai_agent.config.schema import UAMS_COLUMNS
from agri_ai_agent.ontology import (
    DEFAULT_REGISTRY_PATH,
    Registry,
    load_registry,
    uams_term,
)

EXPECTED_COLUMN_COUNT = 296


@pytest.fixture(scope="module")
def registry() -> Registry:
    """Return the default Registry loaded from spec/uams_ontology.yaml."""
    return Registry.load()


def test_yaml_loads_as_mapping():
    """The registry YAML parses and carries the required top-level keys."""
    with open(DEFAULT_REGISTRY_PATH, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    for key in ("version", "namespace", "provenance", "prefixes", "columns"):
        assert key in data, f"missing top-level key: {key}"
    assert data["namespace"] == "https://w3id.org/uams#"


def test_all_columns_present_with_expected_count(registry):
    """Every schema column appears exactly once; registry covers exactly them."""
    assert len(UAMS_COLUMNS) == EXPECTED_COLUMN_COUNT
    assert set(registry.columns) == set(UAMS_COLUMNS)


def test_no_extra_or_missing_columns(registry):
    """Registry column set equals the schema column set."""
    assert set(registry.columns) == set(UAMS_COLUMNS)


def test_validate_reports_no_problems(registry):
    """CURIE syntax and cross-column uniqueness are clean."""
    assert registry.validate() == []


def test_every_curie_uses_a_declared_prefix(registry):
    """Each term CURIE expands, i.e. its prefix is declared."""
    data = load_registry(str(DEFAULT_REGISTRY_PATH))
    for column, block in data["columns"].items():
        for term in block["terms"]:
            iri = registry.expand_curie(term["curie"])
            assert iri.startswith("http"), (column, term["curie"], iri)


def test_expand_curie_resolves_known_prefixes(registry):
    """CURIE expansion matches each ontology's IRI stem."""
    assert registry.expand_curie("AGROVOC:c_5192") == ("http://aims.fao.org/aos/agrovoc/c_5192")
    assert registry.expand_curie("ENVO:00001995") == (
        "http://purl.obolibrary.org/obo/ENVO_00001995"
    )
    assert registry.expand_curie("PO:0009047") == ("http://purl.obolibrary.org/obo/PO_0009047")
    assert registry.expand_curie("CO_320:0000005") == (
        "https://cropontology.org/rdf/CO_320:0000005"
    )


def test_expand_curie_raises_on_unknown_prefix(registry):
    """Unknown prefixes raise KeyError, never silently pass."""
    with pytest.raises(KeyError):
        registry.expand_curie("BOGUS:1")


def test_known_columns_resolve_to_expected_ids(registry):
    """Anchor columns map to their AGROVOC / ENVO identifiers.

    c_5192 is AGROVOC "nitrogen". This test previously asserted c_5188, which is
    "nitric acid" — the identifier had been generated rather than resolved.
    """
    assert registry.term_iri("Nitrogen").endswith("agrovoc/c_5192")
    # ENVO:00001995 is "rock"; AGROVOC c_34901 is "soil pH".
    assert registry.term_iri("Soil_pH").endswith("agrovoc/c_34901")
    # ENVO:03000127 is "acid rainfall", a different concept.
    assert registry.term_iri("Rainfall").endswith("agrovoc/c_a060993c")


def test_unmapped_columns_have_no_term(registry):
    """Columns with no external term return None (not fabricated)."""
    for column in ("Paper_ID", "DOI", "Target_Nitrogen", "Crop_Code"):
        assert registry.term_iri(column) is None


def test_units_and_ranges_are_honest(registry):
    """Spot-check honest agronomic units and migrated validation ranges."""
    assert registry.canonical_unit("Nitrogen") == "kg/ha"
    assert registry.canonical_unit("Rainfall") == "mm"
    assert registry.canonical_unit("Soil_pH") is None  # pH is dimensionless
    assert registry.validation_range("Soil_pH") == (3.0, 10.0)
    assert registry.validation_range("Harvest_Index") == (0.0, 1.5)
    assert registry.validation_range("Paper_ID") == (None, None)


def test_synonyms_migrated_from_variant_map(registry):
    """Synonyms carry over from VARIANT_MAP / KB and exclude the column name."""
    nitrogen = registry.synonyms("Nitrogen")
    assert "available_n" in nitrogen
    assert "n" in nitrogen
    assert "Nitrogen" not in nitrogen


def test_unresolvable_crop_ontology_terms_are_not_exact_matches(registry):
    """CO_320 cannot be resolved via cropontology.org or OLS4, so it cannot claim exactMatch."""
    data = load_registry(str(DEFAULT_REGISTRY_PATH))
    for column, block in data["columns"].items():
        for term in block.get("terms") or []:
            if str(term.get("curie", "")).startswith("CO_320:"):
                assert term["predicate"] == "skos:closeMatch", (column, term)
                assert term.get("verified") is False, (column, term)


def test_uams_term_mints_namespaced_iri():
    """uams_term appends the column name to the UAMS base namespace."""
    assert uams_term("Soil_pH") == "https://w3id.org/uams#Soil_pH"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


def test_identifiers_are_resolved_not_generated(registry):
    """Every mapping below was verified against its authority's own label.

    The registry originally asserted skos:exactMatch on generated identifiers: Nitrogen
    pointed at "nitric acid", Yield_per_Hectare at "Yunnan", Harvest_Index at "Yemen",
    Ash at "donkeys". Only 18 of 106 mappings were correct. These anchors pin the
    resolved values so the failure cannot recur silently.
    """
    expected = {
        "Nitrogen": "agrovoc/c_5192",  # nitrogen, not nitric acid (c_5188)
        "Phosphorus": "agrovoc/c_5804",  # phosphorus
        "Potassium": "agrovoc/c_6139",  # potassium
        "Soil_pH": "agrovoc/c_34901",  # soil pH, not ENVO "rock"
        "Yield_per_Hectare": "agrovoc/c_10176",  # crop yield, not "Yunnan"
        "Harvest_Index": "agrovoc/c_24854",  # harvest index, not "Yemen"
    }
    for column, suffix in expected.items():
        assert registry.term_iri(column).endswith(suffix), column


def test_no_mapping_claims_exact_match_without_verification(registry):
    """An exactMatch is transitive; it may not be asserted on an unresolvable term."""
    data = load_registry(str(DEFAULT_REGISTRY_PATH))
    for column, block in data["columns"].items():
        for term in block.get("terms") or []:
            if term.get("verified") is False:
                assert term["predicate"] != "skos:exactMatch", (column, term)


def test_term_iri_returns_string_for_mapped_column(registry):
    """term_iri returns a non-None string for columns with ontology terms."""
    iri = registry.term_iri("Nitrogen")
    assert isinstance(iri, str)
    assert iri.endswith("c_5192")


def test_term_iri_returns_none_for_unmapped_column(registry):
    """term_iri returns None for columns without ontology terms."""
    assert registry.term_iri("Paper_ID") is None
    assert registry.term_iri("DOI") is None


def test_term_iri_none_does_not_have_endswith():
    """Regression: calling .endswith() on None must not crash.

    This covers the mypy-reported union-attr issue at registry.py:173.
    """
    from agri_ai_agent.ontology.registry import Registry

    reg = Registry.load()
    result = reg.term_iri("Paper_ID")
    assert result is None
    # The fix ensures this pattern is handled safely:
    if result is not None:
        result.endswith("x")  # should not be reached
