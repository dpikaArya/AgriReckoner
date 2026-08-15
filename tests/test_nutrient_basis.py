"""Nutrients are stored as the element, and an undeclared basis is refused, not guessed.

Two-sided by construction: every test that asserts a conversion fires has a partner that
asserts it stays clean on an elemental or non-nutrient label. A converter that fired on
everything would pass the first half and fail the second.
"""

import pandas as pd
import pytest
import yaml

from agri_ai_agent.config.schema import VARIANT_MAP
from agri_ai_agent.external_data.column_mapper import map_dataframe, to_elemental_basis
from agri_ai_agent.extractors.nutrients import (
    CONVERT,
    ELEMENTAL,
    KEEP,
    OXIDE,
    AMBIGUOUS,
    REFUSE,
    UNKNOWN,
    basis_sensitive_columns,
    detect_basis,
    oxide_aliases,
    oxide_factor,
    plan_conversions,
    to_elemental,
)

ONTOLOGY = "spec/uams_ontology.yaml"

# Published mass fractions of the element in each oxide.
EXPECTED_FACTORS = {
    "P2O5": 0.4364,
    "K2O": 0.8302,
    "CaO": 0.7147,
    "MgO": 0.6030,
    "SO3": 0.4005,
    "Na2O": 0.7419,
}


@pytest.mark.parametrize("formula,expected", sorted(EXPECTED_FACTORS.items()))
def test_conversion_factors_match_stoichiometry(formula, expected):
    assert oxide_factor(formula) == pytest.approx(expected, abs=5e-5)


def test_phosphorus_factor_matches_the_figure_the_issue_quotes():
    """P = P2O5 x 0.4364, i.e. an oxide figure is 2.29x the elemental one."""
    assert 1 / oxide_factor("P2O5") == pytest.approx(2.2913, abs=1e-3)
    assert 1 / oxide_factor("K2O") == pytest.approx(1.2046, abs=1e-3)


# --- detection: fires on every spelling an author might use --------------------------- #


@pytest.mark.parametrize(
    "label",
    [
        "P2O5",
        "p2o5_kg_ha",
        "P₂O₅",
        "P₂O₅ (kg ha⁻¹)",
        "kg P2O5/ha",
        "P 2 O 5",
        "P2O5 rate",
        "Applied P2O5 (kg ha-1)",
    ],
)
def test_oxide_spellings_are_detected(label):
    basis, formula = detect_basis(label)
    assert (basis, formula) == (OXIDE, "P2O5"), label


@pytest.mark.parametrize(
    "label,formula",
    [
        ("K2O_kg_ha", "K2O"),
        ("K₂O", "K2O"),
        ("CaO (mg/kg)", "CaO"),
        ("MgO", "MgO"),
        ("SO3 kg/ha", "SO3"),
        ("Na2O", "Na2O"),
    ],
)
def test_every_oxide_in_the_table_is_detected(label, formula):
    assert detect_basis(label) == (OXIDE, formula)


# --- detection: stays clean on labels that are not oxides ----------------------------- #


@pytest.mark.parametrize(
    "label",
    [
        "available_p",
        "fertiliser_p_kg_p_ha",
        "soil_p_olsen_mg_kg",
        "soil_p_bray_1_mg_p_kg",
        "phosphorus",
        "Phosphorus",
        "available_k",
        "soil_k_exchangeable_mg_k_kg",
        "potassium",
    ],
)
def test_elemental_labels_are_not_converted(label):
    basis, _ = detect_basis(label)
    assert basis == ELEMENTAL, label
    value, ok, _ = to_elemental(100.0, label)
    assert ok and value == 100.0


@pytest.mark.parametrize(
    "label",
    ["cacao_yield", "Yield_per_Hectare", "Rainfall", "Phosphatase", "Spike_Length", "Season"],
)
def test_unrelated_columns_are_never_read_as_oxides(label):
    """The formula matcher must not fire inside an unrelated word ('cacao' contains 'cao')."""
    basis, _ = detect_basis(label)
    assert basis != OXIDE, label


@pytest.mark.parametrize("label", ["phosphate", "potash", "muriate of potash", "DAP", "SSP"])
def test_undeclared_basis_is_refused_not_guessed(label):
    assert detect_basis(label)[0] == AMBIGUOUS, label
    value, ok, reason = to_elemental(100.0, label)
    assert not ok, reason
    assert value == 100.0, "a refused value must come back untouched, not silently scaled"


# --- the plan the pandas callers apply ------------------------------------------------ #


def test_plan_marks_each_label_with_what_the_caller_must_do():
    plan = plan_conversions(
        {
            "P2O5_kg_ha": "Phosphorus",
            "available_p": "Phosphorus",
            "potash": "Potassium",
            "Rainfall_mm": "Rainfall",
        }
    )
    assert plan["P2O5_kg_ha"][0] == CONVERT
    assert plan["available_p"][0] == KEEP
    assert plan["potash"][0] == REFUSE
    assert plan["Rainfall_mm"][0] == KEEP


def test_an_oxide_label_outside_a_nutrient_column_is_left_alone():
    """Only columns whose meaning depends on the basis are judged."""
    plan = plan_conversions({"P2O5_kg_ha": "Notes"})
    assert plan["P2O5_kg_ha"][0] == KEEP


def test_nitrogen_is_never_treated_as_basis_sensitive():
    assert "Nitrogen" not in basis_sensitive_columns()
    plan = plan_conversions({"Available_N": "Nitrogen", "nitrogen_kg_ha": "Nitrogen"})
    assert all(action == KEEP for action, _, _ in plan.values())


# --- end to end through the shared mapper --------------------------------------------- #


def test_mapper_converts_oxides_and_leaves_elemental_alone():
    df = pd.DataFrame({"P2O5_kg_ha": [60.0], "K2O_kg_ha": [40.0], "Available_N": [120.0]})
    out = map_dataframe("X", df)
    assert out["Phosphorus"].iloc[0] == pytest.approx(26.1856, abs=1e-3)
    assert out["Potassium"].iloc[0] == pytest.approx(33.2061, abs=1e-3)
    assert out["Nitrogen"].iloc[0] == 120.0


def test_mapper_refuses_to_place_an_undeclared_basis_in_a_nutrient_column():
    renamed = {"potash": "Potassium"}
    to_elemental_basis(pd.DataFrame({"potash": [50.0]}), renamed)
    assert "potash" not in renamed, "an undeclared basis must lose its UAMS mapping"


def test_conversion_is_applied_once_not_twice():
    df = pd.DataFrame({"P2O5_kg_ha": [100.0]})
    once = map_dataframe("X", df)["Phosphorus"].iloc[0]
    assert once == pytest.approx(43.642, abs=1e-2)


# --- every path that renames into a UAMS column must convert -------------------------- #


def test_extraction_agent_converts_on_the_way_into_the_schema():
    from agri_ai_agent.agents.extraction_agent import ExtractionAgent

    out = ExtractionAgent()._map_to_schema(pd.DataFrame({"P2O5_kg_ha": [60.0]}))
    assert out["Phosphorus"].iloc[0] == pytest.approx(26.1856, abs=1e-3)


def test_universal_schema_generator_converts_on_the_way_into_the_schema():
    from universal_schema_generator import map_columns_to_uams

    out = map_columns_to_uams(pd.DataFrame({"P2O5_kg_ha": [60.0]}))
    assert out["Phosphorus"].iloc[0] == pytest.approx(26.1856, abs=1e-3)


def test_prose_extraction_tells_an_oxide_figure_from_an_elemental_one():
    from run_aaf_pipeline import _elemental_values

    pattern = (
        r"(available\s*phosphorus|phosphorus|phosphorous|p\s*2\s*o\s*5|p₂o₅)"
        r"\s*[:=]?\s*((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?"
    )
    found = _elemental_values(pattern, "available phosphorus 24.5 kg/ha and P2O5 60 kg/ha")
    assert found == pytest.approx([24.5, 26.1856], abs=1e-3)


# --- drift: the spec and the code must agree ------------------------------------------ #


def _ontology():
    with open(ONTOLOGY) as handle:
        return yaml.safe_load(handle)["columns"]


def test_every_oxide_synonym_in_the_spec_is_recognised_by_the_code():
    """A term the spec calls an oxide must convert, or the declaration is decorative."""
    for column, block in _ontology().items():
        for synonym in block.get("oxide_synonyms") or []:
            basis, formula = detect_basis(synonym)
            assert basis == OXIDE, (column, synonym)
            assert formula in oxide_aliases()


def test_every_ambiguous_synonym_in_the_spec_is_refused_by_the_code():
    for column, block in _ontology().items():
        for synonym in block.get("ambiguous_synonyms") or []:
            assert detect_basis(synonym)[0] == AMBIGUOUS, (column, synonym)


def test_spec_declares_a_basis_for_every_basis_sensitive_column():
    ontology = _ontology()
    for column in basis_sensitive_columns():
        assert ontology[column].get("basis") == "elemental", column


def test_no_oxide_term_is_still_listed_as_a_plain_synonym():
    """A plain synonym means 'the same quantity'; an oxide is a different quantity."""
    for column, block in _ontology().items():
        if column not in basis_sensitive_columns():
            continue
        for synonym in block.get("synonyms") or []:
            assert detect_basis(synonym)[0] != OXIDE, (column, synonym)


def test_master_column_map_oxide_keys_are_all_known_to_the_converter():
    """The legacy runner has its own column map; it must not smuggle in an unknown basis."""
    from run_aaf_pipeline import MASTER_COLUMN_MAP

    sensitive = basis_sensitive_columns()
    aliases = {a: c for a, c in MASTER_COLUMN_MAP.items() if c in sensitive}
    for alias, (action, _, reason) in plan_conversions(aliases).items():
        assert action != REFUSE, f"{alias!r} would be dropped from the pipeline: {reason}"


def test_variant_map_oxide_keys_are_all_known_to_the_converter():
    """A new oxide alias in VARIANT_MAP must be one the converter can handle."""
    sensitive = basis_sensitive_columns()
    aliases = {a: c for a, c in VARIANT_MAP.items() if c in sensitive}
    for alias, (action, _, reason) in plan_conversions(aliases).items():
        assert action != REFUSE, (
            f"VARIANT_MAP maps {alias!r} into a nutrient column, but the pipeline would "
            f"drop it: {reason}"
        )


# --- regressions found by adversarial review of the first cut of this change ---


@pytest.mark.parametrize(
    "label,formula",
    [
        ("potassium oxide", "K2O"),
        ("phosphorus pentoxide", "P2O5"),
        ("calcium oxide", "CaO"),
        ("magnesium oxide", "MgO"),
        ("sulphur trioxide", "SO3"),
        ("sodium oxide", "Na2O"),
        ("Potassium (oxide basis)", "K2O"),
        ("Phosphorus, oxide basis", "P2O5"),
    ],
)
def test_an_oxide_spelled_in_words_is_still_an_oxide(label, formula):
    """Papers write "potassium oxide" as often as K2O; reading it as K loses 17 percent."""
    assert detect_basis(label) == (OXIDE, formula), label


@pytest.mark.parametrize("label", ["Polyphenol_Oxidase", "oxidation_rate", "nitrogen oxide"])
def test_the_oxide_word_does_not_fire_without_a_nutrient_element(label):
    """'oxidase' is not 'oxide', and NOx is not a fertiliser basis."""
    assert detect_basis(label)[0] != OXIDE, label


@pytest.mark.parametrize("label,formula", [("P²O⁵", "P2O5"), ("K²O", "K2O")])
def test_superscript_formulas_are_read_like_subscript_ones(label, formula):
    """Typesetting varies; P²O⁵ is the same compound as P₂O₅ and must convert."""
    assert detect_basis(label) == (OXIDE, formula), label


def test_a_mass_unit_is_not_mistaken_for_its_element_symbol():
    """The 'mg' in 'ap (mg/kg)' is a milligram, not magnesium."""
    assert detect_basis("ap (mg/kg)")[0] != ELEMENTAL
    assert detect_basis("magnesium (mg/kg)") == (ELEMENTAL, "Mg")


def test_a_label_that_says_nothing_about_basis_is_kept_not_dropped():
    """Only an actively ambiguous form is refused; silence means the schema basis."""
    plan = plan_conversions({"ap (mg/kg)": "Phosphorus", "potash": "Potassium"})
    assert plan["ap (mg/kg)"][0] == KEEP
    assert plan["potash"][0] == REFUSE


def test_values_that_are_not_numbers_survive_the_conversion():
    """Coercing an unreadable cell to NaN would delete data the elemental path keeps."""
    frame = pd.DataFrame({"p2o5_kg_ha": ["60,5", "60 kg", "1,234"]})
    assert map_dataframe("X", frame)["Phosphorus"].tolist() == ["60,5", "60 kg", "1,234"]


def test_duplicate_source_labels_do_not_crash_the_converter():
    """Two columns with the same header used to raise TypeError inside the conversion."""
    frame = pd.DataFrame([[60.0, 40.0]], columns=["P2O5_kg_ha", "P2O5_kg_ha"])
    converted = to_elemental_basis(frame, {"P2O5_kg_ha": "Phosphorus"})
    assert converted.iloc[0, 0] == pytest.approx(26.1856, rel=1e-4)
    assert converted.iloc[0, 1] == pytest.approx(17.4571, rel=1e-4)


def test_the_basis_is_read_from_the_unit_when_the_term_names_the_element():
    """"phosphorus 60 kg P2O5 ha-1" states the element but means the oxide."""
    from run_aaf_pipeline import _elemental_values

    pattern = (
        r"(available\s*phosphorus|phosphorus|phosphorous|p\s*2\s*o\s*5)\s*[:=]?\s*"
        r"((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?"
    )
    assert _elemental_values(pattern, "phosphorus 60 kg P2O5 ha-1") == [pytest.approx(26.1856, rel=1e-4)]
    assert _elemental_values(pattern, "available phosphorus 24.5 kg/ha") == [24.5]


def test_a_later_clause_does_not_change_an_earlier_value():
    """The unit window must stop at the clause break, or K2O would rescale phosphorus."""
    from run_aaf_pipeline import _elemental_values

    pattern = (
        r"(available\s*phosphorus|phosphorus|phosphorous|p\s*2\s*o\s*5)\s*[:=]?\s*"
        r"((?:\d+\.?\d*|\.\d+))\s*(?:kg/ha|kg|ppm|mg)?"
    )
    assert _elemental_values(pattern, "phosphorus 60 kg/ha and K2O 40 kg/ha") == [60.0]
