"""Tests for reading experimental results out of a paper's tables.

These cover `agri_ai_agent.extractors.tables`, which both the pipeline's
`TableIntelligenceAgent` and the open-access harvester use, so a rule proven here holds
everywhere it is applied.

Each case corresponds to a failure measured by auditing extraction against source papers.
They matter because every one produced a confident wrong number rather than a missing one.
"""

import xml.etree.ElementTree as ET

import pytest

from agri_ai_agent.extractors.tables import (
    caption_is_an_analysis,
    dose_in_label,
    find_treatment_column,
    find_yield_column,
    is_treatment_label,
    parse_grid,
    parse_measurement,
    read_rows,
    selection_problem,
    valid_dose_columns,
)

TABLE = """<table-wrap><label>Table 6</label><caption><p>Grain yield by treatment</p></caption>
<table>
<tr><th>Year</th><th>Treatment</th><th>N applied (kg/ha)</th><th>Grain yield (kg ha-1)</th></tr>
<tr><td rowspan="2">2021</td><td>T1</td><td>0</td><td>1 446ab</td></tr>
<tr><td>T2</td><td>120</td><td>2,884a</td></tr>
<tr><td>2022</td><td>T3</td><td>60</td><td>6173.40 b</td></tr>
<tr><td>Mean</td><td></td><td></td><td>3501.13</td></tr>
</table></table-wrap>"""


@pytest.fixture
def table():
    caption, grid = parse_grid(ET.fromstring(TABLE))
    return caption, grid[:1], grid[1:]


def test_rowspan_is_expanded(table):
    """An unexpanded span shifts every later cell one column left."""
    _, _, body = table
    assert body[0][0] == "2021"
    assert body[1][0] == "2021"
    assert body[1][1] == "T2"


def test_the_treatment_column_is_found_not_assumed(table):
    """Column 0 is the year here; assuming position replaces the variable with a date."""
    _, header, body = table
    assert find_treatment_column(header, body) == 1


def test_the_yield_column_is_found_by_its_header(table):
    caption, header, _ = table
    assert find_yield_column(header, caption) == 3


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("1 446ab", 1446.0),
        ("2,884a", 2884.0),
        ("6173.40 b", 6173.40),
        ("12506 ± 456b", 12506.0),
        ("65.50ab", 65.50),
        ("ns", None),
        ("", None),
    ],
)
def test_measurements_survive_separators_dispersion_and_letters(cell, expected):
    """Rejecting "1 446" deleted the highest-yielding rows and biased the corpus down."""
    assert parse_measurement(cell) == expected


def test_rows_are_read_from_the_located_columns(table):
    _, header, body = table
    rows = read_rows(body, 1, 3, valid_dose_columns([2], header, 3), "kg ha-1")
    assert [r["treatment"] for r in rows] == ["T1", "T2", "T3"]
    assert [r["yield_value"] for r in rows] == [1446.0, 2884.0, 6173.40]
    assert [r["doses"][0] for r in rows] == [0.0, 120.0, 60.0]
    assert rows[0]["source_cell"] == "1 446ab"


def test_the_summary_row_is_excluded(table):
    _, _, body = table
    assert all(r["treatment"] != "Mean" for r in read_rows(body, 1, 3))


@pytest.mark.parametrize(
    "label", ["T1", "T2 (120 kg N/ha)", "CK", "100% RDF", "Control", "FYM + NPK", "N0"]
)
def test_treatment_labels_are_kept(label):
    assert is_treatment_label(label)


@pytest.mark.parametrize(
    "label",
    [
        "M",
        "N",
        "M x N",
        "M x N x Y",
        "Mean",
        "CD (0.05)",
        "Treatments",
        "2021",
        "F-Rep (2,12)",
        "P-Rep",
        "Contrast 4",
        "Main effects",
        "R2",
        "RMSE",
        "Root length (cm)",
        "TKW (g)",
        "NSM (n. m-2)",
    ],
)
def test_statistics_and_variable_labels_are_rejected(label):
    assert not is_treatment_label(label)


def test_an_ambiguous_code_is_a_treatment_in_the_body_and_a_statistic_at_the_foot():
    """SD meant straw deep incorporation, and was the highest-yielding treatment."""
    assert is_treatment_label("SD", near_foot=False)
    assert not is_treatment_label("SD", near_foot=True)


def test_the_analysis_block_is_detected_by_its_run_not_by_position():
    body = [["T1", "5.2"], ["T2", "6.1"], ["T3", "5.9"], ["SD", "0.31"], ["CV", "4.2"]]
    assert [r["treatment"] for r in read_rows(body, 0, 1)] == ["T1", "T2", "T3"]


def test_a_treatment_named_sd_survives_in_the_body_of_the_table():
    body = [["T1", "5.2"], ["SD", "6173.4"], ["T3", "5.9"], ["Mean", "5.7"]]
    assert [r["treatment"] for r in read_rows(body, 0, 1)] == ["T1", "SD", "T3"]


SOIL_HEADER = [["Treatments", "pH", "AP (mg kg-1)", "AK", "MBC", "GY (kg ha-1)"]]


def test_a_column_that_is_not_a_yield_is_refused():
    """The largest measured error class: soil available phosphorus read as grain yield."""
    problem = selection_problem(0, 2, SOIL_HEADER, "Soil properties and grain yield")
    assert problem and "does not name a yield" in problem


def test_the_real_yield_column_in_the_same_table_is_accepted():
    assert selection_problem(0, 5, SOIL_HEADER, "Soil properties and grain yield") is None


@pytest.mark.parametrize("columns", [(-1, 1), (0, -1)])
def test_a_column_the_caller_could_not_find_is_refused(columns):
    header = [["Treatment", "Grain yield (kg ha-1)"]]
    assert selection_problem(*columns, header, "Grain yield") == "no usable column"


@pytest.mark.parametrize(
    "caption",
    [
        "Phytate correlations with cereal grain parameters",
        "Analysis of variance: treatment effects",
        "AquaCrop model evaluation (RMSE)",
        "Simulated yields under SSP4.5 for the 2050s",
        "Entropy weight membership values",
    ],
)
def test_captions_about_an_analysis_are_refused(caption):
    assert caption_is_an_analysis(caption)


@pytest.mark.parametrize(
    "caption",
    [
        "Effect of aerated irrigation with nitrogen fertiliser on maize yield and fitting curve",
        "Grain yield by treatment",
        "Influence of nutrient recommendations on yield and benefit",
    ],
)
def test_an_experiment_that_also_reports_a_fitted_curve_is_still_an_experiment(caption):
    """Over-rejecting cost a paper with 12 genuine treatment rows."""
    assert not caption_is_an_analysis(caption)


def test_a_dose_column_that_is_not_a_rate_is_dropped():
    """Costs, grain weights and the yield column itself were all emitted as doses."""
    header = [["Treatment", "Fertilizer cost ($ ha-1)", "Grain yield (q ha-1)"]]
    assert valid_dose_columns([1, 2], header, 2) == []


def test_a_transposed_table_is_rejected_whole():
    """If most row labels are variables, the column is not a treatment column."""
    body = [
        ["Root length (cm)", "175.0"],
        ["TKW (g)", "49.7"],
        ["NSM (n. m-2)", "176.0"],
        ["T1", "5.2"],
    ]
    assert read_rows(body, 0, 1) == []


def test_an_out_of_range_column_does_not_raise(table):
    _, _, body = table
    assert read_rows(body, 1, 99) == []


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("N2 (300 kg·hm-2)", 300.0),
        ("N1 (150 kg ha-1)", 150.0),
        ("T5 (2.5 t/ha)", 2.5),
        ("T1", None),
        ("CK", None),
        ("100% RDF", None),
    ],
)
def test_a_rate_written_into_the_treatment_label_is_recovered(label, expected):
    """The commonest place a dose appears; needs no dose column and no methods legend."""
    assert dose_in_label(label) == expected


def test_a_validated_dose_column_takes_precedence_over_the_label():
    header = [["Treatment", "N applied (kg ha-1)", "Grain yield (kg ha-1)"]]
    body = [["N1 (150 kg ha-1)", "160", "5100"]]
    rows = read_rows(body, 0, 2, valid_dose_columns([1], header, 2))
    assert rows[0]["doses"] == [160.0]


def test_the_label_is_used_when_no_dose_column_survives_validation():
    body = [["N1 (150 kg ha-1)", "5100"], ["N2 (300 kg ha-1)", "5800"]]
    assert [r["doses"] for r in read_rows(body, 0, 1)] == [[150.0], [300.0]]


@pytest.mark.parametrize(
    "label", ["LSD", "SEm±", "SE ±", "SEM ±", "CD", "CD (0.05)", "NS", "S.E.m", "SD ±"]
)
def test_dispersion_labels_are_never_treatments(label):
    """LSD and SEm are never treatment names; a trailing plus-minus settles the rest."""
    assert not is_treatment_label(label, near_foot=False)


@pytest.mark.parametrize("label", ["SD", "SDR", "Seaweed", "CV1"])
def test_codes_that_merely_start_like_a_statistic_survive(label):
    """SD was a real treatment (straw deep incorporation) and the highest-yielding one."""
    assert is_treatment_label(label, near_foot=False)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
