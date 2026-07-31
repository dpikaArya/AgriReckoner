"""Tests for the deterministic half of open-access corpus extraction.

The model chooses which table and which columns; code reads every number. Only the reading
is tested here, because that is the half that can silently corrupt a value — a mishandled
span shifts a whole row into the wrong column, and a dispersion suffix parsed as part of the
number changes the figure itself.
"""

import xml.etree.ElementTree as ET

import pytest

from scripts.extract_oa_corpus import number_in, parse_table, rows_from, summarise

TABLE = """<table-wrap><label>Table 6</label><caption><p>Grain yield by treatment</p></caption>
<table>
<tr><th>Year</th><th>Treatment</th><th>N applied (kg/ha)</th><th>Grain yield (q ha-1)</th></tr>
<tr><td rowspan="2">2021</td><td>T1</td><td>0</td><td>65.50ab</td></tr>
<tr><td>T2</td><td>120</td><td>68.85a</td></tr>
<tr><td>2022</td><td>T3</td><td>60</td><td>61.70 abc</td></tr>
<tr><td>Mean</td><td></td><td></td><td>65.35</td></tr>
</table></table-wrap>"""

CHOICE = {
    "treatment_column": 1,
    "yield_column": 3,
    "yield_unit": "q ha-1",
    "dose_columns": [2],
}


@pytest.fixture
def grid():
    return parse_table(ET.fromstring(TABLE))[1][1:]  # body rows only


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("65.50ab", 65.50),
        ("68.85a", 68.85),
        ("61.70 abc", 61.70),
        ("12506 ± 456b", 12506.0),
        ("16918 ± 713 a", 16918.0),
        ("1,205", None),
        ("ns", None),
        ("", None),
    ],
)
def test_significance_letters_and_dispersion_are_not_read_as_digits(cell, expected):
    assert number_in(cell) == expected


def test_rows_are_read_from_the_chosen_columns(grid):
    rows = rows_from(grid, CHOICE)
    assert [r["treatment"] for r in rows] == ["T1", "T2", "T3"]
    assert [r["yield_value"] for r in rows] == [65.50, 68.85, 61.70]


def test_the_dose_column_travels_with_the_row(grid):
    rows = rows_from(grid, CHOICE)
    assert [r["doses"][0] for r in rows] == [0.0, 120.0, 60.0]


def test_summary_rows_are_excluded(grid):
    assert all(r["treatment"] != "Mean" for r in rows_from(grid, CHOICE))


def test_the_verbatim_cell_is_kept_for_checking(grid):
    """Every value must be traceable to the text it was read from."""
    assert rows_from(grid, CHOICE)[0]["source_cell"] == "65.50ab"


def test_rowspan_does_not_shift_the_treatment_column(grid):
    """Without span expansion the second row's treatment would read as the dose."""
    assert grid[1][0] == "2021"
    assert grid[1][1] == "T2"


def test_out_of_range_column_choices_are_survived(grid):
    """A model may name a column that does not exist; that must not raise."""
    assert rows_from(grid, {**CHOICE, "yield_column": 99}) == []


def test_the_summary_shown_to_the_model_carries_no_full_data():
    """The model sees enough to choose a table and too little to read one."""
    tables = [parse_table(ET.fromstring(TABLE))]
    text = summarise(tables, max_rows=2)
    assert "Grain yield by treatment" in text
    assert "61.70" not in text  # a later row, beyond the preview


@pytest.mark.parametrize(
    "label",
    ["T1", "T2 (120 kg N/ha)", "CK", "N1 (150 kg-hm-2)", "100% RDF", "Control", "FYM + NPK", "N0"],
)
def test_treatment_labels_are_kept(label):
    from scripts.extract_oa_corpus import is_treatment_label

    assert is_treatment_label(label)


@pytest.mark.parametrize(
    "label",
    [
        "M",
        "N",
        "M x N",
        "M x N x Y",  # ANOVA factor and interaction terms
        "Mean",
        "CD (0.05)",
        "CV (%)",
        "SEm",  # summary rows
        "Treatments",
        "2021",  # header and year rows
        "F-Rep (2,12)",
        "P-Rep",
        "Contrast 4",
        "Main effects",
        "R2",
        "RMSE",
        "Root length (cm)",
        "TKW (g)",
        "NSM (n. m-2)",  # transposed table: variables as rows
    ],
)
def test_statistics_and_variable_labels_are_rejected(label):
    """Agronomy tables append an ANOVA block; reading it gives F values shaped like yields."""
    from scripts.extract_oa_corpus import is_treatment_label

    assert not is_treatment_label(label)


def test_a_transposed_table_is_rejected_whole():
    """If most row labels are variables, the column is not a treatment column."""
    from scripts.extract_oa_corpus import rows_from

    grid = [
        ["Root length (cm)", "175.0"],
        ["TKW (g)", "49.7"],
        ["NSM (n. m-2)", "176.0"],
        ["T1", "5.2"],
    ]
    choice = {"treatment_column": 0, "yield_column": 1, "yield_unit": "t/ha", "dose_columns": []}
    assert rows_from(grid, choice) == []


def test_a_mostly_clean_table_survives_a_stray_summary_row():
    from scripts.extract_oa_corpus import rows_from

    grid = [["T1", "5.2"], ["T2", "6.1"], ["T3", "5.8"], ["Mean", "5.7"]]
    choice = {"treatment_column": 0, "yield_column": 1, "yield_unit": "t/ha", "dose_columns": []}
    assert [r["treatment"] for r in rows_from(grid, choice)] == ["T1", "T2", "T3"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
