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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
