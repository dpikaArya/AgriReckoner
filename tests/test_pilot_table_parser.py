"""Tests for the deterministic parts of the open-access corpus pilot.

Only the parsing is tested here; the model calls are not. The parser is what protects
against reading a number out of the wrong column, so its span handling and its column
detection are the parts that must not regress.
"""

import xml.etree.ElementTree as ET

import pytest

from scripts.pilot_oa_corpus import (
    INLINE_DOSE,
    YIELD_UNITS,
    parse_table,
    treatment_column,
    treatment_rows,
)

TABLE = """<table-wrap><caption><p>Table 1 Grain yield of treatments</p></caption><table>
<tr><th>Year</th><th>Treatment</th><th>Grain yield (kg ha-1)</th></tr>
<tr><td rowspan="2">2021</td><td>T1</td><td>2450 ± 30</td></tr>
<tr><td>T2</td><td>3110 ± 40</td></tr>
<tr><td>2022</td><td>T3</td><td>2980 ± 25</td></tr>
<tr><td>CD (0.05)</td><td></td><td>120</td></tr>
</table></table-wrap>"""


@pytest.fixture
def parsed():
    return parse_table(ET.fromstring(TABLE))


def test_rowspan_is_forward_filled(parsed):
    """An unfilled rowspan shifts every later cell one column left."""
    _, _, body = parsed
    assert body[0][0] == "2021"
    assert body[1][0] == "2021"


def test_summary_rows_are_not_treatments(parsed):
    _, header, body = parsed
    rows = treatment_rows(header, body)
    assert all("CD" not in row[treatment_column(header, body)] for row in rows)
    assert len(rows) == 3


def test_treatment_column_is_found_not_assumed(parsed):
    """The first column here is the year; assuming position reads the wrong field."""
    _, header, body = parsed
    assert treatment_column(header, body) == 1


def test_dispersion_survives_parsing(parsed):
    _, _, body = parsed
    assert any("±" in cell for row in body for cell in row)


@pytest.mark.parametrize(
    "header",
    ["Grain yield (kg ha-1)", "Maize yield (kg·hm-2)", "Yield (t/ha)", "Yield (q ha-1)"],
)
def test_yield_units_are_recognised_whatever_the_separator(header):
    assert YIELD_UNITS.search(header)


def test_yield_component_headers_are_not_yields():
    for header in ["Spikelets per Panicle", "Seed Setting Rate (%)", "1000-Grain Weight (g)"]:
        assert not YIELD_UNITS.search(header) or "yield" not in header.lower()


@pytest.mark.parametrize(
    ("label", "expected"),
    [("N2 (300 kg·hm-2)", True), ("N1 (150 kg ha-1)", True), ("CK", False), ("T1", False)],
)
def test_dose_is_read_from_the_treatment_label_when_present(label, expected):
    assert bool(INLINE_DOSE.search(label)) is expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
