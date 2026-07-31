"""Yield parsed from prose must carry its unit into kg/ha.

Regression: the unit sat in a non-capturing group and was discarded, so "2.5 t ha-1"
was stored as 2.5 kg/ha — a 1000x error in the target variable — and unitless numbers
were accepted as yields.
"""

import pytest

from enhanced_extraction import yield_regex_extraction

KG_PER_HA = "Yield_per_Hectare"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Grain yield was 2.5 t ha-1 in the treated plots.", 2500.0),
        ("grain yield was 3.1 t/ha", 3100.0),
        ("The maximum grain yield of 35 q/ha was recorded.", 3500.0),
        ("Seed yield: 4200 kg/ha under N120.", 4200.0),
        ("Biological yield = 9.4 t ha-1", 9400.0),
    ],
)
def test_units_are_converted_to_kg_per_hectare(text, expected):
    assert yield_regex_extraction(text)[KG_PER_HA] == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "Yield ranged from 12 to 18 across plots.",
        "Yield increased by 23 percent over the control.",
        "Plant height was 45 cm.",
        "",
    ],
)
def test_a_number_without_a_unit_is_not_a_yield(text):
    """Guessing the unit is a 100-1000x error, so no value is better than a wrong one."""
    assert yield_regex_extraction(text) == {}


def test_tonnes_and_kilograms_do_not_collapse_to_the_same_number():
    tonnes = yield_regex_extraction("grain yield was 2.5 t/ha")[KG_PER_HA]
    kilos = yield_regex_extraction("grain yield was 2.5 kg/ha")[KG_PER_HA]
    assert tonnes == 1000 * kilos


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
