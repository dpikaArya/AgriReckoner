"""Tests for unit normalization/conversion and unit-aware grounding."""

from agri_ai_agent.extractors.base import ExtractedField
from agri_ai_agent.extractors.grounding import ground_field
from agri_ai_agent.extractors.units import convert, normalize_unit


def test_normalize_unit_forms():
    assert normalize_unit("t ha-1") == "t/ha"
    assert normalize_unit("g kg-1") == "g/kg"
    assert normalize_unit("mgkg-1") == "mg/kg"
    assert normalize_unit("dS m-1") == "ds/m"
    assert normalize_unit("pH") is None


def test_convert_compatible():
    v, ok, _ = convert(4.2, "t/ha", "kg/ha")
    assert ok and abs(v - 4200) < 1e-6
    v, ok, _ = convert(13.4, "g/kg", "%")
    assert ok and abs(v - 1.34) < 1e-6
    v, ok, _ = convert(95.0, "f", "cel")
    assert ok and abs(v - 35.0) < 1e-6


def test_convert_rejects_incompatible_dimensions():
    _, ok, reason = convert(1.2, "g/kg", "kg/ha")
    assert not ok and "incompatible" in reason


def test_grounding_converts_yield_tha_to_kgha():
    item = ground_field(ExtractedField("Yield_per_Hectare", 4.2, "t ha-1", "yield 4.2 t ha-1"))
    assert item.status == "reported"
    assert abs(item.value_canonical - 4200) < 1e-6


def test_grounding_accepts_organic_carbon_gkg_as_percent():
    """Regression: 13.4 g/kg OC = 1.34%, must no longer be rejected as > 10% max."""
    item = ground_field(ExtractedField("Organic_Carbon", 13.4, "g kg-1", "OC 13.4 g kg-1"))
    assert item.status == "reported"
    assert abs(item.value_canonical - 1.34) < 1e-6


def test_grounding_rejects_soil_concentration_for_rate_column():
    item = ground_field(ExtractedField("Nitrogen", 1.2, "g kg-1", "N 1.2 g kg-1"))
    assert item.status == "rejected"
    assert "incompatible" in item.reject_reason


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
