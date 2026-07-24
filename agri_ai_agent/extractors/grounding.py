"""Grounding: verify an LLM-proposed value against its cited source and the registry.

The LLM is untrusted. A value is accepted only if its number appears literally in the
source quote (a numeric-echo check — regex now checks the LLM), and it falls within the
registry's validated range for that column. Failures are flagged with a reason, never
silently dropped, so a human can review them.
"""

import re

from agri_ai_agent.extractors.base import ExtractedField
from agri_ai_agent.extractors.fields import EXTRACTION_FIELDS

_FIELD_BY_COLUMN = {f.column: f for f in EXTRACTION_FIELDS}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _quote_contains_value(value: float, quote: str) -> bool:
    """True if ``value`` appears as a number in ``quote`` (within rounding tolerance)."""
    for token in _NUMBER.findall(quote.replace(",", "")):
        try:
            if abs(float(token) - value) <= max(0.01, abs(value) * 1e-3):
                return True
        except ValueError:
            continue
    return False


def ground_field(item: ExtractedField) -> ExtractedField:
    """Return ``item`` with its status set to reported/rejected after grounding checks."""
    if item.value is None:
        item.status = "rejected"
        item.reject_reason = "no value"
        return item

    spec = _FIELD_BY_COLUMN.get(item.column)
    if spec is None:
        item.status = "rejected"
        item.reject_reason = "unknown column"
        return item

    if not item.source_quote or not _quote_contains_value(item.value, item.source_quote):
        item.status = "rejected"
        item.reject_reason = "value not found in cited source quote"
        return item

    if spec.min_value is not None and item.value < spec.min_value:
        item.status = "rejected"
        item.reject_reason = f"below range min {spec.min_value}"
        return item
    if spec.max_value is not None and item.value > spec.max_value:
        item.status = "rejected"
        item.reject_reason = f"above range max {spec.max_value}"
        return item

    item.status = "reported"
    return item


def ground_all(items: list[ExtractedField]) -> list[ExtractedField]:
    """Ground every field; keep rejected ones (flagged) for the review queue."""
    return [ground_field(i) for i in items]


if __name__ == "__main__":
    ok = ground_field(ExtractedField("Soil_pH", 6.8, None, "The soil pH was 6.8 in loam plots."))
    assert ok.status == "reported", ok
    bad_echo = ground_field(ExtractedField("Soil_pH", 9.9, None, "The soil pH was 6.8."))
    assert bad_echo.status == "rejected" and "not found" in bad_echo.reject_reason
    out_of_range = ground_field(ExtractedField("Soil_pH", 42.0, None, "pH 42.0 (typo)."))
    assert out_of_range.status == "rejected" and "range" in out_of_range.reject_reason
    print("grounding smoke OK ->", ok.status, "|", bad_echo.reject_reason, "|", out_of_range.reject_reason)
