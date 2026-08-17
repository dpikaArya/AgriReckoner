"""The UAMS fields an LLM is asked to extract, derived from the ontology registry.

Only columns that carry a validated numeric range in the registry are targeted — those
are the measurable agronomic variables. Tying this list to the registry keeps extraction,
units, and validation in lockstep with the single source of ontology truth.
"""

from dataclasses import dataclass

from agri_ai_agent.ontology.registry import load_registry


@dataclass(frozen=True)
class ExtractionField:
    column: str
    canonical_unit: str | None
    min_value: float | None
    max_value: float | None
    synonyms: tuple


def _build_fields() -> list[ExtractionField]:
    registry = load_registry()
    fields = []
    for column, entry in registry["columns"].items():
        validation = entry.get("validation") or {}
        vmin, vmax = validation.get("min"), validation.get("max")
        if vmin is None and vmax is None:
            continue
        fields.append(
            ExtractionField(
                column=column,
                canonical_unit=entry.get("unit_ucum"),
                min_value=vmin,
                max_value=vmax,
                synonyms=tuple(entry.get("synonyms") or []),
            )
        )
    return fields


EXTRACTION_FIELDS: list[ExtractionField] = _build_fields()
EXTRACTION_COLUMNS: list[str] = [f.column for f in EXTRACTION_FIELDS]


if __name__ == "__main__":
    assert EXTRACTION_FIELDS, "no extraction fields derived from registry"
    assert "Soil_pH" in EXTRACTION_COLUMNS, EXTRACTION_COLUMNS
    print(f"fields smoke OK -> {len(EXTRACTION_FIELDS)} fields:", EXTRACTION_COLUMNS[:8], "...")
