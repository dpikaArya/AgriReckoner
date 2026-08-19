"""Loader and validator for the UAMS ontology registry (spec/uams_ontology.yaml).

The YAML is the single source of ontology truth: which external ontology terms
(AGROVOC, ENVO, PO, Crop Ontology, FoodOn) each UAMS column maps to, its honest
agronomic unit, its synonyms, and its numeric validation range.

`Registry` wraps that data with lookups and a `validate()` self-check that
verifies CURIE hygiene (declared prefixes, per-prefix local-id syntax) and that
no external CURIE is shared across two different columns.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "spec" / "uams_ontology.yaml"

# Per-prefix local-id syntax. Kept simple and honest: enough to catch typos,
# not a full ontology validator.
_LOCALID_PATTERNS = {
    # AGROVOC issues both legacy numeric ids (c_5192) and newer hex ones (c_7db831f9).
    "AGROVOC": re.compile(r"^c_(\d+|[0-9a-f]{6,})$"),
    "ENVO": re.compile(r"^\d{8}$"),
    "PO": re.compile(r"^\d{7}$"),
    "CO_320": re.compile(r"^\d{7}$"),
    "FOODON": re.compile(r"^\d{8}$"),
    "UO": re.compile(r"^\d{7}$"),
    "ECO": re.compile(r"^\d{7}$"),
    "PATO": re.compile(r"^\d{7}$"),
    "QUDT": re.compile(r"^\S+$"),
    "UCUM": re.compile(r"^\S+$"),
}

_CURIE_SEP = ":"


@lru_cache(maxsize=4)
def load_registry(path: str = str(DEFAULT_REGISTRY_PATH)) -> dict:
    """Load and cache the raw registry mapping from a YAML path.

    Args:
        path: Filesystem path to the ontology YAML.

    Returns:
        The parsed mapping (version, namespace, provenance, prefixes, columns).
    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if "columns" not in data or "prefixes" not in data:
        raise ValueError(f"Registry at {path} missing 'columns' or 'prefixes'")
    return data


def _split_curie(curie: str) -> tuple[str, str]:
    """Split a CURIE into (prefix, localid), respecting CO_320's colon in id."""
    prefix, _, localid = curie.partition(_CURIE_SEP)
    return prefix, localid


class Registry:
    """Typed accessor over the UAMS ontology YAML."""

    def __init__(self, data: dict):
        """Wrap an already-loaded registry mapping."""
        self._data = data
        self._columns = data["columns"]
        self._prefixes = data["prefixes"]

    @classmethod
    def load(cls, path: str = str(DEFAULT_REGISTRY_PATH)) -> "Registry":
        """Construct a Registry from a YAML path."""
        return cls(load_registry(path))

    @property
    def namespace(self) -> str:
        """Return the UAMS base namespace IRI."""
        return self._data.get("namespace", "")

    @property
    def columns(self) -> list[str]:
        """Return all UAMS column names in registry order."""
        return list(self._columns.keys())

    def _column(self, column: str) -> dict:
        """Return the block for a column or raise KeyError."""
        if column not in self._columns:
            raise KeyError(f"Unknown UAMS column: {column}")
        return self._columns[column]

    def term_iri(self, column: str) -> str | None:
        """Return the expanded IRI of the column's first (preferred) term.

        Returns None when the column has no external ontology term.
        """
        terms = self._column(column).get("terms") or []
        if not terms:
            return None
        return self.expand_curie(terms[0]["curie"])

    def canonical_unit(self, column: str) -> str | None:
        """Return the column's honest UCUM unit, or None if dimensionless."""
        return self._column(column).get("unit_ucum")

    def synonyms(self, column: str) -> list[str]:
        """Return the column's synonym list (possibly empty)."""
        return list(self._column(column).get("synonyms") or [])

    def validation_range(self, column: str) -> tuple[float | None, float | None]:
        """Return the (min, max) validation range; either bound may be None."""
        validation = self._column(column).get("validation") or {}
        return validation.get("min"), validation.get("max")

    def expand_curie(self, curie: str) -> str:
        """Expand a CURIE (PREFIX:localid) to its full IRI.

        Raises KeyError if the prefix is not declared in the registry.
        """
        prefix, localid = _split_curie(curie)
        if prefix not in self._prefixes:
            raise KeyError(f"Unknown prefix in CURIE: {curie}")
        return f"{self._prefixes[prefix]}{localid}"

    def validate(self) -> list[str]:
        """Check CURIE hygiene and cross-column uniqueness.

        Returns a list of human-readable problem strings (empty when clean).
        Does not raise on data problems; only expand_curie raises (unknown
        prefix) and that path is exercised explicitly here per CURIE.
        """
        problems: list[str] = []
        owner: dict[str, str] = {}
        shared = set(self._data.get("permitted_shared_terms") or {})
        for column, block in self._columns.items():
            for term in block.get("terms") or []:
                curie = term.get("curie", "")
                problems.extend(self._check_curie_syntax(column, curie))
                if curie not in shared:
                    self._check_collision(column, curie, owner, problems)
        return problems

    def _check_curie_syntax(self, column: str, curie: str) -> list[str]:
        """Return syntax problems for a single CURIE on a column."""
        prefix, localid = _split_curie(curie)
        if prefix not in self._prefixes:
            return [f"{column}: undeclared prefix in CURIE '{curie}'"]
        pattern = _LOCALID_PATTERNS.get(prefix)
        if pattern is None:
            return [f"{column}: no syntax rule for prefix '{prefix}'"]
        if not pattern.match(localid):
            return [f"{column}: bad local id '{localid}' for prefix '{prefix}'"]
        return []

    @staticmethod
    def _check_collision(column: str, curie: str, owner: dict, problems: list) -> None:
        """Record CURIE ownership and append a problem on cross-column reuse."""
        prior = owner.get(curie)
        if prior is not None and prior != column:
            problems.append(f"CURIE collision: '{curie}' used by both {prior} and {column}")
        else:
            owner[curie] = column


if __name__ == "__main__":
    reg = Registry.load()
    assert len(reg.columns) == 296, len(reg.columns)
    issues = reg.validate()
    assert issues == [], issues
    assert reg.expand_curie("AGROVOC:c_5192").endswith("agrovoc/c_5192")
    assert reg.expand_curie("CO_320:0000005").endswith("CO_320:0000005")
    # c_5192 is "nitrogen"; c_5188, asserted here previously, is "nitric acid".
    nitrogen_iri = reg.term_iri("Nitrogen")
    assert nitrogen_iri is not None and nitrogen_iri.endswith("c_5192"), nitrogen_iri
    assert reg.term_iri("Paper_ID") is None
    assert reg.canonical_unit("Rainfall") == "mm"
    assert reg.validation_range("Soil_pH") == (3.0, 10.0)
    assert "ph" in reg.synonyms("Soil_pH")
    try:
        reg.expand_curie("NOPE:123")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown prefix should raise")
    print("registry smoke OK: 296 cols, validate() clean")
