"""UAMS namespace helpers — the base IRI and per-column term minting.

The base IRI is read from the ontology YAML so there is a single source of
truth; do not hardcode the namespace string elsewhere.
"""

from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_YAML = Path(__file__).resolve().parents[2] / "spec" / "uams_ontology.yaml"


@lru_cache(maxsize=4)
def _read_namespace(path: str) -> str:
    """Return the 'namespace' value declared in the ontology YAML."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    namespace = data.get("namespace")
    if not namespace:
        raise ValueError(f"No 'namespace' declared in {path}")
    return namespace


def uams_base(path: Path = _DEFAULT_YAML) -> str:
    """Return the UAMS base IRI (e.g. 'https://w3id.org/uams#')."""
    return _read_namespace(str(path))


# Module-level constant for the common case (default registry path).
UAMS_BASE = uams_base()


def uams_term(name: str, base: str = UAMS_BASE) -> str:
    """Mint the UAMS IRI for a column/term name by appending it to the base.

    Args:
        name: A UAMS column name, e.g. 'Soil_pH'.
        base: The base IRI to append to (defaults to UAMS_BASE).

    Returns:
        The full term IRI, e.g. 'https://w3id.org/uams#Soil_pH'.
    """
    if not name:
        raise ValueError("term name must be non-empty")
    return f"{base}{name}"


if __name__ == "__main__":
    assert UAMS_BASE.startswith("http"), UAMS_BASE
    iri = uams_term("Soil_pH")
    assert iri == f"{UAMS_BASE}Soil_pH", iri
    assert iri.endswith("#Soil_pH"), iri
    try:
        uams_term("")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("empty name should raise")
    print("namespace smoke OK:", UAMS_BASE, "->", iri)
