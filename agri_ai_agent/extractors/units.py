"""Unit normalization and conversion for extracted agronomic values.

The LLM reports values AS-WRITTEN ("4.2 t/ha", "13.4 g kg-1"). To range-check and store a
value against a column's canonical unit (from the ontology registry) we convert within a
dimensional group, and REFUSE conversion across incompatible dimensions — soil "g/kg" is a
different quantity from an applied "kg/ha" rate, so it must be flagged, not silently coerced.
"""

import re

# canonical token -> (dimension, factor to that dimension's base unit)
_UNIT_FACTORS = {
    # mass per area (base: kg/ha)
    "kg/ha": ("mass_per_area", 1.0),
    "t/ha": ("mass_per_area", 1000.0),
    "q/ha": ("mass_per_area", 100.0),
    "g/m2": ("mass_per_area", 10.0),
    # mass fraction / concentration (base: %)
    "%": ("fraction", 1.0),
    "g/100g": ("fraction", 1.0),
    "g/kg": ("fraction", 0.1),
    "mg/g": ("fraction", 0.1),
    "mg/kg": ("fraction", 1e-4),
    "ppm": ("fraction", 1e-4),
    # length (base: cm)
    "cm": ("length", 1.0),
    "mm": ("length", 0.1),
    "m": ("length", 100.0),
    # mass (base: g)
    "g": ("mass", 1.0),
    "kg": ("mass", 1000.0),
    "mg": ("mass", 0.001),
    # electrical conductivity (base: ds/m)
    "ds/m": ("conductivity", 1.0),
    "ms/cm": ("conductivity", 1.0),
    "us/cm": ("conductivity", 0.001),
}
_TEMPERATURE = {"c", "cel", "celsius", "degc", "f", "fahrenheit", "degf", "k", "kelvin"}
_TEMPERATURE_TOKENS = {"cel", "f", "k"}

_NUMERATORS = ("mg", "kg", "cmol", "meq", "ds", "ms", "us", "t", "q", "g", "mm", "cm", "m")
_DENOMINATORS = ("ha", "100g", "kg", "m2", "m", "cm", "g", "l")


def _split_ratio(token: str) -> str:
    """Split a concatenated ratio like 'gkg' -> 'g/kg' using a small unit lexicon."""
    for num in _NUMERATORS:
        if token.startswith(num):
            rest = token[len(num) :]
            if rest in _DENOMINATORS:
                return f"{num}/{rest}"
    return token


def normalize_unit(raw):
    """Return a canonical unit token (e.g. 'kg/ha', 'g/kg', 'cel') or None if unrecognized."""
    if not raw:
        return None
    unit = raw.strip().lower().replace("−", "-").replace("·", "").replace(" ", "")
    unit = unit.replace("°", "").replace("^", "")
    if unit in {"ph", "-", ""}:
        return None
    if unit in _TEMPERATURE or unit.rstrip("s") in _TEMPERATURE:
        return {"f": "f", "fahrenheit": "f", "degf": "f", "k": "k", "kelvin": "k"}.get(unit, "cel")
    unit = re.sub(r"(-1|\b1)$", "", unit)  # strip trailing inverse marker (kgha-1 -> kgha)
    unit = unit.replace("-", "/")
    if unit in _UNIT_FACTORS:
        return unit
    ratio = _split_ratio(unit)
    return ratio if ratio in _UNIT_FACTORS else None


def canonical_token(registry_unit):
    """Map a registry UCUM unit string to a token understood by this module."""
    if not registry_unit:
        return None
    return normalize_unit(registry_unit) or registry_unit.strip().lower()


def _to_celsius(value, unit):
    if unit == "f":
        return (value - 32.0) * 5.0 / 9.0
    if unit == "k":
        return value - 273.15
    return value


def convert(value, from_unit, to_unit):
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    Returns (converted_value, ok, reason). ok is False with a reason when the units are of
    incompatible dimensions or unknown; when from_unit is None the value passes through
    unchanged and unverified.
    """
    if to_unit is None:
        return value, True, "target dimensionless"
    src = normalize_unit(from_unit)
    dst = canonical_token(to_unit)
    if src is None:
        return value, True, "reported unit missing/unrecognized (unverified)"
    if src in _TEMPERATURE_TOKENS or dst == "cel":
        if src not in _TEMPERATURE_TOKENS and src != "cel":
            return value, False, f"cannot convert {from_unit!r} to temperature"
        return _to_celsius(value, src), True, "temperature"
    if src not in _UNIT_FACTORS or dst not in _UNIT_FACTORS:
        return value, True, "unit not in conversion table (unverified)"
    src_dim, src_factor = _UNIT_FACTORS[src]
    dst_dim, dst_factor = _UNIT_FACTORS[dst]
    if src_dim != dst_dim:
        return value, False, f"unit {from_unit!r} incompatible with canonical {to_unit!r}"
    return value * src_factor / dst_factor, True, "converted"


if __name__ == "__main__":
    assert normalize_unit("t ha-1") == "t/ha"
    assert normalize_unit("g kg-1") == "g/kg"
    assert normalize_unit("mgkg-1") == "mg/kg"
    assert normalize_unit("dS m-1") == "ds/m"
    assert normalize_unit("pH") is None
    v, ok, _ = convert(4.2, "t/ha", "kg/ha")
    assert ok and abs(v - 4200) < 1e-6, v
    v, ok, _ = convert(13.4, "g/kg", "%")
    assert ok and abs(v - 1.34) < 1e-6, v
    v, ok, reason = convert(1.2, "g/kg", "kg/ha")  # concentration vs rate -> incompatible
    assert not ok, reason
    v, ok, _ = convert(95.0, "f", "cel")
    assert ok and abs(v - 35.0) < 1e-6, v
    print("units smoke OK -> 4.2 t/ha=4200 kg/ha; 13.4 g/kg=1.34%; g/kg->kg/ha rejected; 95F=35C")
