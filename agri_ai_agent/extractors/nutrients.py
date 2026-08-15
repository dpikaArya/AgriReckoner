"""Nutrient basis resolution: oxide figures converted to the elemental basis.

UAMS stores nutrients as the ELEMENT. `Phosphorus` means P, not P2O5, and `Potassium`
means K, not K2O (decided in issue #20). Sources use both bases and they differ by a
fixed stoichiometric factor, so folding an oxide figure into an elemental column without
converting overstates phosphorus by 2.29x and potassium by 1.20x. Indian package-of-
practices recommendations are conventionally written N:P2O5:K2O, so this is the common
case rather than an edge one.

Three outcomes, never two: a label naming an oxide is converted, a label naming the
element is kept, and a label whose basis cannot be read -- `phosphate`, `potash`, a
fertiliser product name -- is REFUSED. Refusing is the point: silently assuming a basis
is what the specification did before, and it is unrecoverable once the number is stored.
"""

import re

ELEMENTAL = "elemental"
OXIDE = "oxide"
AMBIGUOUS = "ambiguous"
UNKNOWN = "unknown"

# What a caller must do with a reported value to store it on the elemental basis.
CONVERT = "convert"
KEEP = "keep"
REFUSE = "refuse"

# IUPAC 2021 standard atomic weights, in u.
_ATOMIC_MASS = {
    "O": 15.999,
    "Na": 22.98976928,
    "Mg": 24.305,
    "P": 30.973762,
    "S": 32.06,
    "K": 39.0983,
    "Ca": 40.078,
}

# The oxide bases used in fertiliser and soil-test reporting.
# formula -> (element symbol, atoms of that element, atoms of oxygen, UAMS column)
_OXIDES = {
    "P2O5": ("P", 2, 5, "Phosphorus"),
    "K2O": ("K", 2, 1, "Potassium"),
    "CaO": ("Ca", 1, 1, "Calcium"),
    "MgO": ("Mg", 1, 1, "Magnesium"),
    "SO3": ("S", 1, 3, "Sulphur"),
    "Na2O": ("Na", 2, 1, "Sodium"),
}

# Element names and symbols that declare the elemental basis outright.
_ELEMENT_WORDS = {
    "p": "P",
    "phosphorus": "P",
    "phosphorous": "P",
    "k": "K",
    "potassium": "K",
    "ca": "Ca",
    "calcium": "Ca",
    "mg": "Mg",
    "magnesium": "Mg",
    "s": "S",
    "sulphur": "S",
    "sulfur": "S",
    "na": "Na",
    "sodium": "Na",
}

# Words that name a nutrient without declaring its basis. `phosphate` and `potash` are
# reported as the oxide in some conventions and as the element in others, and fertiliser
# products imply a basis by convention only. Reading either as elemental is a guess.
_AMBIGUOUS = re.compile(
    r"(?<![a-z])(phosphate|phosphoric|potash|muriate|superphosphate|mop|sop|dap|ssp|tsp)(?![a-z])"
)

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")

# A label can name the oxide in words instead of a formula ("potassium oxide", "phosphorus
# pentoxide", "K on an oxide basis"). Any of these plus an element name means that element's
# oxide, which spares us a second table of spelled-out names.
_OXIDE_WORD = re.compile(r"(?<![a-z])(oxide|pentoxide|trioxide|monoxide)(?![a-z])")

# Mass units that are also element symbols. "mg" in "ap (mg/kg)" is a milligram, so a mass
# unit standing in front of a denominator is a unit, not magnesium.
_MASS_UNITS = frozenset({"mg", "g", "kg"})
_DENOMINATORS = frozenset({"kg", "g", "l", "ha", "m2", "m3", "ml", "plant", "pot"})


def _normalise(label):
    """Lower-case a label, fold sub/superscripts to digits, and reduce separators to spaces."""
    if not label:
        return ""
    folded = str(label).translate(_SUBSCRIPTS).lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _formula_pattern(formula):
    """Build a delimited, space-tolerant pattern for an oxide formula ('P2O5' -> p 2 o 5)."""
    body = r"\s*".join(re.escape(ch) for ch in formula.lower())
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


_OXIDE_PATTERNS = tuple((name, _formula_pattern(name)) for name in _OXIDES)


def oxide_factor(formula):
    """Return the mass fraction of the element in an oxide (P2O5 -> 0.4364)."""
    element, n_element, n_oxygen, _ = _OXIDES[formula]
    element_mass = n_element * _ATOMIC_MASS[element]
    return element_mass / (element_mass + n_oxygen * _ATOMIC_MASS["O"])


def _is_unit_token(tokens, index):
    """True when a mass-unit symbol is the numerator of a ratio ('mg' in 'mg kg')."""
    if tokens[index] not in _MASS_UNITS:
        return False
    return index + 1 < len(tokens) and tokens[index + 1] in _DENOMINATORS


def _element_in(text):
    """Return the element a label names, preferring spelled-out names over symbols.

    Symbols are searched last, and a mass unit reading of a symbol wins over the element,
    because the "mg" in "ap (mg/kg)" is a milligram rather than magnesium.
    """
    tokens = text.split()
    usable = [(i, t) for i, t in enumerate(tokens) if not _is_unit_token(tokens, i)]
    names = [(i, t) for i, t in usable if len(t) > 2]
    for _, token in names + usable:
        if token in _ELEMENT_WORDS:
            return _ELEMENT_WORDS[token]
    return None


def _oxide_of(element):
    """Return the oxide formula whose element is ``element``, or None."""
    for formula, spec in _OXIDES.items():
        if spec[0] == element:
            return formula
    return None


def detect_basis(label):
    """Return (basis, term) for a column label or reported nutrient name.

    basis is OXIDE, ELEMENTAL, AMBIGUOUS or UNKNOWN, and term is what declared it -- a formula
    or an element symbol. An oxide formula anywhere in the label wins, because
    "kg P2O5 ha-1" names both the element and the oxide and means the oxide.
    """
    text = _normalise(label)
    if not text:
        return UNKNOWN, None
    for formula, pattern in _OXIDE_PATTERNS:
        if pattern.search(text):
            return OXIDE, formula
    if _OXIDE_WORD.search(text):
        spelled = _oxide_of(_element_in(text))
        if spelled:
            return OXIDE, spelled
    if _AMBIGUOUS.search(text):
        return AMBIGUOUS, None
    element = _element_in(text)
    return (ELEMENTAL, element) if element else (UNKNOWN, None)


def elemental_column(label):
    """Return the UAMS column an oxide label belongs in, or None if it names no oxide."""
    basis, formula = detect_basis(label)
    return _OXIDES[formula][3] if basis == OXIDE else None


def to_elemental(value, label):
    """Convert ``value`` to the elemental basis using the basis declared by ``label``.

    Returns (value, ok, reason). ok is False only when the label names a form whose basis is
    genuinely not fixed ("potash"); such a value must not be stored as an elemental figure.
    A label that simply says nothing about basis is read on the schema's basis, elemental.
    """
    if value is None:
        return None, True, "no value"
    basis, formula = detect_basis(label)
    if basis == OXIDE:
        factor = oxide_factor(formula)
        element = _OXIDES[formula][0]
        return value * factor, True, f"{formula} -> {element} (x{factor:.4f})"
    if basis == AMBIGUOUS:
        return value, False, f"{label!r} names a form whose basis is not fixed; converting is a guess"
    return value, True, f"read on the schema basis, elemental ({formula or 'unstated'})"


def oxide_aliases():
    """Return the oxide formulas this module recognises, for drift tests against the schema."""
    return frozenset(_OXIDES)


def basis_sensitive_columns():
    """Return the UAMS columns whose meaning changes with the oxide/elemental basis.

    Nitrogen is absent deliberately: fertiliser nitrogen is always reported as N.
    """
    return frozenset(spec[3] for spec in _OXIDES.values())


def plan_conversions(labels_to_columns):
    """Decide how each source label must be treated to reach the elemental basis.

    Takes {source_label: uams_column} and returns {source_label: (action, factor, reason)},
    where action is CONVERT, KEEP or REFUSE. Only labels landing in a basis-sensitive column
    are judged; every other column passes through as KEEP so callers can apply this blindly.
    """
    sensitive = basis_sensitive_columns()
    plan = {}
    for label, column in labels_to_columns.items():
        if column not in sensitive:
            plan[label] = (KEEP, 1.0, "not basis-sensitive")
            continue
        basis, formula = detect_basis(label)
        if basis == OXIDE:
            factor = oxide_factor(formula)
            plan[label] = (CONVERT, factor, f"{formula} -> {_OXIDES[formula][0]} (x{factor:.4f})")
        elif basis == AMBIGUOUS:
            plan[label] = (
                REFUSE,
                1.0,
                f"{label!r} names a nutrient form whose basis is not fixed; converting is a guess",
            )
        else:
            plan[label] = (KEEP, 1.0, f"read on the schema basis, elemental ({formula or 'unstated'})")
    return plan


if __name__ == "__main__":
    assert abs(oxide_factor("P2O5") - 0.4364) < 1e-4, oxide_factor("P2O5")
    assert abs(oxide_factor("K2O") - 0.8301) < 1e-4, oxide_factor("K2O")
    assert detect_basis("P2O5_kg_ha") == (OXIDE, "P2O5")
    assert detect_basis("P₂O₅ (kg ha⁻¹)") == (OXIDE, "P2O5")
    assert detect_basis("fertiliser_p_kg_p_ha")[0] == ELEMENTAL
    assert detect_basis("soil_p_olsen_mg_kg")[0] == ELEMENTAL
    assert detect_basis("potash")[0] == AMBIGUOUS
    assert detect_basis("cacao_yield")[0] == UNKNOWN, detect_basis("cacao_yield")
    assert detect_basis("potassium oxide") == (OXIDE, "K2O")
    assert detect_basis("P²O⁵") == (OXIDE, "P2O5")
    v, ok, why = to_elemental(60.0, "P2O5 kg/ha")
    assert ok and abs(v - 26.19) < 0.01, (v, why)
    v, ok, why = to_elemental(60.0, "available_p")
    assert ok and v == 60.0, (v, why)
    v, ok, why = to_elemental(60.0, "potash")
    assert not ok, why
    plan = plan_conversions(
        {
            "P2O5_kg_ha": "Phosphorus",
            "available_p": "Phosphorus",
            "potash": "Potassium",
            "Rainfall_mm": "Rainfall",
        }
    )
    assert plan["P2O5_kg_ha"][0] == CONVERT, plan["P2O5_kg_ha"]
    assert plan["available_p"][0] == KEEP, plan["available_p"]
    assert plan["potash"][0] == REFUSE, plan["potash"]
    assert plan["Rainfall_mm"][0] == KEEP, plan["Rainfall_mm"]
    print("nutrients smoke OK -> 60 kg P2O5/ha = 26.19 kg P/ha; elemental kept; potash refused")
