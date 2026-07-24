"""Leakage control for the ML feature matrix.

A feature must never encode information unavailable at prediction time. Leaky columns
are identified from the UAMS schema (post-harvest, target, and prediction groups) plus
general structural rules — stripping engineered suffixes, detecting outcome-family
interaction terms, and target-family collinearity — so the guard generalizes to
engineered features it has never seen, rather than matching a hardcoded list of names.
"""

import re

from agri_ai_agent.config.schema import (
    NON_FEATURE_COLS,
    POST_HARVEST_VARIABLES,
    SCHEMA_GROUPS,
)

PREDICTION_COLUMNS = frozenset(SCHEMA_GROUPS.get("N. ML Predictions", []))
TARGET_COLUMNS = frozenset(SCHEMA_GROUPS.get("J. ML Target Variables", []))

# Columns that are outcomes, model outputs, or otherwise unavailable at prediction time.
OUTCOME_COLUMNS = frozenset(POST_HARVEST_VARIABLES) | PREDICTION_COLUMNS | TARGET_COLUMNS

# Post-harvest outcome families the feature engineer builds interactions/transforms from.
# Any engineered feature whose name references one of these roots leaks the outcome
# (e.g. Yield_x_N, Biomass_log). These are the two agronomic quantities that are
# themselves prediction targets, so they never legitimately appear inside a predictor.
OUTCOME_INTERACTION_ROOTS = ("yield", "biomass")

_DUP_SUFFIX = re.compile(r"_\d+$")
_ENGINEERED_SUFFIXES = (
    "_calc", "_7d_ma", "_squared", "_cubed", "_log1p", "_log", "_sqrt",
    "_category", "_bin", "_zscore",
)
_UNIT_STAGE_SUFFIX = re.compile(r"_(cm2|cm|mm|kg|ha|g|pct|percent|\d+)$")

# Tokens too generic to signal that a feature belongs to a target's family.
_GENERIC_TOKENS = frozenset({
    "plant", "per", "of", "the", "and", "ratio", "index", "value",
    "content", "avg", "mean", "total", "sum", "log", "sqrt",
})


def strip_engineered(name: str) -> str:
    """Strip trailing engineered/duplicate suffixes to recover a base column name."""
    current = name
    prev = None
    while prev != current:
        prev = current
        current = _DUP_SUFFIX.sub("", current)
        for suffix in _ENGINEERED_SUFFIXES:
            if current.lower().endswith(suffix):
                current = current[: -len(suffix)]
                break
    return current


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _target_stem(target_col: str) -> str:
    """Family stem of a target, dropping trailing unit/stage tokens (cm, g, 30, 60...)."""
    stem = strip_engineered(target_col)
    prev = None
    while prev != stem:
        prev = stem
        stem = _UNIT_STAGE_SUFFIX.sub("", stem)
    return stem


def _compact(name: str) -> str:
    """Lowercase name with all separators removed (Fruit_Weight -> fruitweight)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _acronym(name: str) -> str:
    """First letter of each token (Harvest_Index -> hi); empty for single-token names."""
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", name) if t]
    return "".join(t[0] for t in tokens).lower() if len(tokens) >= 2 else ""


# Signatures of outcome columns, so engineered features that abbreviate them
# (FruitWeight_log from Fruit_Weight, HI_log from Harvest_Index) are still caught.
_OUTCOME_COMPACT = frozenset(_compact(c) for c in OUTCOME_COLUMNS)
_OUTCOME_ACRONYMS = frozenset(a for a in (_acronym(c) for c in OUTCOME_COLUMNS) if a)
_INTERACTION_MARKERS = ("_x_", "_interaction", "_ratio", "_div_", "_per_")


def _is_engineered(name: str) -> bool:
    """True if the name looks like an engineered feature (suffix, dup, or interaction)."""
    low = name.lower()
    return bool(
        _DUP_SUFFIX.search(name)
        or any(low.endswith(s) for s in _ENGINEERED_SUFFIXES)
        or any(marker in low for marker in _INTERACTION_MARKERS)
    )


def _is_outcome_derived(col: str) -> bool:
    """True if an engineered feature derives from an outcome, matched by compact/acronym."""
    if not _is_engineered(col):
        return False
    compact_base = _compact(strip_engineered(col))
    if compact_base in _OUTCOME_COMPACT or compact_base in _OUTCOME_ACRONYMS:
        return True
    for signature in _OUTCOME_COMPACT:
        if len(signature) >= 5 and len(compact_base) >= 4 and (
            compact_base in signature or signature in compact_base
        ):
            return True
    return False


def is_leaky_feature(col: str, target_col: str | None = None) -> bool:
    """Return True if ``col`` must be excluded from the feature matrix for ``target_col``."""
    base = strip_engineered(col)
    if col in OUTCOME_COLUMNS or base in OUTCOME_COLUMNS:
        return True
    if col in PREDICTION_COLUMNS or base in PREDICTION_COLUMNS:
        return True
    if _tokens(col) & set(OUTCOME_INTERACTION_ROOTS):
        return True
    if _is_outcome_derived(col):
        return True
    if target_col:
        if col == target_col:
            return True
        stem = _target_stem(target_col)
        if stem and (col == stem or col.startswith(stem + "_") or base == stem):
            return True
        target_tokens = {t for t in _tokens(stem) if len(t) >= 3 and t not in _GENERIC_TOKENS}
        feature_tokens = {t for t in _tokens(col) if len(t) >= 3 and t not in _GENERIC_TOKENS}
        if target_tokens & feature_tokens:
            return True
    return False


def select_feature_columns(df, target_col, base_exclude=frozenset()):
    """Return the numeric columns of ``df`` that are safe predictors of ``target_col``."""
    numeric_cols = df.select_dtypes(include="number").columns
    return [
        col for col in numeric_cols
        if col != target_col
        and col not in NON_FEATURE_COLS
        and col not in base_exclude
        and not is_leaky_feature(col, target_col)
    ]


CORRELATION_LEAK_THRESHOLD = 0.999


def drop_suspected_leaks(X, y, threshold=CORRELATION_LEAK_THRESHOLD):
    """Drop features whose correlation with the target is near-perfect (suspected leakage).

    A data-driven safety net for outcome-derived features the name-based guard might miss.
    Only near-identical-to-target columns (|corr| >= threshold) are removed, so genuine
    strong predictors survive. Returns (X_without_leaks, dropped_column_names).
    """
    dropped = []
    for col in X.columns:
        series = X[col]
        if series.nunique(dropna=True) < 2:
            continue
        corr = series.corr(y)
        if corr is not None and abs(corr) >= threshold:
            dropped.append(col)
    return (X.drop(columns=dropped) if dropped else X), dropped


if __name__ == "__main__":
    import pandas as pd

    # Outcome-derived features leak regardless of target.
    assert is_leaky_feature("Yield_x_N", "Plant_Height_cm")
    assert is_leaky_feature("Predicted_Yield", "Plant_Height_cm")
    assert is_leaky_feature("Yield_per_Hectare_1", "Plant_Height_cm")
    # Same-family collinearity leaks only relative to the active target.
    assert is_leaky_feature("Plant_Height_30_cm", "Plant_Height_cm")
    assert is_leaky_feature("Height_x_N", "Plant_Height_cm")
    assert not is_leaky_feature("Plant_Height_30_cm", "Yield_per_Hectare")
    # Legitimate design/soil predictors are never falsely excluded.
    assert not is_leaky_feature("Nitrogen", "Yield_per_Hectare")
    assert not is_leaky_feature("Spacing_Plant", "Plant_Height_cm")

    frame = pd.DataFrame({
        "Nitrogen": [1.0, 2.0], "Soil_pH": [6.5, 7.0], "Rainfall": [100, 200],
        "Yield_per_Hectare": [10, 20], "Yield_x_N": [10, 40], "Predicted_Yield": [9, 19],
        "Yield_per_Hectare_1": [10, 20],
    })
    feats = select_feature_columns(frame, "Yield_per_Hectare")
    assert set(feats) == {"Nitrogen", "Soil_pH", "Rainfall"}, feats
    print("leakage smoke OK ->", feats)
