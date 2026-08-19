"""Automatic Boolean search-query construction for agronomic literature.

The module builds a compact set of Boolean queries that target the study
types and response variables the framework cares about (field experiments,
RCBD, split plot, factorial designs, fertilizer response, …).  Connectors pass
the queries through their own native search syntax, so a plain-AND query is
also emitted for providers without field-aware query languages.
"""

from __future__ import annotations

# Core study-design / trial keywords.
STUDY_TERMS: tuple[str, ...] = (
    "field experiments",
    "field trials",
    "RCBD",
    "randomized complete block",
    "split plot",
    "split-split plot",
    "factorial experiments",
    "strip plot",
    "latin square",
)

# Treatment / response keywords.
TREATMENT_TERMS: tuple[str, ...] = (
    "fertilizer response",
    "nitrogen response",
    "phosphorus response",
    "potassium response",
    "micronutrients",
    "irrigation",
    "plant density",
    "sowing date",
    "cultivar evaluation",
    "yield response",
)

# Context / measurement keywords that sharpen relevance.
CONTEXT_TERMS: tuple[str, ...] = (
    "soil properties",
    "weather observations",
    "crop management",
    "grain yield",
    "biomass",
    "nutrient uptake",
)


def _quoted(term: str) -> str:
    """Quote terms that contain spaces so they are treated as phrases."""
    return f'"{term}"' if " " in term else term


def build_boolean_queries(
    study_terms: tuple[str, ...] = STUDY_TERMS,
    treatment_terms: tuple[str, ...] = TREATMENT_TERMS,
) -> list[str]:
    """Build Boolean queries: one per study-design term ANDed with the
    treatment/response vocabulary (OR-ed), which keeps each query focused on
    an experiment design and its measured responses."""
    queries: list[str] = []
    treatments_or = " OR ".join(_quoted(t) for t in treatment_terms)
    for study in study_terms:
        queries.append(f"{_quoted(study)} AND ({treatments_or})")
    return queries


def build_simple_queries(
    study_terms: tuple[str, ...] = STUDY_TERMS,
    context_terms: tuple[str, ...] = CONTEXT_TERMS,
) -> list[str]:
    """Simpler queries for providers that choke on long Boolean strings."""
    return [
        f"{_quoted(study)} AND {_quoted(context)}"
        for study in study_terms
        for context in context_terms
    ]


def default_search_terms(config_search_terms: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The full query set used when the user does not override it."""
    if config_search_terms:
        return config_search_terms
    return tuple(build_boolean_queries() + build_simple_queries()[:6])


# Keywords that indicate a *non-original-experimental* record and should be
# filtered out of the verified corpus (reviews, editorials, conference
# abstracts, patents, simulation-only studies).
EXCLUDE_TITLE_KEYWORDS: tuple[str, ...] = (
    "editorial",
    "letter to the editor",
    "book review",
    "conference abstract",
    "abstract only",
    "retraction",
    "corrigendum",
    "erratum",
    "patent",
)

EXCLUDE_ABSTRACT_KEYWORDS: tuple[str, ...] = (
    "we review",
    "this review",
    "the review",
    "literature review",
    "systematic review",
    "meta-analysis",
    "meta analysis",
)

# Keywords that signal a genuine experimental study in title/abstract.
EXPERIMENTAL_KEYWORDS: tuple[str, ...] = (
    "field trial",
    "field experiment",
    "field test",
    "rcbd",
    "randomized complete block",
    "randomized block",
    "split plot",
    "split-plot",
    "factorial",
    "strip plot",
    "latin square",
    "trial",
    "experiment",
    "treatments",
    "replicated",
    "plot",
)

# Experimental design vocabulary -> canonical design labels.
DESIGN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"rcbd|randomized\s+complete\s+block", "Randomized Complete Block Design"),
    (r"randomized\s+block", "Randomized Block Design"),
    (r"split[\s-]*plot", "Split Plot"),
    (r"split[\s-]*split[\s-]*plot", "Split-Split Plot"),
    (r"factorial", "Factorial"),
    (r"strip[\s-]*plot", "Strip Plot"),
    (r"latin\s+square", "Latin Square"),
    (r"crd|completely\s+randomized", "Completely Randomized Design"),
    (r"rcbd|augmented\s+design", "Augmented Design"),
)

# Study variable vocabulary -> canonical variable labels.
VARIABLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bN\b|nitrogen", "Nitrogen"),
    (r"\bP\b|phosphorus|phosphate", "Phosphorus"),
    (r"\bK\b|potassium|potash", "Potassium"),
    (r"micronutrient", "Micronutrients"),
    (r"irrigation|water\s+use", "Irrigation"),
    (r"plant\s+density|planting\s+density|spacing", "Plant Density"),
    (r"sowing\s+date|planting\s+date", "Sowing Date"),
    (r"cultivar|variety|genotype", "Cultivar Evaluation"),
    (r"yield", "Yield"),
    (r"biomass", "Biomass"),
    (r"soil\s+properties|soil\s+physical|soil\s+chemical", "Soil Properties"),
    (r"weather|rainfall|temperature|climat", "Weather Observations"),
    (r"fertilizer|fertiliser|manure|compost", "Fertilizer Response"),
    (r"nue|nutrient\s+use\s+efficiency", "Nutrient Use Efficiency"),
)
