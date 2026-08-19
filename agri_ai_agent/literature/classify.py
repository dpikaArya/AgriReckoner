"""Classification heuristics: original experimental studies, designs,
variables and crops.

These rules are intentionally conservative and deterministic — they operate on
official metadata (title, abstract, keywords, publication type) and never on
hallucinated content.  A record is marked ``is_original_study`` only when it
clearly describes a primary experiment and does not match any exclusion
pattern (reviews, editorials, conference abstracts, patents, simulation-only
studies).
"""

from __future__ import annotations

import re

from agri_ai_agent.literature.models import LiteratureRecord
from agri_ai_agent.literature.query import (
    DESIGN_PATTERNS,
    EXCLUDE_ABSTRACT_KEYWORDS,
    EXCLUDE_TITLE_KEYWORDS,
    EXPERIMENTAL_KEYWORDS,
    VARIABLE_PATTERNS,
)

# Publication-type families that are not original experimental studies.
NON_ORIGINAL_TYPES: frozenset[str] = frozenset(
    {
        "review",
        "book-review",
        "editorial",
        "letter",
        "editorial-letter",
        "conference-abstract",
        "proceedings-abstract",
        "meeting-abstract",
        "patent",
        "patent-application",
        "retraction",
        "corrigendum",
        "erratum",
        "reference-entry",
        "preprint" if False else "noop",
    }
)

# Terms that mark a *simulation/modeling-only* study.
SIMULATION_TERMS: tuple[str, ...] = (
    "simulation study",
    "modeling study",
    "modelling study",
    "in silico",
    "computational model",
    "we simulated",
    "we modeled",
    "we modelled",
)

CROP_TERMS: tuple[tuple[str, str], ...] = (
    (r"\brice\b", "Rice"),
    (r"\bwheat\b", "Wheat"),
    (r"\bmaize\b|\bcorn\b", "Maize"),
    (r"\bsorghum\b", "Sorghum"),
    (r"\bmillet\b|\bpearl millet\b", "Millet"),
    (r"\bbarley\b", "Barley"),
    (r"\bsoybean\b|\bsoya\b", "Soybean"),
    (r"\bgroundnut\b|\bpeanut\b", "Groundnut"),
    (r"\bchickpea\b|\bgram\b", "Chickpea"),
    (r"\bpigeon pea\b|\btoor\b|\bred gram\b", "Pigeon Pea"),
    (r"\blentil\b", "Lentil"),
    (r"\bsugarcane\b", "Sugarcane"),
    (r"\bcotton\b", "Cotton"),
    (r"\bpotato\b", "Potato"),
    (r"\btomato\b", "Tomato"),
    (r"\bpepper\b|\bbell pepper\b", "Pepper"),
    (r"\bonion\b", "Onion"),
    (r"\bcarrot\b", "Carrot"),
    (r"\bspinach\b", "Spinach"),
    (r"\bcabbage\b|\bbrassica\b", "Cabbage"),
    (r"\bcanola\b|\brapeseed\b", "Canola"),
    (r"\bsunflower\b", "Sunflower"),
    (r"\btea\b", "Tea"),
    (r"\bcoffee\b", "Coffee"),
    (r"\brubber\b", "Rubber"),
    (r"\bmaize-wheat\b", "Maize-Wheat System"),
    (r"\brice-wheat\b", "Rice-Wheat System"),
)


def classify_publication_type(raw_type: str | None) -> str | None:
    """Fold a provider-specific publication type into a coarse taxonomy."""
    if not raw_type:
        return None
    t = raw_type.strip().lower()
    if "review" in t:
        return "review"
    if "editorial" in t or "editor" in t:
        return "editorial"
    if "letter" in t:
        return "letter"
    if "abstract" in t or "meeting" in t or "conference" in t:
        return "conference-abstract"
    if "patent" in t:
        return "patent"
    if "retraction" in t or "erratum" in t or "corrigendum" in t:
        return "erratum"
    if "preprint" in t:
        return "preprint"
    if "journal" in t or "article" in t or "research" in t or "original" in t:
        return "journal-article"
    if "dataset" in t or "data paper" in t:
        return "dataset"
    if "book" in t or "chapter" in t:
        return "book-chapter"
    return t


def _has_any(text: str | None, terms: tuple[str, ...]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _match_patterns(text: str | None, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return [label for pattern, label in patterns if re.search(pattern, lowered)]


def is_original_experimental_study(record: LiteratureRecord) -> bool:
    """True when the record is (likely) an original experimental study.

    Reviews, editorials, conference abstracts, patents and simulation-only
    studies are excluded.  Records without a clear experimental signal are
    conservatively *not* classified as original (they still enter the raw
    corpus but are excluded from the verified corpus).
    """
    ptype = classify_publication_type(record.publication_type)
    if ptype in NON_ORIGINAL_TYPES:
        return False

    title = record.title or ""
    abstract = record.abstract or ""
    text = f"{title}\n{abstract}".lower()

    if _has_any(title, EXCLUDE_TITLE_KEYWORDS):
        return False
    if _has_any(abstract, EXCLUDE_ABSTRACT_KEYWORDS):
        return False
    if _has_any(text, SIMULATION_TERMS):
        # Only exclude when no experimental method signal is present.
        if not _has_any(text, EXPERIMENTAL_KEYWORDS):
            return False

    # Require at least one explicit experimental-design signal.
    if _match_patterns(text, DESIGN_PATTERNS):
        return True
    return _has_any(text, EXPERIMENTAL_KEYWORDS)


def detect_experimental_designs(record: LiteratureRecord) -> list[str]:
    text = f"{record.title or ''}\n{record.abstract or ''}"
    return _match_patterns(text, DESIGN_PATTERNS)


def detect_study_variables(record: LiteratureRecord) -> list[str]:
    text = f"{record.title or ''}\n{record.abstract or ''}"
    return _match_patterns(text, VARIABLE_PATTERNS)


def detect_crops(record: LiteratureRecord) -> list[str]:
    text = f"{record.title or ''}\n{record.abstract or ''}"
    return _match_patterns(text, CROP_TERMS)


def enrich_record(record: LiteratureRecord) -> LiteratureRecord:
    """Populate pipeline-derived classification fields on a record in place."""
    record.publication_type = classify_publication_type(record.publication_type)
    record.is_original_study = is_original_experimental_study(record)
    record.experimental_design = detect_experimental_designs(record)
    record.study_variables = detect_study_variables(record)
    record.crop_terms = detect_crops(record)
    record.experimental_keywords = [
        kw
        for kw in EXPERIMENTAL_KEYWORDS
        if kw in (record.title or "").lower() or kw in (record.abstract or "").lower()
    ]
    return record
