"""Pluggable extraction strategies that turn agronomy paper text into UAMS rows.

Strategies share one interface (``Extractor``): ``RegexExtractor`` is the offline/CI
default, ``LLMExtractor`` uses OpenAI structured outputs, and grounding validates every
LLM-extracted value against the source span and the ontology registry before it is
trusted. The framework must still run without an API key, so callers fall back to the
regex strategy when no key or client is available.
"""

from agri_ai_agent.extractors.base import ExtractedField, Extractor
from agri_ai_agent.extractors.fields import EXTRACTION_FIELDS, ExtractionField

__all__ = ["ExtractedField", "Extractor", "EXTRACTION_FIELDS", "ExtractionField"]
