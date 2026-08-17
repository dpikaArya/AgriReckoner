"""Pluggable extraction strategies that turn agronomy paper text into UAMS rows.

Strategies share one interface (``Extractor``): ``LLMExtractor`` uses OpenAI
structured outputs, and grounding validates every LLM-extracted value against
the source span and the ontology registry before it is trusted.  When no API
key or client is available the framework falls back to a no-op extraction
(no data is fabricated).
"""

from agri_ai_agent.extractors.base import ExtractedField, Extractor
from agri_ai_agent.extractors.fields import EXTRACTION_FIELDS, ExtractionField

__all__ = ["ExtractedField", "Extractor", "EXTRACTION_FIELDS", "ExtractionField"]
