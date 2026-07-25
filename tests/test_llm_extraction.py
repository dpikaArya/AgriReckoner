"""Offline tests for the LLM extraction strategy (no network; injected fake completer)."""

import json

import pytest

from agri_ai_agent.agents.llm_extraction_agent import LLMExtractionAgent
from agri_ai_agent.extractors.grounding import ground_field
from agri_ai_agent.extractors.base import ExtractedField
from agri_ai_agent.extractors.llm_extractor import LLMExtractor, build_extraction_schema


def _fake_completer(payload):
    def complete(system, user, schema):
        return json.dumps(payload)
    return complete


def test_schema_lists_registry_columns():
    schema = build_extraction_schema()
    enum = schema["properties"]["fields"]["items"]["properties"]["column"]["enum"]
    assert "Soil_pH" in enum and "Nitrogen" in enum


def test_extractor_parses_structured_output():
    payload = {"crop": "Rice", "doi": "10.1/y", "year": 2019, "fields": [
        {"column": "Rainfall", "value": 850, "unit_as_reported": "mm",
         "source_quote": "Total rainfall was 850 mm.", "model_confidence": 0.8}]}
    result = LLMExtractor(_fake_completer(payload)).extract("...", "p9")
    assert result.crop == "Rice"
    assert result.fields[0].column == "Rainfall"
    assert result.fields[0].status == "unverified"


def test_grounding_accepts_echoed_in_range_value():
    item = ground_field(ExtractedField("Rainfall", 850.0, "mm", "Total rainfall was 850 mm."))
    assert item.status == "reported"


def test_grounding_rejects_hallucinated_value():
    item = ground_field(ExtractedField("Rainfall", 850.0, "mm", "Rainfall was moderate."))
    assert item.status == "rejected"
    assert "not found" in item.reject_reason


def test_agent_row_contains_only_grounded_values():
    payload = {"crop": "Wheat", "doi": None, "year": 2020, "fields": [
        {"column": "Soil_pH", "value": 6.8, "unit_as_reported": None,
         "source_quote": "Soil pH was 6.8.", "model_confidence": 0.9},
        {"column": "Nitrogen", "value": 9999, "unit_as_reported": "kg/ha",
         "source_quote": "N was 9999 kg/ha.", "model_confidence": 0.2}]}
    agent = LLMExtractionAgent(extractor=LLMExtractor(_fake_completer(payload)))
    agent.run(df=None, papers={"p1": "Soil pH was 6.8. N was 9999 kg/ha."})
    row = agent.dataframe.iloc[0]
    assert row["Soil_pH"] == 6.8
    assert "Nitrogen" not in agent.dataframe.columns  # out-of-range -> rejected -> excluded


def test_agent_offline_no_key_is_noop():
    agent = LLMExtractionAgent(extractor=None)
    agent.settings.OPENAI_API_KEY = ""
    contract = agent.run(df=None, papers={"p1": "text"})
    assert contract.status == "success"
    assert any("skipped" in w for w in contract.warnings)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
