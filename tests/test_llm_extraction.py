"""Offline tests for the LLM extraction strategy (no network; injected fake completer)."""

import json

import pytest

from agri_ai_agent.agents.llm_extraction_agent import LLMExtractionAgent
from agri_ai_agent.extractors.base import ExtractedField
from agri_ai_agent.extractors.grounding import ground_field
from agri_ai_agent.extractors.llm_extractor import LLMExtractor, build_extraction_schema


def _fake_completer(payload):
    def complete(system, user, schema):
        return json.dumps(payload)

    return complete


def test_schema_lists_registry_columns():
    schema = build_extraction_schema()
    enum = schema["properties"]["fields"]["items"]["properties"]["column"]["enum"]
    assert "Soil_pH" in enum and "Nitrogen" in enum


def test_a_whole_paper_is_sent_when_it_fits():
    from agri_ai_agent.extractors.llm_extractor import MAX_PROMPT_CHARS, _select_text

    paper = "a" * (MAX_PROMPT_CHARS - 10)
    assert _select_text(paper) == paper


def test_results_tables_reach_the_model_in_a_long_paper():
    """Regression: the prompt was cut to the first 12,000 characters.

    Measured first-table offsets in real open-access agronomy papers were 14.7k, 22.3k,
    51.5k and 82.0k characters, so the model was shown title, abstract and introduction
    and never a single results table — it could only report metadata.
    """
    from agri_ai_agent.extractors.llm_extractor import MAX_PROMPT_CHARS, _select_text

    filler = "Introduction and background prose. " * 4000
    table = "Table 3 Grain yield of treatments: T1 2450 kg ha-1, T2 3110 kg ha-1."
    paper = filler + ("padding text. " * 8000) + table + ("more prose. " * 4000)
    assert len(paper) > MAX_PROMPT_CHARS

    selected = _select_text(paper)
    assert table in selected
    assert len(selected) <= MAX_PROMPT_CHARS + 200  # separators only


def test_the_head_is_kept_because_it_defines_the_treatment_codes():
    """T1/T2 mean nothing without the methods section that assigns doses to them."""
    from agri_ai_agent.extractors.llm_extractor import MAX_PROMPT_CHARS, _select_text

    head = "Methods: T1 = 0 kg N/ha control, T2 = 120 kg N/ha. "
    paper = head + ("filler prose. " * 20000) + "Table 2 yield T1 2450 T2 3110"
    assert len(paper) > MAX_PROMPT_CHARS
    selected = _select_text(paper)
    assert "T2 = 120 kg N/ha" in selected


def test_extractor_parses_structured_output():
    payload = {
        "crop": "Rice",
        "doi": "10.1/y",
        "year": 2019,
        "fields": [
            {
                "column": "Rainfall",
                "value": 850,
                "unit_as_reported": "mm",
                "source_quote": "Total rainfall was 850 mm.",
                "model_confidence": 0.8,
            }
        ],
    }
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
    payload = {
        "crop": "Wheat",
        "doi": None,
        "year": 2020,
        "fields": [
            {
                "column": "Soil_pH",
                "value": 6.8,
                "unit_as_reported": None,
                "source_quote": "Soil pH was 6.8.",
                "model_confidence": 0.9,
            },
            {
                "column": "Nitrogen",
                "value": 9999,
                "unit_as_reported": "kg/ha",
                "source_quote": "N was 9999 kg/ha.",
                "model_confidence": 0.2,
            },
        ],
    }
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
