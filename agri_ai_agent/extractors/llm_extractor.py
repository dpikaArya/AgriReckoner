"""LLM extraction strategy: OpenAI structured outputs -> as-reported UAMS fields.

The extractor is decoupled from the OpenAI SDK: it is given a ``complete(system, user,
schema) -> json_string`` callable, so tests inject a deterministic mock and production
injects :func:`make_openai_completer`. Values come back UNVERIFIED — grounding (see
``extractors.grounding``) is what promotes them to trusted.
"""

import json
from collections.abc import Callable
from pathlib import Path

from agri_ai_agent.extractors.base import ExtractedField, ExtractionResult
from agri_ai_agent.extractors.fields import EXTRACTION_COLUMNS

Completer = Callable[[str, str, dict], str]
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def build_extraction_schema() -> dict:
    """JSON schema for the extraction call, generated from the registry-derived columns."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "crop": {"type": ["string", "null"]},
            "doi": {"type": ["string", "null"]},
            "year": {"type": ["integer", "null"]},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "column": {"type": "string", "enum": EXTRACTION_COLUMNS},
                        "value": {"type": ["number", "null"]},
                        "unit_as_reported": {"type": ["string", "null"]},
                        "source_quote": {"type": ["string", "null"]},
                        "model_confidence": {"type": ["number", "null"]},
                    },
                    "required": ["column", "value", "unit_as_reported", "source_quote", "model_confidence"],
                },
            },
        },
        "required": ["crop", "doi", "year", "fields"],
    }


def _build_user_prompt(text: str) -> str:
    columns = ", ".join(EXTRACTION_COLUMNS)
    body = text if len(text) <= 12000 else text[:12000]
    return (
        f"Extract these variables if explicitly stated: {columns}.\n\n"
        f"PAPER TEXT:\n{body}"
    )


class LLMExtractor:
    """Extract UAMS fields from paper text via a structured-output completion."""

    def __init__(self, complete: Completer, model: str = "gpt-4o-mini"):
        self.complete = complete
        self.model = model
        self._system = load_prompt("extraction_system")
        self._schema = build_extraction_schema()

    def extract(self, text: str, paper_id: str) -> ExtractionResult:
        raw = self.complete(self._system, _build_user_prompt(text), self._schema)
        data = json.loads(raw)
        fields = [
            ExtractedField(
                column=f["column"],
                value=f.get("value"),
                unit_as_reported=f.get("unit_as_reported"),
                source_quote=f.get("source_quote"),
                model_confidence=f.get("model_confidence"),
                status="unverified",
            )
            for f in data.get("fields", [])
            if f.get("column") in EXTRACTION_COLUMNS
        ]
        return ExtractionResult(
            paper_id=paper_id,
            crop=data.get("crop"),
            doi=data.get("doi"),
            year=data.get("year"),
            fields=fields,
            method=f"llm:{self.model}",
        )


def make_openai_completer(client, model: str = "gpt-4o-mini", temperature: float = 0.0) -> Completer:
    """Wrap an OpenAI client into a ``complete(system, user, schema)`` callable."""

    def complete(system: str, user: str, schema: dict) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "uams_extraction", "strict": True, "schema": schema},
            },
        )
        return response.choices[0].message.content

    return complete


if __name__ == "__main__":
    def fake_complete(system, user, schema):
        return json.dumps({
            "crop": "Wheat", "doi": "10.1/x", "year": 2021,
            "fields": [{"column": "Soil_pH", "value": 6.8, "unit_as_reported": None,
                        "source_quote": "Soil pH was 6.8.", "model_confidence": 0.9}],
        })

    result = LLMExtractor(fake_complete).extract("Soil pH was 6.8.", "paper1")
    assert result.crop == "Wheat" and result.fields[0].column == "Soil_pH"
    assert result.fields[0].status == "unverified"
    print("llm_extractor smoke OK ->", result.method, result.fields[0].column)
