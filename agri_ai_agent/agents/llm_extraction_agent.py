"""LLM extraction agent: paper text -> grounded UAMS rows with per-value provenance.

This is an OPTIONAL, non-load-bearing path. It only runs when an OpenAI key (or an injected
extractor) is available; otherwise it degrades to a no-op so the framework still runs offline
and in CI. Only grounded values reach the schema row; rejected values are kept, flagged, in a
provenance artifact for human review — never silently dropped.
"""


import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.extractors.base import Extractor
from agri_ai_agent.extractors.grounding import ground_all


class LLMExtractionAgent(BaseAgent):
    def __init__(self, settings=None, extractor: Extractor | None = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self._extractor = extractor

    @property
    def agent_name(self) -> str:
        return "LLMExtractionAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract UAMS rows from ``papers`` (dict of paper_id -> text)."""
        papers = kwargs.get("papers") or {}
        extractor = self._resolve_extractor()
        if extractor is None:
            self.contract.warnings.append("No OpenAI key/extractor; LLM extraction skipped (offline).")
            self.log.warning("LLM extraction skipped: no OpenAI key or injected extractor")
            return df if df is not None else pd.DataFrame()

        rows, provenance = [], []
        for paper_id, text in papers.items():
            result = extractor.extract(text, paper_id)
            grounded = ground_all(result.fields)
            row = {"Paper_ID": paper_id, "Crop": result.crop, "DOI": result.doi, "Year": result.year}
            for item in grounded:
                if item.status == "reported":
                    row[item.column] = item.value_canonical
                provenance.append({
                    "Paper_ID": paper_id,
                    "column": item.column,
                    "value_reported": item.value,
                    "unit_as_reported": item.unit_as_reported,
                    "value_canonical": item.value_canonical,
                    "canonical_unit": item.canonical_unit,
                    "source_quote": item.source_quote,
                    "model_confidence": item.model_confidence,
                    "status": item.status,
                    "reject_reason": item.reject_reason,
                    "method": result.method,
                })
            rows.append(row)

        out = pd.DataFrame(rows)
        if provenance:
            self.save_artifact(pd.DataFrame(provenance), "LLM_Extraction_Provenance.csv")
        accepted = sum(1 for p in provenance if p["status"] == "reported")
        self.log.info("LLM extraction: %d papers, %d grounded values, %d flagged",
                      len(papers), accepted, len(provenance) - accepted)
        self.dataframe = out
        return out

    def _resolve_extractor(self) -> Extractor | None:
        if self._extractor is not None:
            return self._extractor
        key = getattr(self.settings, "OPENAI_API_KEY", "")
        model = getattr(self.settings, "LLM_MODEL", "gpt-4o-mini")
        temperature = getattr(self.settings, "LLM_TEMPERATURE", 0.0)
        if not key:
            return None
        from openai import OpenAI

        from agri_ai_agent.extractors.llm_extractor import LLMExtractor, make_openai_completer
        completer = make_openai_completer(OpenAI(api_key=key), model, temperature)
        return LLMExtractor(completer, model)


if __name__ == "__main__":
    import json

    from agri_ai_agent.extractors.llm_extractor import LLMExtractor

    def fake(system, user, schema):
        return json.dumps({"crop": "Wheat", "doi": None, "year": 2020, "fields": [
            {"column": "Soil_pH", "value": 6.8, "unit_as_reported": None,
             "source_quote": "Soil pH was 6.8.", "model_confidence": 0.9},
            {"column": "Nitrogen", "value": 9999, "unit_as_reported": "kg/ha",
             "source_quote": "N was 9999 kg/ha.", "model_confidence": 0.3},  # out of range -> rejected
        ]})

    agent = LLMExtractionAgent(extractor=LLMExtractor(fake))
    contract = agent.run(df=None, papers={"p1": "Soil pH was 6.8. N was 9999 kg/ha."})
    out = agent.dataframe
    assert out.loc[0, "Soil_pH"] == 6.8
    assert "Nitrogen" not in out.columns  # rejected value did not enter the row
    print("llm_extraction_agent smoke OK ->", contract.status, dict(out.iloc[0]))
