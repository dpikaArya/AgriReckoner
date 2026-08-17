"""Live OpenAI smoke test for the LLM extraction agent.

Run this yourself (it makes a real API call, which the sandbox blocks):

    export OPENAI_API_KEY=...            # already in your ~/.zshrc
    source .venv/bin/activate
    python scripts/smoke_llm_extract.py

It extracts from a tiny synthetic abstract, grounds the values, and prints the resulting
UAMS row plus the provenance table. No files in the repo are modified.
"""

import os
import sys

from agri_ai_agent.agents.llm_extraction_agent import LLMExtractionAgent

SAMPLE = (
    "Field trials of wheat (Triticum aestivum) were conducted in loam soil. "
    "The soil pH was 6.8 and organic carbon was 0.72%. "
    "Applied nitrogen was 120 kg/ha and phosphorus 60 kg/ha. "
    "Total seasonal rainfall was 540 mm and maximum temperature reached 34.5 C. "
    "Grain yield was 4.2 t/ha and mean plant height 92 cm."
)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; export it (it is in your ~/.zshrc) and retry.")
        return 1

    agent = LLMExtractionAgent()
    contract = agent.run(df=None, papers={"smoke_paper": SAMPLE})
    print("status:", contract.status)
    print("warnings:", contract.warnings)
    if agent.dataframe is not None and not agent.dataframe.empty:
        print("\nExtracted UAMS row:")
        print(agent.dataframe.iloc[0].dropna().to_string())
    print("\nProvenance artifact(s):", contract.artifacts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
