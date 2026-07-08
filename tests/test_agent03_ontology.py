"""Tests for Ontology Mapping Agent."""

import pandas as pd

from ades.agents.agent03_ontology import OntologyMappingAgent


def test_ontology_mapping():
    df = pd.DataFrame({"Crop": ["Wheat"], "Soil_pH": [6.5], "Nitrogen": [50]})
    agent = OntologyMappingAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    assert "Ontology_Mapping.csv" in [Path(a).name for a in result.artifacts]


def test_ontology_unknown_column():
    df = pd.DataFrame({"Some_Obscure_Col": [1]})
    agent = OntologyMappingAgent()
    result = agent.run(df=df)
    assert result.status == "success"


from pathlib import Path
