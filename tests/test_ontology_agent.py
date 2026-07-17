"""Tests for Agricultural Ontology Agent."""

import pandas as pd
import pytest

from agri_ai_agent.agents.ontology_agent import OntologyAgent
from agri_ai_agent.contracts.messages import AgentContract


class TestOntologyAgent:
    def setup_method(self):
        self.agent = OntologyAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "OntologyAgent"

    def test_process_empty_dataframe(self):
        df = pd.DataFrame()
        result = self.agent.process(df)
        assert result.empty

    def test_resolve_column_exact_match(self):
        canonical = self.agent._resolve_column("Yield_per_Hectare")
        assert canonical == "Yield_per_Hectare"

    def test_resolve_column_synonym(self):
        canonical = self.agent._resolve_column("yield")
        assert canonical == "Yield_per_Hectare"

    def test_resolve_column_case_insensitive(self):
        canonical = self.agent._resolve_column("YIELD_PER_HECTARE")
        assert canonical == "Yield_per_Hectare"

    def test_resolve_column_with_spaces(self):
        canonical = self.agent._resolve_column("plant height")
        assert canonical == "Plant_Height_cm"

    def test_resolve_column_with_units(self):
        canonical = self.agent._resolve_column("yield (kg/ha)")
        assert canonical == "Yield_per_Hectare"

    def test_resolve_column_unknown(self):
        canonical = self.agent._resolve_column("xyz_unknown_col")
        assert canonical is None

    def test_process_renames_columns(self):
        df = pd.DataFrame({
            "yield": [100, 200],
            "plant height": [50, 60],
            "ph": [6.5, 7.0],
        })
        result = self.agent.process(df)
        assert "Yield_per_Hectare" in result.columns
        assert "Plant_Height_cm" in result.columns
        assert "Soil_pH" in result.columns

    def test_process_preserves_data(self):
        df = pd.DataFrame({
            "yield": [3500.0, 4200.0],
            "plant height": [85.5, 92.3],
        })
        result = self.agent.process(df)
        assert result["Yield_per_Hectare"].tolist() == [3500.0, 4200.0]
        assert result["Plant_Height_cm"].tolist() == [85.5, 92.3]

    def test_process_handles_duplicate_columns(self):
        df = pd.DataFrame({
            "yield": [100, 200],
            "Yield": [300, 400],
        })
        result = self.agent.process(df)
        assert len(result.columns) == 2

    def test_resolve_value_valid(self):
        assert self.agent.resolve_value("35.5") == 35.5
        assert self.agent.resolve_value("1000") == 1000.0

    def test_resolve_value_with_symbols(self):
        assert self.agent.resolve_value("35.5†") == 35.5
        assert self.agent.resolve_value("1000 ± 50") == 1000.0

    def test_resolve_value_invalid(self):
        assert self.agent.resolve_value(None) is None
        assert self.agent.resolve_value("") is None
        assert self.agent.resolve_value("NA") is None
        assert self.agent.resolve_value("N/A") is None

    def test_resolve_value_with_commas(self):
        assert self.agent.resolve_value("1,000.5") == 1000.5

    def test_get_canonical_columns(self):
        canonicals = self.agent.get_canonical_columns()
        assert "Yield_per_Hectare" in canonicals
        assert "Plant_Height_cm" in canonicals
        assert "Soil_pH" in canonicals
        assert len(canonicals) > 50

    def test_get_synonyms(self):
        synonyms = self.agent.get_synonyms("Yield_per_Hectare")
        assert "yield" in synonyms
        assert "yield/ha" in synonyms
        assert len(synonyms) > 5

    def test_add_synonym(self):
        initial_count = len(self.agent.get_synonyms("Yield_per_Hectare"))
        self.agent.add_synonym("Yield_per_Hectare", "grain_yield_kg")
        new_synonyms = self.agent.get_synonyms("Yield_per_Hectare")
        assert len(new_synonyms) == initial_count + 1
        assert "grain_yield_kg" in new_synonyms

    def test_add_synonym_no_duplicates(self):
        self.agent.add_synonym("Yield_per_Hectare", "grain_yield_kg")
        self.agent.add_synonym("Yield_per_Hectare", "grain_yield_kg")
        synonyms = self.agent.get_synonyms("Yield_per_Hectare")
        assert synonyms.count("grain_yield_kg") == 1

    def test_run_method_returns_contract(self):
        df = pd.DataFrame({"yield": [100, 200], "height": [50, 60]})
        contract = self.agent.run(df)
        assert isinstance(contract, AgentContract)
        assert contract.status == "success"

    def test_process_mixed_known_unknown_columns(self):
        df = pd.DataFrame({
            "yield": [100],
            "custom_metric": [42],
            "plant height": [50],
        })
        result = self.agent.process(df)
        assert "Yield_per_Hectare" in result.columns
        assert "Plant_Height_cm" in result.columns
        assert "custom_metric" in result.columns

    def test_normalize_name(self):
        assert self.agent._normalize_name("Yield per Hectare") == "yield_per_hectare"
        assert self.agent._normalize_name("plant-height (cm)") == "plant_height_cm"
        assert self.agent._normalize_name("  Soil pH  ") == "soil_ph"
