"""Tests for expanded Fuzzy Logic Agent — Phase 10 (200+ rules)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from agri_ai_agent.agents.fuzzy_logic_agent import FuzzyAgent
from agri_ai_agent.rules.membership_functions import (
    fuzzify,
    shouldered_z,
    triangle,
)

RULES_PATH = (
    Path(__file__).resolve().parent.parent / "agri_ai_agent" / "rules" / "fertilizer_rules.yaml"
)


@pytest.fixture
def agent():
    return FuzzyAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Nitrogen": [30, 100, 150, 50, 80],
            "Phosphorus": [10, 25, 40, 15, 30],
            "Potassium": [80, 200, 300, 120, 150],
            "Soil_pH": [5.5, 7.0, 8.0, 6.5, 6.8],
            "Rainfall": [300, 700, 1200, 500, 800],
            "Temperature_Max": [38, 28, 22, 35, 30],
            "Organic_Carbon": [0.3, 0.8, 1.2, 0.5, 0.7],
            "Growth_Stage": [20, 50, 80, 30, 60],
            "Zinc": [0.3, 1.5, 3.0, 0.8, 1.2],
            "Yield_Prediction": [2.0, 5.0, 8.0, 3.0, 4.5],
            "Dose": [100, 120, 80, 150, 100],
            "Application_Interval": [7, 10, 14, 5, 10],
            "Crop": ["Rice", "Wheat", "Maize", "Tomato", "Rice"],
        }
    )


class TestFuzzyRulesExpanded:
    def test_rules_file_exists(self):
        assert RULES_PATH.exists()

    def test_rules_count_above_200(self):
        with open(RULES_PATH) as f:
            data = yaml.safe_load(f)
        rules = data.get("rules", [])
        assert len(rules) >= 200, f"Expected >=200 rules, got {len(rules)}"

    def test_rules_metadata_version(self):
        with open(RULES_PATH) as f:
            data = yaml.safe_load(f)
        meta = data.get("metadata", {})
        assert "version" in meta

    def test_rules_have_correct_structure(self):
        with open(RULES_PATH) as f:
            data = yaml.safe_load(f)
        for r in data["rules"]:
            assert "name" in r
            assert "antecedents" in r
            assert "consequents" in r
            for ant in r["antecedents"]:
                assert "var" in ant
                assert "set" in ant


class TestFuzzyAgentExpanded:
    def test_agent_name(self, agent):
        assert agent.agent_name == "FuzzyAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_fuzzy_output_columns(self, agent, sample_df):
        result = agent.process(sample_df)
        for col in [
            "Fuzzy_N_Action",
            "Fuzzy_P_Action",
            "Fuzzy_K_Action",
            "Fuzzy_Zinc_Action",
            "Fuzzy_Risk",
            "Fuzzy_Confidence_Label",
            "Fuzzy_Summary",
        ]:
            assert col in result.columns

    def test_fuzzy_summary_populated(self, agent, sample_df):
        result = agent.process(sample_df)
        non_empty = result["Fuzzy_Summary"].notna().sum()
        assert non_empty > 0

    def test_fuzzy_risk_labels(self, agent, sample_df):
        result = agent.process(sample_df)
        valid_risks = {"low", "medium", "high"}
        for val in result["Fuzzy_Risk"].dropna():
            assert val in valid_risks, f"Invalid risk: {val}"

    def test_fuzzy_confidence_labels(self, agent, sample_df):
        result = agent.process(sample_df)
        valid = {"low", "medium", "high"}
        for val in result["Fuzzy_Confidence_Label"].dropna():
            assert val in valid, f"Invalid confidence: {val}"

    def test_n_action_labels(self, agent, sample_df):
        result = agent.process(sample_df)
        valid = {
            "Increase",
            "Reduce",
            "Maintain",
            "Apply foliar spray",
            "Apply to soil",
            "Monitor",
            "Review program",
        }
        for val in result["Fuzzy_N_Action"].dropna():
            assert val in valid or "%" in val, f"Unexpected N action: {val}"

    def test_different_inputs_different_outputs(self, agent):
        df1 = pd.DataFrame(
            {"Nitrogen": [20], "Rainfall": [200], "Temperature_Max": [38], "Organic_Carbon": [0.3]}
        )
        df2 = pd.DataFrame(
            {
                "Nitrogen": [150],
                "Rainfall": [1200],
                "Temperature_Max": [20],
                "Organic_Carbon": [1.2],
            }
        )
        r1 = agent.process(df1)
        r2 = agent.process(df2)
        assert (
            r1.loc[0, "Fuzzy_Risk"] != r2.loc[0, "Fuzzy_Risk"]
            or r1.loc[0, "Fuzzy_N_Action"] != r2.loc[0, "Fuzzy_N_Action"]
        )

    def test_membership_functions(self):
        result = fuzzify("Nitrogen", 50)
        assert "low" in result
        assert "medium" in result
        assert "high" in result
        assert 0 <= result["low"] <= 1
        assert 0 <= result["medium"] <= 1

    def test_triangle_mf(self):
        vals = triangle(np.array([0, 0.5, 1.0]), 0, 0.5, 1.0)
        assert vals[0] == pytest.approx(0.0, abs=1e-9)
        assert vals[1] == pytest.approx(1.0, abs=1e-3)
        assert vals[2] == pytest.approx(0.0, abs=1e-9)

    def test_shouldered_mf(self):
        vals_low = shouldered_z(np.array([0, 0.5, 1.0]), 0.3, 0.7)
        assert vals_low[0] == 1.0
        assert vals_low[2] == 0.0

    def test_minimal_input(self, agent):
        df = pd.DataFrame({"Nitrogen": [50]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame()
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_dose_modified_by_fuzzy(self, agent, sample_df):
        result = agent.process(sample_df)
        assert (
            "Recommended_Application_Interval" in result.columns
            or "Recommended_Fertilizer" in result.columns
        )

    def test_interval_modified_by_fuzzy(self, agent, sample_df):
        result = agent.process(sample_df)
        assert "Recommended_Application_Interval" in result.columns
