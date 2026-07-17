"""Tests for RecommendationAgent v2 — Phase 9."""

import json
import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.recommendation_agent import (
    RecommendationAgent, CROP_FERTILIZER_PREFS, FERTILIZER_COST_PER_KG,
)


@pytest.fixture
def agent():
    return RecommendationAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize", "Tomato"],
        "Soil_pH": [6.0, 7.2, 5.8, 6.5],
        "Nitrogen": [30, 100, 80, 60],
        "Phosphorus": [10, 25, 30, 20],
        "Potassium": [80, 200, 150, 120],
        "Rainfall": [1200, 600, 500, 800],
        "Average_Temperature": [28, 22, 26, 25],
        "Yield_per_Hectare": [3000, 2500, 4000, 20000],
        "Predicted_Yield": [3500, 2800, 4500, 25000],
        "Dose": [120, 100, 150, 80],
        "N_Status": ["low", "medium", "medium", "medium"],
        "P_Status": ["low", "medium", "medium", "medium"],
        "K_Status": ["low", "medium", "medium", "medium"],
    })


class TestRecommendationAgentV2:

    def test_agent_name(self, agent):
        assert agent.agent_name == "RecommendationAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_all_columns_added(self, agent, sample_df):
        result = agent.process(sample_df)
        required_cols = [
            "Recommended_Fertilizer", "Recommended_Dose",
            "Recommended_Application_Interval", "Expected_Yield_Increase",
            "Expected_Yield_Increase_Pct", "Confidence_Score",
            "Economic_Score", "Environmental_Score", "Risk_Score",
            "Top_Alternatives", "Recommendation_Summary",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing: {col}"

    def test_fertilizer_recommendation_logic(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Recommended_Fertilizer"] == "Urea"
        assert result.loc[1, "Recommended_Fertilizer"] != ""
        assert result.loc[2, "Recommended_Fertilizer"] != ""
        assert result.loc[3, "Recommended_Fertilizer"] != ""

    def test_dose_adjusted_for_n_status(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Recommended_Dose"] > sample_df.loc[0, "Dose"]

    def test_yield_increase_computed(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            assert pd.notna(result.loc[idx, "Expected_Yield_Increase"])

    def test_confidence_score_range(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            conf = result.loc[idx, "Confidence_Score"]
            assert 0 <= conf <= 1

    def test_economic_score_computed(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            econ = result.loc[idx, "Economic_Score"]
            assert pd.notna(econ)
            assert 0 <= econ <= 1

    def test_environmental_score_computed(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            env = result.loc[idx, "Environmental_Score"]
            assert pd.notna(env)
            assert 0 <= env <= 1

    def test_risk_score_computed(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            risk = result.loc[idx, "Risk_Score"]
            assert pd.notna(risk)
            assert 0 <= risk <= 1

    def test_top_alternatives_is_json(self, agent, sample_df):
        result = agent.process(sample_df, top_n=3)
        for idx in range(4):
            alts_str = result.loc[idx, "Top_Alternatives"]
            assert isinstance(alts_str, str)
            alts = json.loads(alts_str)
            assert isinstance(alts, list)
            assert len(alts) <= 3

    def test_top_n_alternatives(self, agent, sample_df):
        result = agent.process(sample_df, top_n=2)
        for idx in range(4):
            alts = json.loads(result.loc[idx, "Top_Alternatives"])
            assert len(alts) <= 2

    def test_recommendation_summary_not_empty(self, agent, sample_df):
        result = agent.process(sample_df)
        for idx in range(4):
            summary = result.loc[idx, "Recommendation_Summary"]
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_existing_values_preserved(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Yield_per_Hectare"] == 3000

    def test_minimal_input(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert "Recommended_Fertilizer" in result.columns

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame()
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_crop_preference_lookup(self):
        assert "Urea" in CROP_FERTILIZER_PREFS["Rice"]
        assert "DAP" in CROP_FERTILIZER_PREFS["Wheat"]

    def test_fertilizer_cost_lookup(self):
        assert FERTILIZER_COST_PER_KG["Urea"] == 6.0
        assert FERTILIZER_COST_PER_KG["DAP"] == 28.0
