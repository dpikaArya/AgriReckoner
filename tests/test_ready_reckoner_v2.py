"""Tests for ReadyReckonerAgent v2 — Phase 11."""

import json

import pandas as pd
import pytest

from agri_ai_agent.agents.ready_reckoner_agent import RECOMMENDATION_COLS, ReadyReckonerAgent


@pytest.fixture
def agent():
    return ReadyReckonerAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Crop": ["Rice", "Wheat", "Maize"],
            "Variety": ["IR64", "HD2967", "PMH1"],
            "Location": ["Hyderabad", "Ludhiana", "Pune"],
            "Country": ["India", "India", "India"],
            "Soil_pH": [6.0, 7.0, 5.8],
            "Nitrogen": [30, 100, 80],
            "Phosphorus": [10, 25, 30],
            "Potassium": [80, 200, 150],
            "Rainfall": [1200, 600, 500],
            "Average_Temperature": [28, 22, 26],
            "Humidity": [75, 60, 55],
            "Predicted_Yield": [3500, 2800, 4500],
            "Expected_Biomass": [6000, 5000, 8000],
            "Expected_Plant_Height": [90, 80, 100],
            "Recommended_Fertilizer": ["Urea", "DAP", "15-15-15"],
            "Recommended_Dose": [150, 120, 100],
            "Recommended_Application_Interval": [10, 14, 12],
            "Expected_Yield_Increase": [500, 300, 800],
            "Expected_Yield_Increase_Pct": [16.7, 12.0, 21.6],
            "Confidence_Score": [0.85, 0.78, 0.82],
            "Economic_Score": [0.75, 0.65, 0.70],
            "Environmental_Score": [0.5, 0.6, 0.4],
            "Risk_Score": [0.25, 0.35, 0.40],
            "Top_Alternatives": ['[{"fertilizer":"DAP","score":0.6}]', "[]", "[]"],
            "Fuzzy_N_Action": ["Increase by 20%", "Maintain", "Increase by 15%"],
            "Fuzzy_P_Action": ["Maintain", "Maintain", "Maintain"],
            "Fuzzy_K_Action": ["Maintain", "Maintain", "Maintain"],
            "Fuzzy_Risk": ["low", "medium", "medium"],
            "Recommendation_Summary": [
                "Apply Urea 150 kg/ha",
                "Apply DAP 120 kg/ha",
                "Apply 15-15-15 100 kg/ha",
            ],
        }
    )


class TestReadyReckonerAgentV2:
    def test_agent_name(self, agent):
        assert agent.agent_name == "ReadyReckonerAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_csv_export(self, agent, sample_df):
        agent.process(sample_df)

        csv_path = agent.settings.RECKONER_DIR / "ready_reckoner.csv"
        assert csv_path.exists()

    def test_json_export(self, agent, sample_df):
        agent.process(sample_df)
        json_path = agent.settings.RECKONER_DIR / "ready_reckoner.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == len(sample_df)

    def test_markdown_export(self, agent, sample_df):
        agent.process(sample_df)
        md_path = agent.settings.RECKONER_DIR / "ready_reckoner.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Ready Reckoner" in content

    def test_html_export(self, agent, sample_df):
        agent.process(sample_df)
        html_path = agent.settings.RECKONER_DIR / "ready_reckoner.html"
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "Ready Reckoner v2" in content

    def test_dashboard_export(self, agent, sample_df):
        agent.process(sample_df)
        dash_path = agent.settings.RECKONER_DIR / "dashboard.html"
        assert dash_path.exists()

    def test_report_export(self, agent, sample_df):
        agent.process(sample_df)
        report_path = agent.settings.RECKONER_DIR / "report.html"
        assert report_path.exists()

    def test_summary_stats_export(self, agent, sample_df):
        agent.process(sample_df)
        stats_path = agent.settings.RECKONER_DIR / "summary_stats.json"
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        assert "total_scenarios" in stats
        assert stats["total_scenarios"] == len(sample_df)
        assert "Confidence_Score" in stats
        assert "Risk_Score" in stats

    def test_reckoner_data_filters_columns(self, agent, sample_df):
        data = agent._reckoner_data(sample_df)
        assert "Crop" in data.columns
        assert "Recommended_Fertilizer" in data.columns

    def test_minimal_dataframe(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"], "Nitrogen": [50]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame()
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_existing_values_preserved(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Crop"] == "Rice"
        assert result.loc[0, "Nitrogen"] == 30

    def test_recommendation_cols_define_output(self):
        assert "Recommended_Fertilizer" in RECOMMENDATION_COLS
        assert "Risk_Score" in RECOMMENDATION_COLS
        assert "Economic_Score" in RECOMMENDATION_COLS
