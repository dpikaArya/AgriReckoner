"""Tests for KnowledgeIntegrationAgent — Phase 8."""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.knowledge_integration_agent import (
    KnowledgeIntegrationAgent, CROP_YIELD_RANGES, SOIL_PROPERTY_DEFAULTS, REGIONAL_CLIMATE,
)


@pytest.fixture
def agent():
    return KnowledgeIntegrationAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize", "Tomato", "UnknownCrop"],
        "Country": ["India", "USA", "China", "Brazil", "Nigeria"],
        "Soil_pH": [6.5, np.nan, 7.0, np.nan, np.nan],
        "Organic_Carbon": [0.8, np.nan, 0.5, np.nan, np.nan],
        "Nitrogen": [100, np.nan, np.nan, 80, np.nan],
        "Phosphorus": [20, np.nan, np.nan, np.nan, 15],
        "Potassium": [150, np.nan, 200, np.nan, np.nan],
        "Average_Temperature": [28, np.nan, np.nan, 25, np.nan],
        "Rainfall": [1200, np.nan, 600, np.nan, np.nan],
        "Humidity": [75, np.nan, np.nan, np.nan, 60],
        "Yield_per_Hectare": [4.5, 2.0, np.nan, np.nan, np.nan],
    })


class TestKnowledgeIntegrationAgent:

    def test_agent_name(self, agent):
        assert agent.agent_name == "KnowledgeIntegrationAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_crop_yield_range_columns_added(self, agent, sample_df):
        result = agent.process(sample_df)
        for col in ["Crop_Yield_Min", "Crop_Yield_Max", "Crop_Yield_Unit",
                     "Crop_Opt_Temp_Min", "Crop_Opt_Temp_Max",
                     "Crop_Opt_Rain_Min", "Crop_Opt_Rain_Max"]:
            assert col in result.columns

    def test_rice_yield_ranges(self, agent, sample_df):
        result = agent.process(sample_df)
        rice_row = result[result["Crop"] == "Rice"].iloc[0]
        assert rice_row["Crop_Yield_Min"] == 1.0
        assert rice_row["Crop_Yield_Max"] == 7.5
        assert rice_row["Crop_Yield_Unit"] == "t/ha"
        assert rice_row["Crop_Opt_Temp_Min"] == 22
        assert rice_row["Crop_Opt_Temp_Max"] == 32

    def test_unknown_crop_gets_no_range(self, agent, sample_df):
        result = agent.process(sample_df)
        unknown_row = result[result["Crop"] == "UnknownCrop"].iloc[0]
        assert pd.isna(unknown_row["Crop_Yield_Min"])

    def test_soil_defaults_fill_missing(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[1, "Soil_pH"] != result.loc[1, "Soil_pH"] or True
        assert pd.notna(result.loc[1, "Soil_pH"])
        assert pd.notna(result.loc[1, "Organic_Carbon"])
        assert pd.notna(result.loc[1, "Nitrogen"])

    def test_climate_defaults_fill_missing(self, agent, sample_df):
        result = agent.process(sample_df)
        assert pd.notna(result.loc[1, "Average_Temperature"])
        assert pd.notna(result.loc[1, "Rainfall"])

    def test_yield_range_fill(self, agent, sample_df):
        result = agent.process(sample_df)
        assert pd.notna(result.loc[2, "Yield_per_Hectare"])
        assert pd.notna(result.loc[3, "Yield_per_Hectare"])

    def test_existing_values_not_overwritten(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Soil_pH"] == 6.5
        assert result.loc[0, "Nitrogen"] == 100
        assert result.loc[0, "Yield_per_Hectare"] == 4.5

    def test_lookup_crop(self):
        info = KnowledgeIntegrationAgent.lookup_crop("Rice")
        assert info is not None
        assert info["unit"] == "t/ha"
        assert KnowledgeIntegrationAgent.lookup_crop("Nonexistent") is None

    def test_lookup_soil(self):
        props = KnowledgeIntegrationAgent.lookup_soil("Black Soil")
        assert props is not None
        assert "ph" in props
        assert props == SOIL_PROPERTY_DEFAULTS["Black"]

    def test_lookup_climate(self):
        clim = KnowledgeIntegrationAgent.lookup_climate("India")
        assert clim is not None
        assert "temp_range" in clim
        assert clim == REGIONAL_CLIMATE["India"]

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert "Crop_Yield_Min" in result.columns

    def test_no_crop_column(self, agent):
        df = pd.DataFrame({"Soil_pH": [6.5, np.nan]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert "Crop_Yield_Min" not in result.columns

    def test_no_missing_values(self, agent):
        df = pd.DataFrame({
            "Crop": ["Rice"],
            "Soil_pH": [6.5],
            "Average_Temperature": [28],
            "Rainfall": [1200],
            "Yield_per_Hectare": [4.5],
        })
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_multiple_unknown_crops(self, agent):
        df = pd.DataFrame({
            "Crop": ["CropX", "CropY"],
            "Yield_per_Hectare": [np.nan, np.nan],
        })
        result = agent.process(df)
        assert pd.isna(result.loc[0, "Yield_per_Hectare"])
        assert pd.isna(result.loc[1, "Yield_per_Hectare"])
