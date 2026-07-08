"""Tests for Schema Mapping Agent."""

import pandas as pd

from ades.agents.agent02_schema_mapping import SchemaMappingAgent


def test_basic_mapping():
    df = pd.DataFrame({
        "tmax_c": [30.0],
        "rainfall_mm": [100],
        "ph": [6.5],
        "custom_col": ["x"],
    })
    agent = SchemaMappingAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    df_out = agent.dataframe
    assert "Temperature_Max" in df_out.columns
    assert "Rainfall" in df_out.columns
    assert "Soil_pH" in df_out.columns
    assert df_out["Temperature_Max"].iloc[0] == 30.0


def test_plant_height_timepoints():
    df = pd.DataFrame({"plantheight_120_cm": [150], "plantheight_60_cm": [80]})
    agent = SchemaMappingAgent()
    agent.run(df=df)
    df_out = agent.dataframe
    assert "Plant_Height_cm" in df_out.columns
    assert "Plant_Height_60_cm" in df_out.columns
    assert df_out["Plant_Height_cm"].iloc[0] == 150


def test_universal_columns_added():
    df = pd.DataFrame({"tmax_c": [25.0]})
    agent = SchemaMappingAgent()
    agent.run(df=df)
    df_out = agent.dataframe
    assert "Paper_ID" in df_out.columns
    assert "Crop" in df_out.columns
