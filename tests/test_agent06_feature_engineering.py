"""Tests for Feature Engineering Agent."""

import pandas as pd

from ades.agents.agent06_feature_engineering import FeatureEngineeringAgent


def test_gdd_calculation():
    df = pd.DataFrame({
        "Temperature_Max": [30.0, 25.0],
        "Temperature_Min": [20.0, 15.0],
    })
    agent = FeatureEngineeringAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    df_out = agent.dataframe
    assert "Growing_Degree_Days" in df_out.columns
    assert "Heat_Units" in df_out.columns


def test_nue_calculation():
    df = pd.DataFrame({
        "Nitrogen": [100, 200],
        "Yield_per_Hectare": [3000, 5000],
    })
    agent = FeatureEngineeringAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    df_out = agent.dataframe
    assert "Nitrogen_Use_Efficiency" in df_out.columns


def test_disease_risk():
    df = pd.DataFrame({
        "Humidity": [85.0, 50.0],
        "Temperature_Max": [30.0, 20.0],
    })
    agent = FeatureEngineeringAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    df_out = agent.dataframe
    assert "Disease_Risk_Index" in df_out.columns
    assert df_out["Disease_Risk_Index"].iloc[0] == 1.0
    assert df_out["Disease_Risk_Index"].iloc[1] == 0.0
