"""Tests for Encoding Agent."""

import pandas as pd

from ades.agents.agent08_encoding import EncodingAgent


def test_categorical_encoding():
    df = pd.DataFrame({
        "Crop": ["Wheat", "Rice", "Wheat"],
        "Season": ["Kharif", "Rabi", "Kharif"],
    })
    agent = EncodingAgent()
    result = agent.run(df=df)
    assert result.status == "success"
    df_out = agent.dataframe
    assert "Crop_Code" in df_out.columns
    assert "Season_Code" in df_out.columns
    assert df_out["Crop_Code"].iloc[0] == df_out["Crop_Code"].iloc[2]
    assert df_out["Crop_Code"].iloc[0] != df_out["Crop_Code"].iloc[1]


def test_no_categorical_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0]})
    agent = EncodingAgent()
    result = agent.run(df=df)
    assert result.status == "success"
