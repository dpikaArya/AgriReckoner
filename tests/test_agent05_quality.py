"""Tests for Quality Assurance Agent."""

import pandas as pd
import numpy as np

from ades.agents.agent05_quality_assurance import QualityAssuranceAgent


def test_quality_clean_data():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    agent = QualityAssuranceAgent()
    result = agent.run(df=df)
    assert result.status == "success"


def test_quality_duplicate_rows():
    df = pd.DataFrame({"A": [1, 2, 2], "B": [4, 5, 5]})
    agent = QualityAssuranceAgent()
    result = agent.run(df=df)
    assert result.status == "success"


def test_quality_impossible_ph():
    df = pd.DataFrame({"Soil_pH": [15.0, -1.0, 7.0]})
    agent = QualityAssuranceAgent()
    result = agent.run(df=df)
    assert result.status == "success"


def test_quality_outliers():
    np.random.seed(42)
    vals = np.random.randn(100).tolist() + [1000, -1000]
    df = pd.DataFrame({"Value": vals})
    agent = QualityAssuranceAgent()
    result = agent.run(df=df)
    assert result.status == "success"
