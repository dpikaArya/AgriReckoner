"""Tests for Dataset Ingestion Agent."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ades.agents.agent01_ingestion import DatasetIngestionAgent


def test_ingestion_from_dataframe():
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    agent = DatasetIngestionAgent()
    result = agent.run(df=df_in)
    assert result.status == "success"
    assert result.row_count == 2
    assert result.column_count == 2
    assert agent.dataframe is not None
    assert list(agent.dataframe.columns) == ["a", "b"]


def test_ingestion_from_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("x,y\n1,2\n3,4\n")
        path = f.name
    agent = DatasetIngestionAgent()
    result = agent.run(filepath=path)
    assert result.status == "success"
    assert result.row_count == 2
    Path(path).unlink(missing_ok=True)


def test_ingestion_file_not_found():
    agent = DatasetIngestionAgent()
    result = agent.run(filepath="nonexistent.csv")
    assert result.status == "failed"


def test_ingestion_missing_input():
    agent = DatasetIngestionAgent()
    result = agent.run()
    assert result.status == "failed"
