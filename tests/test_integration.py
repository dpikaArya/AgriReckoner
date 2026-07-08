"""Integration tests for ADES end-to-end pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ades.config.settings import ADESSettings
from ades.orchestrator import Orchestrator


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "tmax_c": [30.0, 25.0, 35.0],
        "tmin_c": [20.0, 15.0, 22.0],
        "rainfall_mm": [100, 200, 150],
        "ph": [6.5, 7.0, 6.0],
        "ec_ds_m": [0.5, 0.8, 0.3],
        "crop": ["Wheat", "Rice", "Wheat"],
        "season": ["Rabi", "Kharif", "Rabi"],
        "Nitrogen_kg_ha": [100, 120, 90],
        "Yield_per_Plot": [500, 600, 550],
        "Plot_Size": [25, 30, 25],
    })


def test_end_to_end_pipeline(sample_dataset):
    settings = ADESSettings()
    settings.CHECKPOINT_ENABLED = False
    orch = Orchestrator(settings=settings)
    result = orch.run(df=sample_dataset)

    assert result is not None
    assert len(result) == 3

    key_cols = [
        "Temperature_Max", "Temperature_Min", "Rainfall",
        "Soil_pH", "EC", "Crop", "Season", "Nitrogen",
        "Yield_per_Plot", "Plot_Size",
        "Crop_Code", "Season_Code",
        "Growing_Degree_Days",
        "Feature_Available_Before_Prediction",
    ]
    for col in key_cols:
        assert col in result.columns, f"Missing column: {col}"

    assert result["Temperature_Max"].iloc[0] == 30.0
    assert result["Soil_pH"].iloc[0] == 6.5


def test_pipeline_output_files(sample_dataset):
    settings = ADESSettings()
    settings.CHECKPOINT_ENABLED = False
    orch = Orchestrator(settings=settings)
    orch.run(df=sample_dataset)

    outputs = settings.OUTPUT_DIR
    expected_files = [
        "Universal_Agricultural_ML_Master.csv",
        "Variable_Mapping.csv",
        "Ontology_Mapping.csv",
        "Encoding_Map.csv",
        "Statistical_Profile.csv",
        "Correlation_Matrix.csv",
        "VIF_Report.csv",
        "Feature_Dictionary.csv",
        "Universal_Agricultural_Machine_Learning_Schema_v1.xlsx",
    ]
    for fname in expected_files:
        assert (outputs / fname).exists(), f"Missing output file: {fname}"


def test_pipeline_with_csv_input(sample_dataset):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_dataset.to_csv(f, index=False)
        csv_path = f.name

    settings = ADESSettings()
    settings.CHECKPOINT_ENABLED = False
    orch = Orchestrator(settings=settings)
    result = orch.run(filepath=csv_path)

    assert result is not None
    assert len(result) == 3
    Path(csv_path).unlink(missing_ok=True)
