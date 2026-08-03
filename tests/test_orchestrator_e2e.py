"""End-to-end tests for the agent Orchestrator.

Replaces the removed tests that targeted the superseded numbered-agent design.
These exercise the real pipeline on a small dataframe and assert it runs to
completion and produces the headline columns.

The inputs are synthetic, but the run is still marked ``live``: the pipeline's
``external_data`` step downloads from external dataset repositories on every
run, which needs the network and takes tens of minutes. Making that step
opt-out would let these run in CI, where they belong.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.orchestrator import PIPELINE_STEPS, Orchestrator

logging.disable(logging.WARNING)


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 15
    return pd.DataFrame(
        {
            "crop": ["Wheat", "Rice", "Maize"] * 5,
            "season": ["Rabi", "Kharif", "Rabi", "Kharif", "Rabi"] * 3,
            "tmax_c": rng.uniform(25, 38, n).round(1),
            "tmin_c": rng.uniform(10, 22, n).round(1),
            "rainfall_mm": rng.uniform(50, 300, n).round(0),
            "ph": rng.uniform(5.5, 8.0, n).round(2),
            "nitrogen_kg_ha": rng.uniform(40, 160, n).round(0),
            "phosphorus": rng.uniform(10, 60, n).round(0),
            "potassium": rng.uniform(20, 120, n).round(0),
            "yield_per_ha": rng.uniform(1500, 5500, n).round(0),
        }
    )


@pytest.fixture
def temp_settings(tmp_path):
    settings = AgriAISettings()
    settings.CHECKPOINT_ENABLED = False
    settings.OUTPUT_DIR = tmp_path / "outputs"
    settings.MODELS_DIR = tmp_path / "models"
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The literature phase (Phase 0) performs network sync; these e2e tests
    # already exercise the network via `external_data`, so literature is
    # disabled here (it still completes as a no-op, keeping step-count
    # assertions valid).
    settings.INCLUDE_LITERATURE_PHASE = False
    return settings


@pytest.mark.live
def test_pipeline_runs_end_to_end(synthetic_df, temp_settings):
    orch = Orchestrator(settings=temp_settings)
    result = orch.run(df=synthetic_df)

    assert orch.state.status == "completed"
    assert orch.state.failed_agents == []
    # Derived from the pipeline, not hardcoded: adding an agent should not need a test edit.
    assert len(orch.state.completed_agents) == len(PIPELINE_STEPS)
    assert len(result) == len(synthetic_df)


@pytest.mark.live
def test_pipeline_produces_headline_columns(synthetic_df, temp_settings):
    orch = Orchestrator(settings=temp_settings)
    result = orch.run(df=synthetic_df)

    for col in ["Yield_per_Hectare", "Predicted_Yield", "Crop_Code", "Growing_Degree_Days"]:
        assert col in result.columns, f"missing headline column: {col}"


@pytest.mark.live
def test_input_dataframe_is_not_dropped(synthetic_df, temp_settings):
    """Regression: Orchestrator.run(df=) used to drop the input dataframe."""
    orch = Orchestrator(settings=temp_settings)
    orch.run(df=synthetic_df)
    assert "extraction" in orch.state.completed_agents


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
