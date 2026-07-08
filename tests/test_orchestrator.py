"""Tests for the Orchestrator Agent."""

import pandas as pd

from ades.config.settings import ADESSettings
from ades.orchestrator import Orchestrator


def test_orchestrator_with_dataframe():
    df = pd.DataFrame({
        "tmax_c": [30.0, 25.0],
        "rainfall_mm": [100, 200],
        "ph": [6.5, 7.0],
        "crop": ["Wheat", "Rice"],
    })
    settings = ADESSettings()
    settings.CHECKPOINT_ENABLED = False
    orch = Orchestrator(settings=settings)
    result = orch.run(df=df)
    assert result is not None
    assert len(result) == 2
    assert orch.state.status == "completed"


def test_orchestrator_agents_executed():
    df = pd.DataFrame({"tmax_c": [30.0], "rainfall_mm": [100]})
    settings = ADESSettings()
    settings.CHECKPOINT_ENABLED = False
    orch = Orchestrator(settings=settings)
    orch.run(df=df)
    expected_agents = [
        "ingestion", "schema_mapping", "ontology_mapping",
        "unit_harmonization", "quality_assurance", "feature_engineering",
        "leakage_detection", "encoding", "statistical_diagnostics",
        "model_readiness", "documentation", "export",
    ]
    for agent_key in expected_agents:
        assert agent_key in orch.state.completed_agents, f"{agent_key} not completed"
