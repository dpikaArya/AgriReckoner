"""Shared pytest fixtures and marker registration for the AgriAI test suite.

Additive only: named fixtures plus marker registration. No autouse fixtures
and no collection hooks that would alter the existing test suite's behaviour.
"""

import pandas as pd
import pytest

from agri_ai_agent.config.settings import AgriAISettings

SAMPLE_ROW_COUNT = 6


def pytest_configure(config):
    """Register custom markers so ``-m`` selection has no unknown-marker warnings."""
    markers = [
        "unit: fast, isolated unit test",
        "integration: crosses module or component boundaries",
        "live: requires a live external service (e.g. an LLM API)",
        "slow: takes a long time to run",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture
def settings(tmp_path):
    """Return an AgriAISettings with checkpoints off and outputs in a tmp dir."""
    instance = AgriAISettings()
    instance.CHECKPOINT_ENABLED = False
    instance.OUTPUT_DIR = tmp_path
    return instance


@pytest.fixture
def sample_uams_df():
    """Return a small UAMS-shaped DataFrame for exercising table logic."""
    return pd.DataFrame(
        {
            "Crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize"],
            "Soil_pH": [6.5, 7.2, 6.0, 6.8, 7.0, 5.9],
            "Rainfall": [1200, 650, 800, 1100, 700, 850],
            "Nitrogen": [90, 120, 100, 95, 110, 105],
            "Yield_per_Hectare": [4.2, 3.1, 5.0, 4.5, 3.3, 5.2],
        }
    )


class _StubLLM:
    """Minimal LLM stand-in returning a fixed extraction for future LLM tests."""

    def extract(self, text: str) -> dict:
        """Return a fixed extraction dict regardless of input text."""
        return {"crop": "Rice", "yield": 4.2, "source_len": len(text)}


@pytest.fixture
def mock_llm():
    """Return a tiny stub LLM object exposing ``.extract(text) -> dict``."""
    return _StubLLM()
