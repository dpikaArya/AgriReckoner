"""Fixtures shared by the literature module test suite."""

from __future__ import annotations

import pytest

from agri_ai_agent.literature.config import LiteratureConfig


@pytest.fixture
def config_for_tests(tmp_path, monkeypatch):
    """A real LiteratureConfig backed by the externalized config files."""
    monkeypatch.setenv("AGRI_LIT_DATA_DIR", str(tmp_path / "literature_data"))
    monkeypatch.setenv("AGRI_LIT_OUTPUT_DIR", str(tmp_path / "outputs"))
    return LiteratureConfig.from_env(root=tmp_path)
