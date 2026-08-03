"""Tests for the orchestrator LiteratureAgent (Phase 0)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from agri_ai_agent.agents.literature_agent import LiteratureAgent
from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.orchestrator import PIPELINE_STEPS


def _fake_config(tmp_path) -> LiteratureConfig:
    cfg = LiteratureConfig(
        data_dir=tmp_path / "lit",
        output_dir=tmp_path / "out",
        state_db=tmp_path / "lit" / "state.sqlite",
        bibliography_db=tmp_path / "lit" / "bibliography.db",
        cache_dir=tmp_path / "lit" / "cache",
        search_terms=("field experiment AND yield",),
    )
    cfg.ensure_dirs()
    return cfg


class _StubPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, **kwargs):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"Paper_ID": "p1", "DOI": "10.1/x", "Journal": "FCR", "Year": 2021}]
        ).to_csv(self.config.output_dir / "universal_schema.csv", index=False)
        return SimpleNamespace(
            run_id="RUN-STUB",
            total_papers=1,
            verified_papers=1,
            duplicates=0,
            validated_dois=1,
            new_records=1,
            agri_records=0,
            failed_pdfs=0,
            outputs={"x": Path("x")},
            errors=[],
            retrain_triggered=False,
            summary=lambda: "stub complete",
        )


@pytest.fixture
def patch_literature(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agri_ai_agent.literature.pipeline.LiteraturePipeline", _StubPipeline
    )
    monkeypatch.setattr(
        "agri_ai_agent.literature.config.LiteratureConfig.from_env",
        classmethod(lambda cls, **kw: _fake_config(tmp_path)),
    )
    return None


def test_literature_step_is_first_pipeline_step():
    assert PIPELINE_STEPS[0][0] == "literature"
    assert PIPELINE_STEPS[0][1] is LiteratureAgent


def test_literature_agent_disabled_passes_through(settings):
    settings.INCLUDE_LITERATURE_PHASE = False
    agent = LiteratureAgent(settings=settings)
    df = pd.DataFrame({"crop": ["Wheat"], "yield_per_ha": [3000.0]})
    contract = agent.run(df=df)
    assert contract.status == "success"
    assert agent.dataframe is not None
    pd.testing.assert_frame_equal(agent.dataframe.reset_index(drop=True), df.reset_index(drop=True))


def test_literature_agent_merges_uams_rows(settings, patch_literature):
    settings.INCLUDE_LITERATURE_PHASE = True
    agent = LiteratureAgent(settings=settings)
    df = pd.DataFrame({"crop": ["Wheat"], "yield_per_ha": [3000.0]})
    contract = agent.run(df=df)
    assert contract.status == "success"
    assert len(agent.dataframe) == 2
    assert agent.contract.output_data["papers"] == 1
    assert agent.contract.output_data["verified_original_papers"] == 1
    assert "literature_run_id" in agent.contract.output_data


def test_literature_agent_returns_uams_when_no_input(settings, patch_literature):
    settings.INCLUDE_LITERATURE_PHASE = True
    agent = LiteratureAgent(settings=settings)
    contract = agent.run(df=None)
    assert contract.status == "success"
    assert agent.dataframe is not None
    assert len(agent.dataframe) == 1
    assert agent.dataframe.iloc[0]["DOI"] == "10.1/x"
