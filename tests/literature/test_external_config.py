"""Tests for externalized configuration (config/*.yaml, config/api_keys.env)."""

from __future__ import annotations

import os

from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.literature.dedupe import compute_quality_score
from agri_ai_agent.literature.external_config import (
    load_env_file,
    load_literature_config,
    load_yaml,
)
from agri_ai_agent.literature.models import LiteratureRecord


def test_load_env_file_applies_and_does_not_overwrite(tmp_path):
    env = tmp_path / "api_keys.env"
    env.write_text(
        "# comment\n\n"
        "AGRI_TEST_ONE=hello\n"
        "AGRI_TEST_TWO=\"quoted\"\n"
        "AGRI_TEST_THREE='single'\n",
        encoding="utf-8",
    )
    os.environ.pop("AGRI_TEST_ONE", None)
    os.environ.pop("AGRI_TEST_TWO", None)
    os.environ.pop("AGRI_TEST_THREE", None)
    os.environ["AGRI_TEST_EXISTING"] = "keep"
    env.write_text(
        env.read_text(encoding="utf-8") + "AGRI_TEST_EXISTING=overwrite\n",
        encoding="utf-8",
    )

    loaded = load_env_file(env)
    try:
        assert loaded["AGRI_TEST_ONE"] == "hello"
        assert os.environ["AGRI_TEST_ONE"] == "hello"
        assert os.environ["AGRI_TEST_TWO"] == "quoted"
        assert os.environ["AGRI_TEST_THREE"] == "single"
        assert os.environ["AGRI_TEST_EXISTING"] == "keep"
    finally:
        for key in ("AGRI_TEST_ONE", "AGRI_TEST_TWO", "AGRI_TEST_THREE"):
            os.environ.pop(key, None)


def test_load_yaml_missing_file_is_empty(tmp_path):
    assert load_yaml(tmp_path / "nope.yaml") == {}


def test_load_literature_config_reads_repo_files():
    cfg = load_literature_config()
    assert "sources" in cfg
    assert "connector_limits" in cfg
    assert "quality_weights" in cfg
    assert "scheduler" in cfg
    assert "dashboard" in cfg
    assert cfg["sources"]["OpenAlex"] == {"enabled": True}
    assert cfg["sources"]["Web_of_Science"]["enabled"] is True
    assert cfg["quality_weights"]["doi"] == 0.15
    assert cfg["variable_to_uams"]["total_precipitation"] == "Rainfall"
    assert cfg["connector_limits"]["Crossref"]["rate_limit_per_minute"] >= 1


def test_literature_config_from_env_applies_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AGRI_LIT_DATA_DIR", str(tmp_path / "lit"))
    cfg = LiteratureConfig.from_env(root=tmp_path)
    assert cfg.config_dir is not None
    assert cfg.connector_limits["Crossref"]["rate_limit_per_minute"] == 50
    assert cfg.limit("Crossref", "rate_limit_per_minute", 30) == 50
    assert cfg.limit("Unknown_Connector", "rate_limit_per_minute", 30) == 30
    assert cfg.source_enabled("OpenAlex") is True
    assert cfg.source_enabled("NoSuchSource") is None
    assert cfg.quality_weights["doi"] == 0.15


def test_compute_quality_score_accepts_external_weights():
    record = LiteratureRecord(
        source="Crossref",
        source_id="x",
        doi="10.1000/abc123",
        title="A sufficiently long title for a field experiment paper",
        year=2021,
        journal="Field Crops Research",
    )
    base = compute_quality_score(record)
    boosted = compute_quality_score(record, weights={"doi": 0.5, "title": 0.2})
    assert 0.0 < base <= 1.0
    assert boosted > base
    assert compute_quality_score(record, weights={"unknown_key": 9.0}) == base


def test_literature_config_merges_schema_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("AGRI_LIT_DATA_DIR", str(tmp_path / "lit"))
    cfg = LiteratureConfig.from_env(root=tmp_path)
    assert cfg.variable_to_uams["2m_temperature"] == "Average_Temperature"
    assert cfg.extraction_rules["ocr_scanned_pdfs"] is True
