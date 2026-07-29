"""Honesty guard: the README's headline counts must match what the code defines.

These assertions fail the moment the documentation and the code disagree on the
numbers the README advertises, so the two cannot silently drift apart again.
"""

import importlib.metadata
from pathlib import Path

import pytest
import yaml

from agri_ai_agent.config.schema import SCHEMA_GROUPS, UAMS_COLUMNS
from agri_ai_agent.orchestrator import PIPELINE_STEPS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FERTILIZER_RULES = _REPO_ROOT / "agri_ai_agent" / "rules" / "fertilizer_rules.yaml"


def test_uams_column_count_is_296():
    assert len(UAMS_COLUMNS) == 296


def test_schema_group_count_is_26():
    assert len(SCHEMA_GROUPS) == 26


def test_schema_group_sizes_sum_to_column_count():
    assert sum(len(cols) for cols in SCHEMA_GROUPS.values()) == len(UAMS_COLUMNS)


def test_fuzzy_rule_count_is_221():
    with _FERTILIZER_RULES.open(encoding="utf-8") as handle:
        rules = yaml.safe_load(handle)
    assert len(rules["rules"]) == 221


def test_default_pipeline_wires_23_agents():
    assert len(PIPELINE_STEPS) == 23


def test_package_version_is_2_0_0():
    try:
        ver = importlib.metadata.version("agri-ai-agent")
        assert ver == "2.0.0"
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("agri-ai-agent not installed")
