"""Tests for BaseAgent helpers."""

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent


class _Dummy(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "Dummy"

    def process(self, df, **kwargs):
        return df


def test_resolve_duplicate_columns_keeps_one_of_each():
    """Regression: the old implementation multiplied duplicate columns."""
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    out = _Dummy()._resolve_duplicate_columns(df)
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (1, 2)


def test_resolve_duplicate_columns_noop_when_unique():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = _Dummy()._resolve_duplicate_columns(df)
    assert list(out.columns) == ["a", "b"]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
