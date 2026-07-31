"""The training agent must hold out whole papers, not individual rows."""

import numpy as np
import pandas as pd

from agri_ai_agent.agents.training_agent import GROUP_COLS, TrainingAgent


def _frame(n_papers=6, per_paper=4, group_col="Paper_ID"):
    rng = np.random.default_rng(0)
    n = n_papers * per_paper
    papers = np.repeat([f"PAPER/{i}" for i in range(n_papers)], per_paper)
    return pd.DataFrame(
        {
            group_col: papers,
            "Nitrogen": rng.normal(size=n),
            "Soil_pH": rng.normal(size=n),
            "Rainfall": rng.normal(size=n),
            "Yield_per_Hectare": rng.normal(size=n),
        }
    )


def test_paper_id_becomes_the_grouping_key():
    agent = TrainingAgent()
    df = _frame()
    keep = pd.Series(True, index=df.index)
    groups = agent._paper_groups(df, keep)
    assert groups is not None
    assert len(set(groups)) == 6


def test_falls_back_through_candidate_identifier_columns():
    agent = TrainingAgent()
    df = _frame(group_col="DOI")
    keep = pd.Series(True, index=df.index)
    groups = agent._paper_groups(df, keep)
    assert groups is not None
    assert len(set(groups)) == 6
    assert "DOI" in GROUP_COLS


def test_no_identifier_returns_none_rather_than_a_fake_grouping():
    agent = TrainingAgent()
    df = _frame().drop(columns=["Paper_ID"])
    keep = pd.Series(True, index=df.index)
    assert agent._paper_groups(df, keep) is None


def test_single_paper_is_not_used_as_a_grouping():
    """One paper cannot support a held-out-trial estimate, so grouping is declined."""
    agent = TrainingAgent()
    df = _frame(n_papers=1, per_paper=20)
    keep = pd.Series(True, index=df.index)
    assert agent._paper_groups(df, keep) is None


def test_groups_align_with_the_rows_actually_kept():
    """Groups must be filtered by the same mask as X and y, or folds misalign."""
    agent = TrainingAgent()
    df = _frame()
    keep = df.index % 2 == 0
    keep = pd.Series(keep, index=df.index)
    groups = agent._paper_groups(df, keep)
    assert len(groups) == int(keep.sum())


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
