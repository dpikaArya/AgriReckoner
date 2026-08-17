"""The normalization step must not change how many observations exist.

Regression: with no external packages the agent wrapped the input dataframe into a
package and then merged that package *with the input it was built from*, so every row
was concatenated with a copy of itself. Silent row duplication is doubly damaging here
— it inflates the apparent sample size, and an exact duplicate of a test row sitting in
the training fold defeats cross-validation entirely.
"""

import pandas as pd
import pytest

from agri_ai_agent.agents.dataset_normalization_agent import DatasetNormalizationAgent
from agri_ai_agent.external_data.dataset_package import DatasetPackage


@pytest.fixture
def raw_df():
    return pd.DataFrame({"crop": ["Wheat", "Rice", "Maize"], "yield_per_ha": [3000, 4000, 5000]})


def test_row_count_is_unchanged_without_external_packages(raw_df):
    result = DatasetNormalizationAgent().process(raw_df.copy())
    assert len(result) == len(raw_df)


def test_no_duplicate_observations_are_introduced(raw_df):
    result = DatasetNormalizationAgent().process(raw_df.copy())
    assert result.duplicated(subset=["crop", "yield_per_ha"]).sum() == 0


def test_the_original_values_survive_normalization(raw_df):
    result = DatasetNormalizationAgent().process(raw_df.copy())
    assert sorted(result["crop"]) == sorted(raw_df["crop"])
    assert sorted(result["yield_per_ha"]) == sorted(raw_df["yield_per_ha"])


def test_provenance_columns_are_attached(raw_df):
    result = DatasetNormalizationAgent().process(raw_df.copy())
    for col in ["dataset_id", "provider", "document_type", "normalized_at"]:
        assert col in result.columns


def test_external_package_rows_are_appended_to_the_input(raw_df):
    """The merge must still combine genuinely different sources."""
    package = DatasetPackage(
        dataset_id="ext1",
        provider="TEST",
        tables=[pd.DataFrame({"crop": ["Barley"], "yield_per_ha": [2500]})],
    )
    result = DatasetNormalizationAgent().process(raw_df.copy(), packages=[package])
    assert len(result) == len(raw_df) + 1
    assert "Barley" in set(result["crop"])


def test_empty_input_stays_empty():
    result = DatasetNormalizationAgent().process(pd.DataFrame())
    assert result.empty


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
