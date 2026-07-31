"""Tests for ModelSelectionAgent — Phase 7."""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.model_selection_agent import (
    ModelSelectionAgent,
    _get_model_pool,
)


@pytest.fixture
def agent():
    return ModelSelectionAgent()


@pytest.fixture
def small_df():
    rng = np.random.RandomState(42)
    n = 30
    return pd.DataFrame(
        {
            "Nitrogen": rng.uniform(20, 120, n),
            "Phosphorus": rng.uniform(10, 60, n),
            "Potassium": rng.uniform(20, 100, n),
            "Soil_pH": rng.uniform(5.5, 7.5, n),
            "Rainfall": rng.uniform(200, 800, n),
            "Plant_Height_cm": rng.uniform(20, 120, n),
            "SPAD": rng.uniform(15, 55, n),
            "Yield_per_Hectare": rng.uniform(5, 25, n),
            "Target_Yield": rng.uniform(5, 25, n),
        }
    )


@pytest.fixture
def large_df():
    rng = np.random.RandomState(42)
    n = 160
    return pd.DataFrame(
        {
            "Nitrogen": rng.uniform(20, 120, n),
            "Phosphorus": rng.uniform(10, 60, n),
            "Potassium": rng.uniform(20, 100, n),
            "Soil_pH": rng.uniform(5.5, 7.5, n),
            "Rainfall": rng.uniform(200, 800, n),
            "Plant_Height_cm": rng.uniform(20, 120, n),
            "SPAD": rng.uniform(15, 55, n),
            "Biomass_Yield": rng.uniform(3, 20, n),
            "Harvest_Index": rng.uniform(0.3, 0.6, n),
            "Yield_per_Hectare": rng.uniform(5, 25, n),
            "Target_Yield": rng.uniform(5, 25, n),
        }
    )


class TestModelSelectionAgent:
    def test_agent_name(self, agent):
        assert agent.agent_name == "ModelSelectionAgent"

    def test_model_pool_small_sample(self):
        pool = _get_model_pool(20, is_classification=False)
        names = [name for name, _, _ in pool]
        assert "Ridge" in names
        assert "RandomForest" not in names

    def test_model_pool_medium_sample(self):
        pool = _get_model_pool(45, is_classification=False)
        names = [name for name, _, _ in pool]
        assert "Ridge" in names
        assert "RandomForest" in names
        assert "XGBoost" in names

    def test_model_pool_large_sample(self):
        pool = _get_model_pool(200, is_classification=False)
        names = [name for name, _, _ in pool]
        assert "Ridge" in names
        assert "RandomForest" in names
        assert "XGBoost" in names
        assert "GradientBoosting" in names
        assert "MLP" in names

    def test_model_pool_classification(self):
        pool = _get_model_pool(100, is_classification=True)
        names = [name for name, _, _ in pool]
        assert "LogisticRegression" in names
        assert "RandomForest" in names

    def test_process_small_dataframe(self, agent, small_df):
        result = agent.process(small_df, target="Target_Yield", cv_folds=3)
        assert isinstance(result, pd.DataFrame)
        assert "_model_selection_score" in result.columns

    def test_process_large_dataframe(self, agent, large_df):
        result = agent.process(large_df, target="Target_Yield", cv_folds=5)
        assert isinstance(result, pd.DataFrame)
        assert "_model_selection_score" in result.columns
        score = result["_model_selection_score"].iloc[0]
        assert isinstance(score, (int, float))

    def test_no_usable_target_returns_input(self, agent, small_df):
        from agri_ai_agent.config.schema import MEASURED_TARGETS

        stripped = small_df.drop(columns=["Target_Yield", *MEASURED_TARGETS], errors="ignore")
        result = agent.process(stripped)
        assert isinstance(result, pd.DataFrame)
        assert "_model_selection_score" not in result.columns

    def test_falls_back_to_measured_target(self, agent, small_df):
        result = agent.process(small_df.drop(columns=["Target_Yield"]))
        assert "_model_selection_score" in result.columns

    def test_optimized_vs_default(self, agent, small_df):
        r_opt = agent.process(small_df.copy(), target="Target_Yield", optimize=True, cv_folds=3)
        r_def = agent.process(small_df.copy(), target="Target_Yield", optimize=False, cv_folds=3)
        assert "_model_selection_score" in r_opt.columns
        assert "_model_selection_score" in r_def.columns

    def test_cv_folds_adapts_to_sample_size(self, agent, small_df):
        result = agent.process(small_df, target="Target_Yield", cv_folds=20)
        assert isinstance(result, pd.DataFrame)

    def test_custom_scoring(self, agent, small_df):
        result = agent.process(
            small_df,
            target="Target_Yield",
            scoring="r2",
            cv_folds=3,
        )
        assert "_model_selection_score" in result.columns

    def test_very_small_dataset_returns_input(self, agent):
        tiny = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "Target_Yield": [7, 8, 9],
            }
        )
        result = agent.process(tiny, target="Target_Yield")
        assert isinstance(result, pd.DataFrame)
        assert "_model_selection_score" not in result.columns
