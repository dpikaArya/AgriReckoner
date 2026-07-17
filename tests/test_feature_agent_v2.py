"""Tests for Feature Agent v2 - 150+ features."""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.feature_agent import FeatureAgent


class TestFeatureAgentV2:
    def setup_method(self):
        self.agent = FeatureAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "FeatureAgent"

    def test_generates_150_plus_features(self):
        df = pd.DataFrame({
            "Temperature_Max": np.random.uniform(20, 40, 20),
            "Temperature_Min": np.random.uniform(10, 25, 20),
            "Rainfall": np.random.uniform(100, 800, 20),
            "Nitrogen": np.random.uniform(40, 150, 20),
            "Phosphorus": np.random.uniform(20, 80, 20),
            "Potassium": np.random.uniform(30, 100, 20),
            "Soil_pH": np.random.uniform(5.5, 7.5, 20),
            "EC": np.random.uniform(0.1, 2.0, 20),
            "Organic_Carbon": np.random.uniform(0.3, 1.5, 20),
            "Humidity": np.random.uniform(40, 90, 20),
            "Plant_Height_cm": np.random.uniform(30, 150, 20),
            "Yield_per_Hectare": np.random.uniform(1000, 6000, 20),
            "Biomass_Yield": np.random.uniform(2000, 10000, 20),
            "SPAD": np.random.uniform(20, 55, 20),
            "Shoot_Biomass_g": np.random.uniform(10, 50, 20),
            "Root_Biomass_g": np.random.uniform(3, 15, 20),
            "Leaf_Area_cm2": np.random.uniform(10, 80, 20),
            "Fruit_Weight": np.random.uniform(5, 200, 20),
            "100_Seed_Weight": np.random.uniform(5, 50, 20),
            "Harvest_Index": np.random.uniform(0.2, 0.55, 20),
        })
        initial_cols = len(df.columns)
        result = self.agent.process(df)
        final_cols = len(result.columns)
        new_features = final_cols - initial_cols
        print(f"Features generated: {new_features} (from {initial_cols} to {final_cols})")
        assert final_cols >= 150, f"Expected >=150 columns, got {final_cols}"

    def test_process_minimal_dataframe(self):
        df = pd.DataFrame({"Soil_pH": [6.5, 7.0]})
        result = self.agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_preserves_original_columns(self):
        df = pd.DataFrame({
            "Plant_Height_cm": [50, 100],
            "Yield_per_Hectare": [3000, 4000],
        })
        result = self.agent.process(df)
        assert "Plant_Height_cm" in result.columns
        assert "Yield_per_Hectare" in result.columns

    def test_interaction_features_created(self):
        df = pd.DataFrame({
            "Nitrogen": [100, 150],
            "Phosphorus": [40, 60],
            "Potassium": [50, 70],
        })
        result = self.agent.process(df)
        assert "N_x_P" in result.columns
        assert "NPK_sum" in result.columns
        assert "NPK_ratio_N" in result.columns

    def test_log_features_created(self):
        df = pd.DataFrame({
            "Yield_per_Hectare": [3000, 4000, 5000],
            "Rainfall": [200, 400, 600],
            "Nitrogen": [80, 100, 120],
        })
        result = self.agent.process(df)
        assert "Yield_per_Hectare_log" in result.columns
        assert "Rainfall_log" in result.columns
        assert "N_log" in result.columns

    def test_ratio_features_created(self):
        df = pd.DataFrame({
            "Nitrogen": [100, 150],
            "Phosphorus": [40, 60],
            "Organic_Carbon": [0.5, 1.0],
            "Soil_pH": [6.5, 7.0],
        })
        result = self.agent.process(df)
        assert "N_P_ratio" in result.columns
        assert "C_N_ratio" in result.columns

    def test_polynomial_features_created(self):
        df = pd.DataFrame({
            "Soil_pH": [6.5, 7.0],
            "Rainfall": [200, 400],
        })
        result = self.agent.process(df)
        assert "Soil_pH_squared" in result.columns
        assert "Rainfall_squared" in result.columns

    def test_ratio_features_created_v2(self):
        df = pd.DataFrame({
            "Phosphorus": [40, 60],
            "Potassium": [50, 70],
        })
        result = self.agent.process(df)
        assert "P_K_ratio" in result.columns
