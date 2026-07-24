"""Tests for the leakage guard."""

import numpy as np
import pandas as pd

from agri_ai_agent.ml.leakage import (
    is_leaky_feature,
    select_feature_columns,
    strip_engineered,
)


def test_exact_outcome_columns_are_leaky():
    assert is_leaky_feature("Yield_per_Hectare")
    assert is_leaky_feature("Predicted_Yield")
    assert is_leaky_feature("Biomass_Yield")


def test_engineered_and_duplicate_variants_are_leaky():
    assert is_leaky_feature("Yield_per_Hectare_log")
    assert is_leaky_feature("Yield_per_Hectare_1")
    assert is_leaky_feature("Yield_x_N")
    assert is_leaky_feature("Biomass_log")


def test_target_family_collinearity_is_target_relative():
    assert is_leaky_feature("Plant_Height_30_cm", "Plant_Height_cm")
    assert is_leaky_feature("Height_x_N", "Plant_Height_cm")
    # A staged height is a legitimate predictor of a different target.
    assert not is_leaky_feature("Plant_Height_30_cm", "Yield_per_Hectare")


def test_legitimate_predictors_are_not_excluded():
    for col in ["Nitrogen", "Phosphorus", "Soil_pH", "Rainfall", "Spacing_Plant", "N_x_P"]:
        assert not is_leaky_feature(col, "Yield_per_Hectare"), col


def test_strip_engineered():
    assert strip_engineered("Yield_per_Hectare_log") == "Yield_per_Hectare"
    assert strip_engineered("Soil_pH_1") == "Soil_pH"
    assert strip_engineered("Nitrogen") == "Nitrogen"


def test_select_feature_columns_drops_leaks():
    df = pd.DataFrame({
        "Nitrogen": [1.0, 2.0, 3.0], "Soil_pH": [6.5, 7.0, 6.0], "Rainfall": [100, 200, 150],
        "Yield_per_Hectare": [10, 20, 15], "Yield_x_N": [10, 40, 45], "Predicted_Yield": [9, 19, 14],
    })
    feats = select_feature_columns(df, "Yield_per_Hectare")
    assert set(feats) == {"Nitrogen", "Soil_pH", "Rainfall"}


def test_permuted_target_kills_perfect_fit():
    """A model using only safe features cannot achieve R2=1 by memorizing the target."""
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "Nitrogen": rng.normal(size=n), "Soil_pH": rng.normal(size=n),
        "Yield_per_Hectare": rng.normal(size=n),
    })
    df["Yield_x_N"] = df["Yield_per_Hectare"] * df["Nitrogen"]  # leaky
    feats = select_feature_columns(df, "Yield_per_Hectare")
    assert "Yield_x_N" not in feats
    model = LinearRegression().fit(df[feats], df["Yield_per_Hectare"])
    r2 = model.score(df[feats], df["Yield_per_Hectare"])
    assert r2 < 0.5  # noise target, safe features -> no spurious perfect fit


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
