"""Numbers shown to a farmer must come from evidence, not from arithmetic on the input.

These cover three defects found in shipped output: an expected yield gain invented as a flat
fraction, an economic score that pinned at 1.000 because cost was understated a thousandfold,
and predictions emitted with no plausibility check at all.
"""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.prediction_agent import PredictionAgent
from agri_ai_agent.agents.recommendation_agent import RecommendationAgent


@pytest.fixture
def agent():
    return RecommendationAgent()


def test_no_baseline_means_no_yield_increase(agent):
    """An increase is a comparison; without a baseline there is nothing to compare to."""
    df = pd.DataFrame({"Predicted_Yield": [4000.0], "Yield_per_Hectare": [np.nan]})
    out = agent._compute_yield_increase(df)
    assert pd.isna(out.at[0, "Expected_Yield_Increase"])
    assert pd.isna(out.at[0, "Expected_Yield_Increase_Pct"])


def test_the_increase_is_never_a_fixed_fraction_of_the_prediction(agent):
    """Regression: the fallback returned exactly 10% of the model's own prediction."""
    df = pd.DataFrame({"Predicted_Yield": [4000.0, 7000.0], "Yield_per_Hectare": [np.nan, np.nan]})
    out = agent._compute_yield_increase(df)
    assert not (out["Expected_Yield_Increase"] == out["Predicted_Yield"] * 0.1).any()


def test_a_real_baseline_gives_a_real_increase(agent):
    df = pd.DataFrame({"Predicted_Yield": [4400.0], "Yield_per_Hectare": [4000.0]})
    out = agent._compute_yield_increase(df)
    assert out.at[0, "Expected_Yield_Increase"] == pytest.approx(400.0)
    assert out.at[0, "Expected_Yield_Increase_Pct"] == pytest.approx(10.0)


def test_the_percentage_is_filled_when_only_the_absolute_increase_is_supplied(agent):
    """Regression: guarding on the absolute value left the percentage blank in every row."""
    df = pd.DataFrame(
        {
            "Predicted_Yield": [np.nan],
            "Yield_per_Hectare": [4000.0],
            "Expected_Yield_Increase": [200.0],
        }
    )
    out = agent._compute_yield_increase(df)
    assert out.at[0, "Expected_Yield_Increase_Pct"] == pytest.approx(5.0)


def _economic(agent, dose, increase, fertilizer="Urea"):
    df = pd.DataFrame(
        {
            "Recommended_Fertilizer": [fertilizer],
            "Recommended_Dose": [dose],
            "Expected_Yield_Increase": [increase],
        }
    )
    return agent._compute_economic_score(df).at[0, "Economic_Score"]


def test_economic_score_discriminates_instead_of_saturating(agent):
    """Cost was divided by 1000 while revenue was not, so ROI — and the score — pinned high."""
    cheap_big_gain = _economic(agent, dose=60, increase=600)
    dear_small_gain = _economic(agent, dose=200, increase=100)
    assert dear_small_gain < cheap_big_gain
    assert dear_small_gain < 0.5


def test_economic_score_falls_as_the_dose_costs_more(agent):
    scores = [_economic(agent, dose=d, increase=100) for d in (60, 120, 200)]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 3


def test_no_expected_gain_means_no_economic_score(agent):
    """A missing return is unknown, not average."""
    assert pd.isna(_economic(agent, dose=120, increase=np.nan))


@pytest.mark.parametrize(
    ("value", "kept"),
    [(3200.0, True), (8000.0, True), (975150.969, False), (-50.0, False), (60000.0, False)],
)
def test_predictions_outside_the_declared_range_are_dropped(value, kept):
    """A yield of 975,151 kg/ha reached the shipped recommendations unchecked."""
    out = PredictionAgent()._within_agronomic_range(np.array([value]), "Yield_per_Hectare")
    assert (not np.isnan(out[0])) is kept


def test_an_unknown_column_passes_through_untouched():
    values = np.array([1.0, 2.0, 3.0])
    out = PredictionAgent()._within_agronomic_range(values, "Not_A_Registry_Column")
    assert np.array_equal(np.asarray(out, dtype=float), values)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
