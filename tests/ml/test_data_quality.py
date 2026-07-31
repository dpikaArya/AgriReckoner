import pandas as pd
import pytest

from src.ml.data_quality.scoring import DatasetScorer, QualityTier


class TestDatasetScorer:
    @pytest.fixture
    def complete_data(self):
        return pd.DataFrame(
            {
                "Crop": ["Wheat", "Rice", "Maize"],
                "Location": ["Field_A", "Field_B", "Field_C"],
                "Country": ["India", "India", "India"],
                "Soil_pH": [6.5, 7.0, 5.5],
                "Organic_Carbon_pct": [0.8, 1.2, 0.5],
                "Nitrogen_kg_ha": [120, 150, 90],
                "Average_Temperature": [25, 28, 30],
                "Rainfall_mm": [800, 1200, 600],
                "Fertilizer_Name": ["Urea", "DAP", "MOP"],
                "Yield_per_Hectare": [4.5, 6.0, 3.2],
            }
        )

    @pytest.fixture
    def sparse_data(self):
        return pd.DataFrame(
            {
                "Crop": ["Wheat", None, None],
                "Yield_per_Hectare": [4.5, None, None],
            }
        )

    def test_high_quality_scored_high(self, complete_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        assert scores["tier"].iloc[0] == QualityTier.HIGH.value

    def test_sparse_data_gets_low_score(self, sparse_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(sparse_data)
        assert scores["tier"].iloc[1] == QualityTier.LOW.value

    def test_all_records_scored(self, complete_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        assert len(scores) == len(complete_data)

    def test_total_score_between_0_and_1(self, complete_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        assert scores["total_score"].between(0, 1).all()

    def test_quality_tiers_assigned(self, complete_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        assert all(t in ("high", "medium", "low") for t in scores["tier"])

    def test_empty_dataframe(self):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(pd.DataFrame())
        assert scores.empty

    def test_gold_standard_created(self, complete_data, tmp_path):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        high_mask = scorer.get_high_quality_mask(scores)
        assert high_mask.sum() > 0

    def test_sample_weights(self, complete_data):
        scorer = DatasetScorer()
        scores = scorer.score_dataset(complete_data)
        weights = scorer.get_weighted_sample_weights(scores)
        assert weights.min() >= 0.5
        assert weights.max() <= 1.0
