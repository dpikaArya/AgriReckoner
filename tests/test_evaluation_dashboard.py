"""Tests for EvaluationDashboard — Phase 15."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.dashboard import (
    METRIC_NAMES,
    EvaluationDashboard,
)


@pytest.fixture
def dashboard():
    return EvaluationDashboard(output_dir=Path("test_dashboard"))


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Crop": ["Rice", "Wheat", "Maize"],
            "Nitrogen": [100.0, 80.0, 90.0],
            "Phosphorus": [25.0, 20.0, np.nan],
            "Potassium": [150.0, 120.0, 100.0],
            "Yield_per_Hectare": [4.5, 3.2, 5.1],
            "Fertilizer_Name": ["Urea", "DAP", "NPK"],
            "Dose": [50.0, 30.0, 40.0],
            "Application_Interval": [30, 45, 60],
            "Recommendation_Confidence": [0.85, 0.72, 0.91],
        }
    )


@pytest.fixture
def schema_columns():
    return [
        "Crop",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Yield_per_Hectare",
        "Fertilizer_Name",
        "Dose",
        "Application_Interval",
        "Soil_pH",
        "Rainfall",
        "Temperature_Max",
        "Temperature_Min",
        "Organic_Carbon",
    ]


class TestEvaluationDashboard:
    def test_initial_state(self, dashboard):
        assert dashboard.metrics == {}
        assert dashboard.details == {}

    def test_compute_extraction_accuracy_full(self, dashboard, sample_df):
        acc = dashboard.compute_extraction_accuracy(sample_df)
        assert 0.0 <= acc <= 1.0
        assert "extraction_accuracy" in dashboard.metrics

    def test_compute_extraction_accuracy_with_reference(self, dashboard, sample_df):
        ref_df = sample_df.copy()
        ref_df.loc[0, "Nitrogen"] = 999
        acc = dashboard.compute_extraction_accuracy(sample_df, ref_df)
        assert 0.0 <= acc <= 1.0

    def test_compute_schema_mapping_accuracy(self, dashboard, sample_df, schema_columns):
        acc = dashboard.compute_schema_mapping_accuracy(sample_df, schema_columns)
        assert 0.0 <= acc <= 1.0
        assert acc > 0.5
        assert "schema_mapping_accuracy" in dashboard.metrics

    def test_compute_schema_mapping_accuracy_all_mapped(self, dashboard, sample_df):
        acc = dashboard.compute_schema_mapping_accuracy(sample_df, list(sample_df.columns))
        assert acc == 1.0

    def test_compute_missing_data_rate(self, dashboard, sample_df):
        rate = dashboard.compute_missing_data_rate(sample_df)
        assert 0.0 <= rate <= 1.0
        assert rate < 0.5

    def test_compute_missing_data_rate_no_missing(self, dashboard):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        rate = dashboard.compute_missing_data_rate(df)
        assert rate == 0.0

    def test_compute_missing_data_rate_all_missing(self, dashboard):
        df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
        rate = dashboard.compute_missing_data_rate(df)
        assert rate == 1.0

    def test_compute_validation_pass_rate(self, dashboard, sample_df):
        rate = dashboard.compute_validation_pass_rate(sample_df)
        assert rate == 1.0

    def test_compute_validation_pass_rate_with_issues(self, dashboard, sample_df):
        issues = [{"row": 0, "col": "Nitrogen", "issue": "outlier"}]
        rate = dashboard.compute_validation_pass_rate(sample_df, issues)
        assert rate < 1.0

    def test_compute_feature_coverage(self, dashboard):
        orig = pd.DataFrame({"A": [1], "B": [2]})
        eng = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]})
        cov = dashboard.compute_feature_coverage(orig, eng)
        assert cov == 1.0

    def test_compute_feature_coverage_no_new(self, dashboard):
        df = pd.DataFrame({"A": [1]})
        cov = dashboard.compute_feature_coverage(df, df)
        assert cov == 0.0

    def test_compute_recommendation_confidence(self, dashboard, sample_df):
        conf = dashboard.compute_recommendation_confidence(sample_df)
        assert 0.0 <= conf <= 1.0
        assert conf > 0.7

    def test_compute_recommendation_confidence_no_column(self, dashboard):
        df = pd.DataFrame({"Crop": ["Rice"]})
        conf = dashboard.compute_recommendation_confidence(df)
        assert conf == 0.0

    def test_compute_reckoner_completeness(self, dashboard, sample_df):
        comp = dashboard.compute_reckoner_completeness(sample_df)
        assert 0.0 <= comp <= 1.0
        assert comp > 0.8

    def test_compute_pipeline_success_rate(self, dashboard):
        results = [
            {"status": "success"},
            {"status": "success"},
            {"status": "failed"},
        ]
        rate = dashboard.compute_pipeline_success_rate(results)
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_compute_pipeline_success_rate_all_success(self, dashboard):
        results = [{"status": "success"}, {"status": "success"}]
        rate = dashboard.compute_pipeline_success_rate(results)
        assert rate == 1.0

    def test_compute_all(self, dashboard, sample_df, schema_columns):
        metrics = dashboard.compute_all(sample_df, schema_columns)
        assert len(metrics) == 8
        for name in METRIC_NAMES:
            assert name in metrics

    def test_generate_html(self, dashboard, sample_df, schema_columns):
        dashboard.compute_all(sample_df, schema_columns)
        html = dashboard.generate_html()
        assert "<html>" in html
        assert "Dashboard" in html
        assert "extraction" in html.lower()

    def test_save_dashboard(self, dashboard, sample_df, schema_columns, tmp_path):
        dashboard.output_dir = tmp_path
        dashboard.compute_all(sample_df, schema_columns)
        path = dashboard.save_dashboard()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html>" in content

    def test_save_metrics_json(self, dashboard, sample_df, schema_columns, tmp_path):
        dashboard.output_dir = tmp_path
        dashboard.compute_all(sample_df, schema_columns)
        path = dashboard.save_metrics_json()
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "metrics" in data
        assert "overall_score" in data
        assert "timestamp" in data

    def test_summary_text(self, dashboard, sample_df, schema_columns):
        dashboard.compute_all(sample_df, schema_columns)
        text = dashboard.summary_text()
        assert "Extraction Accuracy" in text
        assert "Overall Score" in text

    def test_overall_score(self, dashboard, sample_df, schema_columns):
        dashboard.compute_all(sample_df, schema_columns)
        score = dashboard._overall_score()
        assert 0.0 <= score <= 1.0

    def test_empty_dataframe(self, dashboard):
        df = pd.DataFrame()
        metrics = dashboard.compute_all(df, ["A", "B"])
        assert len(metrics) == 8

    def test_metric_names_count(self):
        assert len(METRIC_NAMES) == 8

    def test_details_populated(self, dashboard, sample_df, schema_columns):
        dashboard.compute_all(sample_df, schema_columns)
        for name in METRIC_NAMES:
            assert name in dashboard.details
            assert "description" in dashboard.details[name]
