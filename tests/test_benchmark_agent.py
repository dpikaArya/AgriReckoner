"""Tests for BenchmarkAgent — AAIF v2.0."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.benchmark_agent import (
    BenchmarkAgent,
    EXTRACTION_METRICS,
    TRAINING_METRICS,
    RECOMMENDATION_METRICS,
    RECKONER_METRICS,
    OVERALL_METRICS,
)
from agri_ai_agent.contracts.messages import AgentContract


@pytest.fixture
def agent():
    return BenchmarkAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize"],
        "Nitrogen": [100.0, 80.0, 90.0],
        "Phosphorus": [25.0, 20.0, 15.0],
        "Potassium": [150.0, 120.0, 100.0],
        "Soil_pH": [6.5, 7.0, 6.8],
        "Rainfall": [1200, 800, 900],
        "Yield_per_Hectare": [4.5, 3.2, 5.1],
        "Target_Yield": [4.8, 3.5, 5.3],
        "Predicted_Yield": [4.6, 3.3, 5.0],
        "Recommended_Fertilizer": ["Urea", "DAP", "NPK"],
        "Recommended_Dose": [50.0, 30.0, 40.0],
        "Confidence_Score": [0.85, 0.72, 0.91],
        "Source_Paper": ["Paper_A", "Paper_B", "Paper_C"],
    })


def _get_report(agent_obj):
    return agent_obj.contract.output_data.get("benchmark", {})


class TestBenchmarkAgent:

    def test_agent_name(self, agent):
        assert agent.agent_name == "BenchmarkAgent"

    def test_run_returns_contract(self, agent, sample_df):
        contract = agent.run(sample_df)
        assert contract.status == "success"

    def test_extraction_metrics_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        ext = report.get("extraction", {})
        for m in EXTRACTION_METRICS:
            assert m in ext

    def test_training_metrics_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        train = report.get("training", {})
        for m in TRAINING_METRICS:
            assert m in train

    def test_training_r2_positive(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        assert report["training"]["r2"] >= 0.0

    def test_recommendation_metrics_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        rec = report.get("recommendation", {})
        for m in RECOMMENDATION_METRICS:
            assert m in rec

    def test_reckoner_metrics_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        reck = report.get("reckoner", {})
        for m in RECKONER_METRICS:
            assert m in reck

    def test_overall_score_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        overall = report.get("overall", {})
        for m in OVERALL_METRICS:
            assert m in overall
        assert 0.0 <= overall["overall_aaif_health"] <= 1.0

    def test_agent_scores_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        assert "agent_scores" in report

    def test_comparison_computed(self, agent, sample_df):
        agent.run(sample_df)
        report = _get_report(agent)
        comp = report.get("comparison", {})
        assert "run_count" in comp
        assert "is_best" in comp

    def test_history_created(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        history_path = tmp_path / "benchmark" / "benchmark_history.json"
        agent.run(sample_df, history_path=history_path)
        assert history_path.exists()
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) >= 1

    def test_history_append(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        history_path = tmp_path / "benchmark" / "benchmark_history.json"
        agent.run(sample_df, history_path=history_path)
        agent.run(sample_df, history_path=history_path)
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) >= 2

    def test_benchmark_report_csv(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.run(sample_df)
        csv_path = tmp_path / "benchmark" / "benchmark_report.csv"
        assert csv_path.exists()

    def test_benchmark_report_html(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.run(sample_df)
        html_path = tmp_path / "benchmark" / "benchmark_report.html"
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html>" in content

    def test_benchmark_dashboard_json(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.run(sample_df)
        dash_path = tmp_path / "benchmark" / "benchmark_dashboard.json"
        assert dash_path.exists()
        data = json.loads(dash_path.read_text(encoding="utf-8"))
        assert "pipeline_score" in data
        assert "overall_health" in data

    def test_with_pipeline_results(self, agent, sample_df):
        results = {
            "training": AgentContract(agent_name="TrainingAgent", status="success"),
            "recommendation": AgentContract(agent_name="RecommendationAgent", status="success"),
        }
        agent.run(sample_df, pipeline_results=results)
        report = _get_report(agent)
        assert "agent_scores" in report

    def test_with_reference_df(self, agent, sample_df):
        ref = sample_df.copy()
        agent.run(sample_df, reference_df=ref)
        report = _get_report(agent)
        assert report["extraction"]["precision"] >= 0.0

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame({"Crop": []})
        contract = agent.run(df)
        report = _get_report(agent)
        assert report["overall"]["overall_aaif_health"] >= 0.0

    def test_missing_yield_columns(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"], "Nitrogen": [100]})
        agent.run(df)
        report = _get_report(agent)
        assert report["training"]["r2"] == 0.0

    def test_no_recommendation_columns(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"]})
        agent.run(df)
        report = _get_report(agent)
        assert report["recommendation"]["coverage"] == 0.0

    def test_metric_constants(self):
        assert len(EXTRACTION_METRICS) == 6
        assert len(TRAINING_METRICS) == 5
        assert len(RECOMMENDATION_METRICS) == 4
        assert len(RECKONER_METRICS) == 4
        assert len(OVERALL_METRICS) == 6

    def test_artifacts_collected(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        contract = agent.run(sample_df)
        assert len(contract.artifacts) > 0
