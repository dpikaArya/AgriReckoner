"""Tests for ExplainabilityAgent — AAIF v2.0."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.explainability_agent import ExplainabilityAgent
from agri_ai_agent.contracts.messages import AgentContract


@pytest.fixture
def agent():
    return ExplainabilityAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize"],
        "Nitrogen": [100.0, 80.0, 90.0],
        "Phosphorus": [25.0, 20.0, 15.0],
        "Potassium": [150.0, 120.0, 100.0],
        "Soil_pH": [6.5, 7.0, 6.8],
        "Rainfall": [1200, 800, 900],
        "Temperature_Max": [33, 30, 32],
        "Temperature_Min": [22, 18, 20],
        "Yield_per_Hectare": [4.5, 3.2, 5.1],
        "Target_Yield": [4.8, 3.5, 5.3],
        "Predicted_Yield": [4.6, 3.3, 5.0],
        "Fertilizer_Name": ["Urea", "DAP", "NPK"],
        "Dose": [50.0, 30.0, 40.0],
        "Application_Interval": [30, 45, 60],
        "Recommended_Fertilizer": ["Urea", "DAP", "NPK"],
        "Recommended_Dose": [50.0, 30.0, 40.0],
        "Confidence_Score": [0.85, 0.72, 0.91],
        "Source_Paper": ["Paper_A", "Paper_B", "Paper_C"],
        "Fuzzy_N_Action": ["Apply Urea", "Apply DAP", "Apply NPK"],
        "Fuzzy_P_Action": ["Maintain", "Increase", "Maintain"],
        "Fuzzy_K_Action": ["No change", "Apply MOP", "No change"],
        "Fuzzy_Risk": ["Low", "Medium", "Low"],
        "Recommendation_Summary": ["Best option for Rice", "Good for Wheat", "Optimal for Maize"],
        "Top_Alternatives": ['[{"fertilizer":"DAP","dose":30}]',
                             '[{"fertilizer":"NPK","dose":40}]',
                             '[{"fertilizer":"Urea","dose":50}]'],
    })


class TestExplainabilityAgent:

    def test_agent_name(self, agent):
        assert agent.agent_name == "ExplainabilityAgent"

    def test_run_returns_contract(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        contract = agent.run(sample_df)
        assert contract.status == "success"

    def test_explanations_generated(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        contract = agent.run(sample_df)
        assert contract.output_data["explanation_count"] == 3

    def test_individual_json_reports(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        expl_dir = tmp_path / "reports" / "explainability"
        assert expl_dir.exists()
        json_files = list(expl_dir.glob("PRED_*.json"))
        assert len(json_files) == 3

    def test_summary_json(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        assert summary.exists()
        data = json.loads(summary.read_text(encoding="utf-8"))
        assert len(data) == 3

    def test_feature_importance_csv(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        csv_path = tmp_path / "reports" / "explainability" / "feature_importance.csv"
        assert csv_path.exists()

    def test_explainability_dashboard(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        html_path = tmp_path / "reports" / "explainability" / "explainability_dashboard.html"
        assert html_path.exists()
        content = html_path.read_text(encoding="utf-8")
        assert "<html>" in content

    def test_explanation_structure(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        exp = data[0]
        assert "prediction_id" in exp
        assert "crop" in exp
        assert "prediction" in exp
        assert "feature_importance" in exp
        assert "fuzzy_rules" in exp
        assert "evidence" in exp
        assert "data_quality" in exp
        assert "recommendation" in exp

    def test_prediction_fields(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        pred = data[0]["prediction"]
        assert "model_used" in pred
        assert "predicted_yield" in pred
        assert "confidence" in pred
        assert "uncertainty" in pred

    def test_feature_importance_top_10(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        features = data[0]["feature_importance"]["top_10_variables"]
        assert len(features) <= 10
        for f in features:
            assert "feature" in f
            assert "importance" in f

    def test_fuzzy_rules_extracted(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        rules = data[0]["fuzzy_rules"]["activated_rules"]
        assert len(rules) > 0

    def test_evidence_papers(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        papers = data[0]["evidence"]["papers_used"]
        assert "Paper_A" in papers

    def test_soil_conditions(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        soil = data[0]["evidence"]["soil_conditions"]
        assert "ph" in soil
        assert soil["ph"] == 6.5

    def test_climate_conditions(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        climate = data[0]["evidence"]["climate_conditions"]
        assert "rainfall_mm" in climate
        assert climate["rainfall_mm"] == 1200

    def test_data_quality_score(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        dq = data[0]["data_quality"]
        assert 0.0 <= dq["data_quality_score"] <= 1.0

    def test_recommendation_reason(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        reason = data[0]["recommendation"]["reason_for_recommendation"]
        assert len(reason) > 0

    def test_alternatives_parsed(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        alts = data[0]["recommendation"]["alternatives"]
        assert isinstance(alts, list)

    def test_uncertainty_computed(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        agent.run(sample_df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        unc = data[0]["prediction"]["uncertainty"]
        assert "uncertainty_score" in unc
        assert 0.0 <= unc["uncertainty_score"] <= 1.0

    def test_avg_confidence(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        contract = agent.run(sample_df)
        avg = contract.output_data["avg_confidence"]
        assert 0.0 <= avg <= 1.0

    def test_empty_dataframe(self, agent, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        df = pd.DataFrame({"Crop": ["Rice"]})
        contract = agent.run(df)
        assert contract.status == "success"

    def test_with_pipeline_results(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        results = {
            "training": AgentContract(agent_name="TrainingAgent", status="success",
                                       artifacts=["models/xgboost_yield.pkl"]),
        }
        agent.run(sample_df, pipeline_results=results)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data[0]["prediction"]["model_used"] == "XGBoost"

    def test_missing_fields_detected(self, agent, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        df = pd.DataFrame({
            "Crop": ["Rice"],
            "Nitrogen": [np.nan],
            "Phosphorus": [20.0],
        })
        agent.run(df)
        summary = tmp_path / "reports" / "explainability" / "prediction_explanations.json"
        data = json.loads(summary.read_text(encoding="utf-8"))
        missing = data[0]["data_quality"]["missing_fields"]
        assert "Nitrogen" in missing

    def test_artifacts_collected(self, agent, sample_df, tmp_path):
        agent.settings.REPORTS_DIR = tmp_path / "reports"
        contract = agent.run(sample_df)
        assert len(contract.artifacts) > 0
