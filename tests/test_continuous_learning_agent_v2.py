"""Tests for ContinuousLearningAgent v2 — Phase 12."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.continuous_learning_agent import (
    ContinuousLearningAgent,
    BEST_MODEL_LABEL,
    RETRAIN_R2_THRESHOLD,
    DATA_DRIFT_THRESHOLD,
    MIN_SAMPLES_FOR_RETRAIN,
)
from agri_ai_agent.contracts.messages import AgentContract


@pytest.fixture
def agent():
    return ContinuousLearningAgent()


@pytest.fixture
def sample_df():
    n = MIN_SAMPLES_FOR_RETRAIN + 5
    return pd.DataFrame({
        "Crop": ["Rice"] * n,
        "Variety": ["HYV"] * n,
        "Soil_pH": np.full(n, 6.5),
        "Nitrogen": np.full(n, 100.0),
        "Phosphorus": np.full(n, 25.0),
        "Potassium": np.full(n, 150.0),
        "Zinc": np.full(n, 1.5),
        "Rainfall": np.full(n, 1200.0),
        "Temperature_Max": np.full(n, 33.0),
        "Temperature_Min": np.full(n, 22.0),
        "Organic_Carbon": np.full(n, 0.8),
        "Growth_Stage": ["Flowering"] * n,
        "Yield_per_Hectare": np.linspace(3.5, 5.0, n),
        "Target_Yield": np.linspace(4.0, 5.5, n),
        "Predicted_Yield": np.linspace(3.8, 5.2, n),
        "Fertilizer_Name": ["Urea"] * n,
        "Dose": [50.0] * n,
        "Application_Interval": [30] * n,
        "Source_Paper": ["TestPaper"] * n,
        "DOI": ["10.1000/test"] * n,
        "Year": [2022] * n,
        "Country": ["India"] * n,
    })


class TestContinuousLearningAgent:

    def test_agent_name(self, agent):
        assert agent.agent_name == "ContinuousLearningAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_process_empty_dataframe(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"]})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)

    def test_constants_defined(self):
        assert BEST_MODEL_LABEL == "best_model"
        assert RETRAIN_R2_THRESHOLD == 0.01
        assert DATA_DRIFT_THRESHOLD == 0.15
        assert MIN_SAMPLES_FOR_RETRAIN == 20

    def test_should_retrain_below_min_samples(self, agent):
        small_df = pd.DataFrame({"Crop": ["Rice"] * 5})
        assert agent._should_retrain(small_df) is False

    def test_should_retrain_no_history(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        assert agent._should_retrain(sample_df) is True

    def test_should_retrain_with_history_no_growth(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        history_dir = tmp_path / "performance"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "cycle_history.jsonl"
        entry = {
            "initial_rows": len(sample_df),
            "final_rows": len(sample_df),
            "completed_at": "2025-01-01T00:00:00",
        }
        history_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        assert agent._should_retrain(sample_df) is False

    def test_should_retrain_with_growth(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        history_dir = tmp_path / "performance"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "cycle_history.jsonl"
        entry = {"initial_rows": 10, "final_rows": 10, "completed_at": "2025-01-01T00:00:00"}
        history_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        big_df = pd.DataFrame({"Crop": ["Rice"] * 30})
        assert agent._should_retrain(big_df) is True

    def test_merge_new_data(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"], "Value": [1.0]})
        new = pd.DataFrame({"Crop": ["Wheat"], "Value": [2.0]})
        merged = agent._merge_new_data(df, new)
        assert len(merged) == 2

    def test_merge_new_data_empty_df(self, agent):
        df = pd.DataFrame()
        new = pd.DataFrame({"Crop": ["Rice"], "Value": [1.0]})
        merged = agent._merge_new_data(df, new)
        assert len(merged) == 1

    def test_validate_and_transform_adds_schema_cols(self, agent, sample_df):
        result = agent._validate_and_transform(sample_df)
        assert "Crop" in result.columns
        assert "Yield_per_Hectare" in result.columns

    def test_validate_and_transform_numeric_coercion(self, agent):
        df = pd.DataFrame({"Crop": ["Rice"], "Nitrogen": ["100"]})
        result = agent._validate_and_transform(df)
        assert pd.api.types.is_numeric_dtype(result["Nitrogen"])

    def test_remove_duplicates(self, agent):
        df = pd.DataFrame({"Crop": ["Rice", "Rice", "Wheat"], "Value": [1, 1, 2]})
        result = agent._remove_duplicates(df)
        assert len(result) == 2

    def test_remove_duplicates_no_dups(self, agent):
        df = pd.DataFrame({"Crop": ["Rice", "Wheat"], "Value": [1, 2]})
        result = agent._remove_duplicates(df)
        assert len(result) == 2

    def test_load_training_results_empty(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        assert agent._load_training_results() == []

    def test_load_training_results_with_data(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        metrics = pd.DataFrame({"r2": [0.85], "rmse": [0.12]})
        metrics.to_csv(tmp_path / "metrics.csv", index=False)
        results = agent._load_training_results()
        assert len(results) == 1
        assert results[0]["r2"] == 0.85

    def test_deploy_if_improved_no_existing(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "yield_model.pkl"
        model_path.write_bytes(b"fake model")
        deployed = agent._deploy_if_improved([model_path], {"r2": 0.85, "rmse": 0.12, "mae": 0.08})
        assert deployed is True
        assert (models_dir / f"{BEST_MODEL_LABEL}.pkl").exists()

    def test_deploy_if_improved_with_existing_worse(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        prev = {"r2": 0.5, "rmse": 0.3, "mae": 0.2}
        (models_dir / f"{BEST_MODEL_LABEL}_metrics.json").write_text(json.dumps(prev))
        model_path = models_dir / "yield_model.pkl"
        model_path.write_bytes(b"fake model")
        deployed = agent._deploy_if_improved([model_path], {"r2": 0.85, "rmse": 0.12, "mae": 0.08})
        assert deployed is True

    def test_deploy_if_improved_with_existing_better(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        prev = {"r2": 0.99, "rmse": 0.01, "mae": 0.005}
        (models_dir / f"{BEST_MODEL_LABEL}_metrics.json").write_text(json.dumps(prev))
        model_path = models_dir / "yield_model.pkl"
        model_path.write_bytes(b"fake model")
        deployed = agent._deploy_if_improved([model_path], {"r2": 0.85, "rmse": 0.12, "mae": 0.08})
        assert deployed is False

    def test_load_current_best_metrics_none(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        assert agent._load_current_best_metrics() == {}

    def test_load_current_best_metrics_valid(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / f"{BEST_MODEL_LABEL}_metrics.json").write_text(json.dumps({"r2": 0.9}))
        result = agent._load_current_best_metrics()
        assert result["r2"] == 0.9

    def test_log_model_version(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        agent._log_model_version({"r2": 0.9, "rmse": 0.1, "mae": 0.05})
        history_path = models_dir / "model_history.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["r2"] == 0.9

    def test_save_cycle_report(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.contract = AgentContract(agent_name="ContinuousLearningAgent")
        stats = {
            "initial_rows": 25,
            "final_rows": 25,
            "final_columns": 10,
            "retrain_triggered": False,
            "models_retrained": 0,
            "model_deployed": False,
            "drift_events": 0,
            "fuzzy_updated": False,
            "papers_processed": 0,
            "new_rows_added": 0,
            "duplicates_removed": 0,
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
        }
        agent._save_cycle_report(stats, {})
        report_dir = tmp_path / "cycle_reports"
        assert report_dir.exists()
        md_files = list(report_dir.glob("cycle_*.md"))
        json_files = list(report_dir.glob("cycle_*_stats.json"))
        assert len(md_files) >= 1
        assert len(json_files) >= 1

    def test_save_cycle_report_with_drift(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.contract = AgentContract(agent_name="ContinuousLearningAgent")
        stats = {
            "initial_rows": 25,
            "final_rows": 25,
            "final_columns": 10,
            "retrain_triggered": True,
            "models_retrained": 1,
            "model_deployed": True,
            "drift_events": 2,
            "fuzzy_updated": False,
            "papers_processed": 0,
            "new_rows_added": 0,
            "duplicates_removed": 0,
            "started_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:00",
        }
        drift = {"Rainfall_Low": {"suggestion": "shift_threshold_right"}}
        agent._save_cycle_report(stats, drift)
        report_dir = tmp_path / "cycle_reports"
        md_files = list(report_dir.glob("cycle_*.md"))
        assert len(md_files) >= 1
        content = md_files[0].read_text(encoding="utf-8")
        assert "Drift Details" in content

    def test_log_performance_history(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        stats = {"initial_rows": 25, "final_rows": 25}
        agent._log_performance_history(stats)
        history_path = tmp_path / "performance" / "cycle_history.jsonl"
        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_load_performance_history_empty(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        assert agent._load_performance_history() == []

    def test_load_performance_history_multiple(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        history_dir = tmp_path / "performance"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "cycle_history.jsonl"
        lines = [
            json.dumps({"final_rows": 25}),
            json.dumps({"final_rows": 30}),
            json.dumps({"final_rows": 35}),
        ]
        history_path.write_text("\n".join(lines), encoding="utf-8")
        result = agent._load_performance_history()
        assert len(result) == 3
        assert result[2]["final_rows"] == 35

    def test_log_drift(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.contract = AgentContract(agent_name="ContinuousLearningAgent")
        drift = {"Rainfall_Low": {"variable": "Rainfall", "suggestion": "shift_threshold_right"}}
        agent._log_drift(drift)
        drift_dir = tmp_path / "drift"
        assert drift_dir.exists()
        files = list(drift_dir.glob("drift_*.json"))
        assert len(files) >= 1
        content = json.loads(files[0].read_text(encoding="utf-8"))
        assert content["drift_events"] == 1

    def test_log_drift_empty(self, agent, sample_df, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        agent.contract = AgentContract(agent_name="ContinuousLearningAgent")
        agent._log_drift({})
        drift_dir = tmp_path / "drift"
        if drift_dir.exists():
            files = list(drift_dir.glob("drift_*.json"))
            for f in files:
                content = json.loads(f.read_text(encoding="utf-8"))
                assert content["drift_events"] == 0

    def test_build_output(self, agent, sample_df):
        output = agent._build_output(sample_df)
        assert output["rows"] == len(sample_df)
        assert "cycle_complete" in output
        assert output["cycle_complete"] is True

    def test_retrain_and_evaluate_no_target(self, agent, tmp_path):
        agent.settings.OUTPUT_DIR = tmp_path
        df = pd.DataFrame({"Crop": ["Rice"] * 25})
        paths, metrics = agent._retrain_and_evaluate(df)
        assert paths == []
        assert metrics == {}
