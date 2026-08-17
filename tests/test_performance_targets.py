"""Tests for PerformanceTargets — Phase 16."""

import json

import pytest

from agri_ai_agent.evaluation import (
    TARGETS,
    PerformanceValidator,
)


@pytest.fixture
def validator():
    return PerformanceValidator()


@pytest.fixture
def perfect_metrics():
    return {
        "extraction_accuracy": 0.97,
        "schema_mapping_accuracy": 0.95,
        "missing_data_rate": 0.05,
        "validation_pass_rate": 0.995,
        "feature_coverage": 0.98,
        "recommendation_confidence": 0.92,
        "reckoner_completeness": 0.96,
        "pipeline_success_rate": 1.0,
    }


@pytest.fixture
def failing_metrics():
    return {
        "extraction_accuracy": 0.80,
        "schema_mapping_accuracy": 0.70,
        "missing_data_rate": 0.30,
        "validation_pass_rate": 0.85,
        "feature_coverage": 0.50,
        "recommendation_confidence": 0.60,
        "reckoner_completeness": 0.75,
        "pipeline_success_rate": 0.90,
    }


class TestPerformanceTargets:
    def test_targets_defined(self):
        assert len(TARGETS) == 8
        assert "extraction_accuracy" in TARGETS
        assert "pipeline_success_rate" in TARGETS

    def test_validate_perfect(self, validator, perfect_metrics):
        result = validator.validate(perfect_metrics)
        assert result["all_passed"] is True
        assert result["passed"] == 8
        assert result["pass_rate"] == 1.0

    def test_validate_failing(self, validator, failing_metrics):
        result = validator.validate(failing_metrics)
        assert result["all_passed"] is False
        assert result["passed"] < 8

    def test_get_failures_empty(self, validator, perfect_metrics):
        validator.validate(perfect_metrics)
        assert validator.get_failures() == []

    def test_get_failures_nonempty(self, validator, failing_metrics):
        validator.validate(failing_metrics)
        failures = validator.get_failures()
        assert len(failures) > 0
        assert all("metric" in f for f in failures)
        assert all("gap" in f for f in failures)

    def test_get_passed(self, validator, perfect_metrics):
        validator.validate(perfect_metrics)
        passed = validator.get_passed()
        assert len(passed) == 8

    def test_get_passed_partial(self, validator, failing_metrics):
        validator.validate(failing_metrics)
        passed = validator.get_passed()
        assert len(passed) < 8

    def test_summary_text(self, validator, perfect_metrics):
        validator.validate(perfect_metrics)
        text = validator.summary_text()
        assert "PASS" in text
        assert "8/8" in text

    def test_summary_text_with_failures(self, validator, failing_metrics):
        validator.validate(failing_metrics)
        text = validator.summary_text()
        assert "FAIL" in text

    def test_save_report(self, validator, perfect_metrics, tmp_path):
        validator.validate(perfect_metrics)
        path = validator.save_report(tmp_path / "report.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "results" in data
        assert "targets" in data
        assert data["passed"] == 8

    def test_missing_rate_direction(self, validator):
        metrics = {
            "extraction_accuracy": 1.0,
            "schema_mapping_accuracy": 1.0,
            "missing_data_rate": 0.01,
            "validation_pass_rate": 1.0,
            "feature_coverage": 1.0,
            "recommendation_confidence": 1.0,
            "reckoner_completeness": 1.0,
            "pipeline_success_rate": 1.0,
        }
        result = validator.validate(metrics)
        assert result["metrics"]["missing_data_rate"]["passed"] is True

    def test_missing_rate_too_high(self, validator):
        metrics = {
            "extraction_accuracy": 1.0,
            "schema_mapping_accuracy": 1.0,
            "missing_data_rate": 0.50,
            "validation_pass_rate": 1.0,
            "feature_coverage": 1.0,
            "recommendation_confidence": 1.0,
            "reckoner_completeness": 1.0,
            "pipeline_success_rate": 1.0,
        }
        result = validator.validate(metrics)
        assert result["metrics"]["missing_data_rate"]["passed"] is False

    def test_empty_metrics(self, validator):
        result = validator.validate({})
        assert result["all_passed"] is False
        assert result["passed"] == 1  # missing_data_rate passes at 0.0

    def test_partial_metrics(self, validator):
        metrics = {"extraction_accuracy": 0.98}
        result = validator.validate(metrics)
        assert result["passed"] == 2  # extraction_accuracy + missing_data_rate

    def test_threshold_values(self):
        assert TARGETS["extraction_accuracy"]["min"] == 0.95
        assert TARGETS["schema_mapping_accuracy"]["min"] == 0.90
        assert TARGETS["missing_data_rate"]["max"] == 0.10
        assert TARGETS["validation_pass_rate"]["min"] == 0.99
        assert TARGETS["feature_coverage"]["min"] == 0.95
        assert TARGETS["recommendation_confidence"]["min"] == 0.90
        assert TARGETS["reckoner_completeness"]["min"] == 0.95
        assert TARGETS["pipeline_success_rate"]["min"] == 0.99
