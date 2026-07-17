"""Tests for Validation Agent v2 - biological/agronomic/statistical validation."""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.validation_agent import (
    ValidationAgent,
    BIOLOGICAL_RULES,
    AGRONOMIC_RULES,
)


class TestValidationAgentV2:
    def setup_method(self):
        self.agent = ValidationAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "ValidationAgent"

    def test_biological_range_check(self):
        df = pd.DataFrame({
            "Plant_Height_cm": [50, 100, 900],
        })
        result = self.agent._validate_biological_rules(df)
        assert any("Plant_Height_cm" in v for v in result)

    def test_biological_range_ok(self):
        df = pd.DataFrame({
            "Plant_Height_cm": [50, 100, 80],
        })
        result = self.agent._validate_biological_rules(df)
        assert not any("Plant_Height_cm" in v for v in result)

    def test_biological_soil_ph(self):
        df = pd.DataFrame({"Soil_pH": [6.5, 7.0, 5.5]})
        result = self.agent._validate_biological_rules(df)
        assert not any("Soil_pH" in v for v in result)

    def test_biological_soil_ph_violation(self):
        df = pd.DataFrame({"Soil_pH": [2.0, 12.0]})
        result = self.agent._validate_biological_rules(df)
        assert any("Soil_pH" in v for v in result)

    def test_biological_harvest_index(self):
        df = pd.DataFrame({"Harvest_Index": [0.3, 0.45, 0.9]})
        result = self.agent._validate_biological_rules(df)
        assert any("Harvest_Index" in v for v in result)

    def test_agronomic_temperature_consistency(self):
        df = pd.DataFrame({
            "Temperature_Max": [30, 25],
            "Temperature_Min": [20, 30],
        })
        result = self.agent._validate_agronomic_rules(df)
        assert any("Tmax < Tmin" in v for v in result)

    def test_agronomic_temperature_ok(self):
        df = pd.DataFrame({
            "Temperature_Max": [30, 25],
            "Temperature_Min": [20, 15],
        })
        result = self.agent._validate_agronomic_rules(df)
        assert not any("Tmax < Tmin" in v for v in result)

    def test_agronomic_yield_biomass_ratio(self):
        df = pd.DataFrame({
            "Yield_per_Hectare": [5000, 8000],
            "Biomass_Yield": [3000, 4000],
        })
        result = self.agent._validate_agronomic_rules(df)
        assert any("Yield/Biomass" in v for v in result)

    def test_mad_outlier_detection(self):
        vals = [10, 12, 11, 13, 10, 12, 100]
        df = pd.DataFrame({"height": vals})
        result = self.agent._detect_outliers_mad(df)
        assert any("height" in v for v in result)

    def test_mad_no_outliers(self):
        vals = [10, 12, 11, 13, 10, 12, 11]
        df = pd.DataFrame({"height": vals})
        result = self.agent._detect_outliers_mad(df)
        assert not any("height" in v for v in result)

    def test_full_quality_checks_include_new_fields(self):
        df = pd.DataFrame({
            "Plant_Height_cm": [50, 100, 900],
            "Soil_pH": [6.5, 2.0, 7.0],
            "Temperature_Max": [30, 25, 35],
            "Temperature_Min": [20, 30, 15],
        })
        issues = self.agent._quality_checks(df)
        assert "biological_violations" in issues
        assert "agronomic_violations" in issues
        assert "outliers_mad" in issues
        assert len(issues["biological_violations"]) > 0
        assert len(issues["agronomic_violations"]) > 0

    def test_quality_checks_empty_dataframe(self):
        df = pd.DataFrame({"Plant_Height_cm": [50, 100]})
        issues = self.agent._quality_checks(df)
        assert "biological_violations" in issues
        assert "agronomic_violations" in issues
        assert "outliers_mad" in issues

    def test_process_returns_dataframe(self):
        df = pd.DataFrame({"Soil_pH": [6.5, 7.0], "EC": [0.3, 0.5]})
        result = self.agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_range_constraint_check(self):
        df = pd.DataFrame({"Soil_pH": [-1.0, 6.5, 15.0]})
        issues = self.agent._quality_checks(df)
        assert any("Soil_pH" in v for v in issues["impossible_values"])

    def test_biological_rules_defined(self):
        assert len(BIOLOGICAL_RULES) > 5
        for col, rules in BIOLOGICAL_RULES.items():
            assert "min" in rules
            assert "max" in rules
            assert "unit" in rules

    def test_agronomic_rules_defined(self):
        assert len(AGRONOMIC_RULES) > 3
        for rule in AGRONOMIC_RULES:
            assert "name" in rule
            assert "condition" in rule
            assert "check" in rule
            assert "message" in rule
            assert "severity" in rule
