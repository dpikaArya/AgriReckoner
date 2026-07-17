"""Tests for Table Intelligence Agent."""

import pandas as pd
import pytest

from agri_ai_agent.agents.table_intelligence_agent import TableIntelligenceAgent
from agri_ai_agent.contracts.messages import AgentContract


class TestTableIntelligenceAgent:
    def setup_method(self):
        self.agent = TableIntelligenceAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "TableIntelligenceAgent"

    def test_process_empty_dataframe(self):
        result = self.agent.process(pd.DataFrame())
        assert result.empty

    def test_classify_yield_table(self):
        table = {
            "headers": ["Treatment", "Yield (kg/ha)", "Biomass (kg/ha)", "Harvest Index"],
            "rows": [["T1", "3500", "8000", "0.44"]],
            "source_file": "test.pdf",
        }
        table_type = self.agent._classify_table(table)
        assert table_type == "yield"

    def test_classify_growth_table(self):
        table = {
            "headers": ["Treatment", "Plant Height (cm)", "Leaf Area (cm2)", "SPAD"],
            "rows": [["T1", "85.5", "45.2", "42.3"]],
            "source_file": "test.pdf",
        }
        table_type = self.agent._classify_table(table)
        assert table_type == "growth"

    def test_classify_soil_table(self):
        table = {
            "headers": ["Soil pH", "EC (dS/m)", "Organic Carbon (%)", "N (kg/ha)"],
            "rows": [["6.5", "0.32", "0.45", "280"]],
            "source_file": "test.pdf",
        }
        table_type = self.agent._classify_table(table)
        assert table_type == "soil"

    def test_classify_treatment_table(self):
        table = {
            "headers": ["Treatment", "Fertilizer", "Dose (kg/ha)", "Method"],
            "rows": [["T1", "Urea", "120", "Soil application"]],
            "source_file": "test.pdf",
        }
        table_type = self.agent._classify_table(table)
        assert table_type == "treatment"

    def test_parse_mean_sd(self):
        result = self.agent._parse_statistical_value("35.5 ± 2.3")
        assert result is not None
        assert result["mean"] == 35.5
        assert result["sd"] == 2.3

    def test_parse_mean_lsd(self):
        result = self.agent._parse_statistical_value("35.5 (LSD = 2.3)")
        assert result is not None
        assert result["mean"] == 35.5
        assert result["lsd"] == 2.3

    def test_parse_mean_cd(self):
        result = self.agent._parse_statistical_value("35.5 (CD = 1.8)")
        assert result is not None
        assert result["mean"] == 35.5
        assert result["cd"] == 1.8

    def test_parse_mean_cv(self):
        result = self.agent._parse_statistical_value("35.5 (CV = 5.2%)")
        assert result is not None
        assert result["mean"] == 35.5
        assert result["cv"] == 5.2

    def test_parse_plain_number(self):
        result = self.agent._parse_statistical_value("35.5")
        assert result is not None
        assert result["mean"] == 35.5

    def test_parse_invalid_value(self):
        result = self.agent._parse_statistical_value("NA")
        assert result is None

    def test_extract_from_structured_table(self):
        tables = [
            {
                "headers": ["Treatment", "Yield (kg/ha)", "Plant Height (cm)"],
                "rows": [
                    ["T1", "3500", "85.5"],
                    ["T2", "4200", "92.3"],
                ],
                "source_file": "test.pdf",
            }
        ]
        result = self.agent.process(pd.DataFrame(), tables=tables)
        assert not result.empty
        assert len(result) == 2
        assert "Treatment" in result.columns

    def test_extract_with_statistics(self):
        tables = [
            {
                "headers": ["Treatment", "Yield (kg/ha)"],
                "rows": [["T1", "3500 ± 250"]],
                "source_file": "test.pdf",
            }
        ]
        result = self.agent.process(pd.DataFrame(), tables=tables)
        assert not result.empty
        assert "Yield (kg/ha)" in result.columns
        assert "Yield (kg/ha)_SD" in result.columns

    def test_identify_control_treatment(self):
        tables = [
            {
                "headers": ["Treatment", "Yield (kg/ha)"],
                "rows": [
                    ["Control", "3500"],
                    ["T1", "4200"],
                ],
                "source_file": "test.pdf",
            }
        ]
        result = self.agent.process(pd.DataFrame(), tables=tables)
        control_row = result[result["Treatment"] == "Control"]
        assert not control_row.empty
        assert control_row.iloc[0]["Is_Control"] is True

    def test_extract_dose_from_treatment(self):
        tables = [
            {
                "headers": ["Treatment", "Yield (kg/ha)"],
                "rows": [["N at 120 kg/ha", "4200"]],
                "source_file": "test.pdf",
            }
        ]
        result = self.agent.process(pd.DataFrame(), tables=tables)
        assert not result.empty
        assert "Dose_Value" in result.columns

    def test_process_raw_text(self):
        raw_text = """Treatment\tYield (kg/ha)\tHeight (cm)
T1\t3500\t85.5
T2\t4200\t92.3
"""
        result = self.agent.process(pd.DataFrame(), raw_text=raw_text)
        assert not result.empty
        assert len(result) == 2

    def test_process_dataframe_passthrough(self):
        df = pd.DataFrame({
            "Treatment": ["T1", "T2"],
            "Yield": [3500, 4200],
        })
        result = self.agent.process(df)
        assert not result.empty
        assert len(result) == 2

    def test_run_method_returns_contract(self):
        tables = [
            {
                "headers": ["Treatment", "Yield"],
                "rows": [["T1", "3500"]],
                "source_file": "test.pdf",
            }
        ]
        contract = self.agent.run(pd.DataFrame(), tables=tables)
        assert isinstance(contract, AgentContract)
        assert contract.status == "success"

    def test_extraction_report_structure(self):
        tables = [
            {
                "headers": ["Treatment", "Yield (kg/ha)", "Plant Height (cm)"],
                "rows": [
                    ["Control", "3500 ± 250"],
                    ["T1", "4200 ± 300"],
                ],
                "source_file": "test.pdf",
            }
        ]
        self.agent.process(pd.DataFrame(), tables=tables)
        report = self.agent.extraction_report
        assert "total_rows_extracted" in report
        assert "table_types" in report
        assert "rows_with_statistics" in report
        assert "control_rows" in report
        assert report["total_rows_extracted"] == 2
        assert report["control_rows"] == 1
