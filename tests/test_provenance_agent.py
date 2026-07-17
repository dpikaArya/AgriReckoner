"""Tests for ProvenanceAgent — Phase 14."""

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.agents.provenance_agent import (
    ProvenanceAgent,
    EXTRACTION_METHODS,
)


@pytest.fixture
def agent():
    return ProvenanceAgent()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Crop": ["Rice", "Wheat", "Maize"],
        "Nitrogen": [100.0, np.nan, 80.0],
        "Phosphorus": [25.0, 20.0, np.nan],
        "Yield_per_Hectare": [4.5, 3.2, 5.1],
        "N_x_P_ratio": [4.0, np.nan, np.nan],
        "N_squared": [10000.0, np.nan, 6400.0],
    })


class TestProvenanceAgent:

    def test_agent_name(self, agent):
        assert agent.agent_name == "ProvenanceAgent"

    def test_process_returns_dataframe(self, agent, sample_df):
        result = agent.process(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_provenance_columns_added(self, agent, sample_df):
        result = agent.process(sample_df)
        for col in [
            "Provenance_Source", "Provenance_DOI", "Provenance_Method",
            "Provenance_Confidence", "Provenance_Timestamp", "Provenance_Page",
            "Provenance_Table", "Provenance_Derived",
        ]:
            assert col in result.columns

    def test_source_paper_applied(self, agent, sample_df):
        result = agent.process(sample_df, source_paper="TestPaper_2022")
        assert (result["Provenance_Source"] == "TestPaper_2022").all()

    def test_doi_applied(self, agent, sample_df):
        result = agent.process(sample_df, source_doi="10.1000/test")
        assert (result["Provenance_DOI"] == "10.1000/test").all()

    def test_extraction_method_applied(self, agent, sample_df):
        result = agent.process(sample_df, extraction_method="camelot")
        non_derived = result[~result["Provenance_Derived"].fillna(False)]
        assert (non_derived["Provenance_Method"] == "camelot").all()

    def test_default_method_is_master_dataset(self, agent, sample_df):
        result = agent.process(sample_df)
        non_derived = result[~result["Provenance_Derived"].fillna(False)]
        assert (non_derived["Provenance_Method"] == "master_dataset").all()

    def test_page_number_applied(self, agent, sample_df):
        result = agent.process(sample_df, page_number=12)
        assert (result["Provenance_Page"] == 12).all()

    def test_table_id_applied(self, agent, sample_df):
        result = agent.process(sample_df, table_id="T3")
        assert (result["Provenance_Table"] == "T3").all()

    def test_derived_columns_tagged(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Provenance_Derived"] is True or result.loc[0, "Provenance_Derived"] == True
        assert result.loc[0, "Provenance_Method"] == "computed"

    def test_missing_values_get_missing_fields_tag(self, agent, sample_df):
        result = agent.process(sample_df)
        assert "Provenance_Missing_Fields" in result.columns
        wheat_missing = result.loc[1, "Provenance_Missing_Fields"]
        assert "Nitrogen" in wheat_missing

    def test_non_missing_retains_confidence(self, agent, sample_df):
        result = agent.process(sample_df)
        assert result.loc[0, "Provenance_Confidence"] == 0.90

    def test_original_columns_preserved(self, agent, sample_df):
        result = agent.process(sample_df)
        assert "Crop" in result.columns
        assert "Yield_per_Hectare" in result.columns
        assert "Nitrogen" in result.columns

    def test_timestamp_format(self, agent, sample_df):
        result = agent.process(sample_df)
        ts = result["Provenance_Timestamp"].iloc[0]
        assert "T" in ts

    def test_coverage_pct(self, agent, sample_df):
        result = agent.process(sample_df)
        cov = agent._coverage_pct(result)
        assert 0.0 <= cov <= 100.0

    def test_total_values(self, agent, sample_df):
        result = agent.process(sample_df)
        total = agent._total_values(result)
        assert total > 0

    def test_report_structure(self, agent, sample_df):
        from agri_ai_agent.contracts.messages import AgentContract
        agent.contract = AgentContract(agent_name="ProvenanceAgent")
        agent.process(sample_df)
        report = agent.contract.output_data.get("provenance_report", {})
        assert "provenance_columns_added" in report
        assert "coverage_pct" in report
        assert "method_distribution" in report
        assert "average_confidence" in report

    def test_validate_provenance_valid(self, agent, sample_df):
        agent.process(sample_df, source_paper="Paper_A")
        result = agent.validate_provenance(sample_df)
        assert result["issue_count"] == 0 or all(
            "unknown source" not in i for i in result["issues"]
        )

    def test_validate_provenance_missing_column(self, agent, sample_df):
        result = agent.validate_provenance(sample_df)
        assert result["valid"] is False
        assert any("Provenance_Source" in i for i in result["issues"])

    def test_validate_provenance_unknown_source(self, agent, sample_df):
        agent.process(sample_df)
        result = agent.validate_provenance(sample_df)
        assert any("unknown source" in i for i in result["issues"])

    def test_provenance_summary_by_paper(self, agent):
        df = pd.DataFrame({
            "Crop": ["Rice", "Wheat"],
            "Nitrogen": [100, 80],
        })
        agent.process(df, source_paper="Paper_A")
        summary = agent.provenance_summary_by_paper(df)
        assert "Paper_A" in summary
        assert summary["Paper_A"]["rows"] == 2

    def test_multiple_extractions(self, agent, sample_df):
        result = agent.process(sample_df, extraction_method="pdfminer")
        non_derived = result[~result["Provenance_Derived"].fillna(False)]
        assert (non_derived["Provenance_Confidence"] == 0.70).all()

    def test_manual_method(self, agent, sample_df):
        result = agent.process(sample_df, extraction_method="manual")
        non_derived = result[~result["Provenance_Derived"].fillna(False)]
        assert (non_derived["Provenance_Confidence"] == 0.95).all()

    def test_computed_method(self, agent, sample_df):
        result = agent.process(sample_df, extraction_method="computed")
        computed = result[result["Provenance_Derived"].fillna(False)]
        assert (computed["Provenance_Confidence"] == 0.75).all()

    def test_extraction_methods_count(self):
        assert len(EXTRACTION_METHODS) >= 12

    def test_empty_dataframe(self, agent):
        df = pd.DataFrame({"Crop": []})
        result = agent.process(df)
        assert isinstance(result, pd.DataFrame)
        assert "Provenance_Source" in result.columns

    def test_no_original_columns(self, agent):
        df = pd.DataFrame()
        result = agent.process(df)
        assert "Provenance_Source" in result.columns
