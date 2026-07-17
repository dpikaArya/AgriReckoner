"""Tests for Evidence Fusion Agent."""

import pandas as pd
import pytest

from agri_ai_agent.agents.evidence_fusion_agent import EvidenceFusionAgent
from agri_ai_agent.contracts.messages import AgentContract


class TestEvidenceFusionAgent:
    def setup_method(self):
        self.agent = EvidenceFusionAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "EvidenceFusionAgent"

    def test_process_empty_stage_results(self):
        result = self.agent.process(pd.DataFrame(), stage_results=[], source_file="test.pdf")
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_process_single_reader(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Crop": "Wheat",
                        "Yield_per_Hectare": 3500.0,
                    }
                ],
            }
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        assert not result.empty
        assert "Yield_per_Hectare" in result.columns
        assert result.iloc[0]["Yield_per_Hectare"] == 3500.0
        assert "_provenance" in result.columns
        assert "_fused_confidence" in result.columns

    def test_fuse_multiple_readers_highest_confidence(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Yield_per_Hectare": 3000.0,
                    }
                ],
            },
            {
                "reader": "camelot",
                "success": True,
                "confidence": 0.85,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Yield_per_Hectare": 3200.0,
                    }
                ],
            },
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        assert not result.empty
        row = result.iloc[0]
        assert row["Yield_per_Hectare"] == 3200.0
        prov = row["_provenance"]
        assert "Yield_per_Hectare" in prov
        assert prov["Yield_per_Hectare"]["source_reader"] == "camelot"
        assert prov["Yield_per_Hectare"]["conflict_resolved"] is True

    def test_provenance_preserved(self):
        stage_results = [
            {
                "reader": "pdfplumber",
                "success": True,
                "confidence": 0.8,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Crop": "Rice",
                        "Plant_Height_cm": 85.5,
                        "Yield_per_Hectare": 4200.0,
                    }
                ],
            }
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        row = result.iloc[0]
        assert "_provenance" in row.index
        assert "_evidence_count" in row.index
        assert "_readers_used" in row.index
        assert "pdfplumber" in row["_readers_used"]
        assert row["_evidence_count"] == 1

    def test_unit_normalization_yield_per_hectare(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Yield_per_Hectare": 3500000.0,
                    }
                ],
            }
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        row = result.iloc[0]
        assert row["Yield_per_Hectare"] == 3500.0

    def test_deduplication_keeps_richest(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Yield_per_Hectare": 3000.0,
                    }
                ],
            },
            {
                "reader": "camelot",
                "success": True,
                "confidence": 0.85,
                "rows": [
                    {
                        "Source_File": "paper1.pdf",
                        "Treatment": "T1",
                        "Yield_per_Hectare": 3200.0,
                        "Plant_Height_cm": 90.0,
                    }
                ],
            },
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        assert len(result) == 1
        assert result.iloc[0]["Plant_Height_cm"] == 90.0

    def test_crop_consistency_correction(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {"Source_File": "paper1.pdf", "Treatment": "T1", "Crop": "Wheat"},
                    {"Source_File": "paper1.pdf", "Treatment": "T2", "Crop": "Rice"},
                    {"Source_File": "paper1.pdf", "Treatment": "T3", "Crop": "Wheat"},
                ],
            }
        ]
        result = self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        crops = result["Crop"].tolist()
        assert len(set(crops)) == 1

    def test_fallback_metadata(self):
        result = self.agent.process(
            pd.DataFrame(), stage_results=[], source_file="/path/to/paper1.pdf"
        )
        assert not result.empty
        assert result.iloc[0]["Source_File"] == "paper1.pdf"
        assert result.iloc[0]["Treatment"] == "UNKNOWN"

    def test_run_method_returns_contract(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [
                    {"Source_File": "paper1.pdf", "Treatment": "T1", "Crop": "Wheat"}
                ],
            }
        ]
        contract = self.agent.run(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        assert isinstance(contract, AgentContract)
        assert contract.status == "success"
        assert "fusion_report" in contract.output_data

    def test_fusion_report_structure(self):
        stage_results = [
            {
                "reader": "pdfminer",
                "success": True,
                "confidence": 0.7,
                "rows": [{"Source_File": "paper1.pdf", "Treatment": "T1"}],
            },
            {
                "reader": "camelot",
                "success": True,
                "confidence": 0.85,
                "rows": [{"Source_File": "paper1.pdf", "Treatment": "T1"}],
            },
        ]
        self.agent.process(
            pd.DataFrame(), stage_results=stage_results, source_file="paper1.pdf"
        )
        report = self.agent.fusion_report
        assert "stages_input" in report
        assert "stages_successful" in report
        assert "total_raw_rows" in report
        assert "rows_after_fusion" in report
        assert "fusion_rate" in report
        assert "readers" in report
        assert report["stages_input"] == 2
        assert report["total_raw_rows"] == 2
        assert report["rows_after_fusion"] == 1
