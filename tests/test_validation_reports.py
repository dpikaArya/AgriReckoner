import json

import pandas as pd

from src.validation.reports import ValidationReport, run_all_validations
from src.validation.schema_validator import ColumnSchema, TableSchema


class TestValidationReport:
    def test_add_result_stores_correctly(self):
        report = ValidationReport()
        report.add_result("SchemaValidator", True, {"valid": True})
        assert len(report.results) == 1
        assert report.results[0]["validator"] == "SchemaValidator"
        assert report.results[0]["passed"] is True
        assert "timestamp" in report.results[0]

    def test_to_dict_returns_expected_structure(self):
        report = ValidationReport()
        report.add_result("TestValidator", True, {})
        d = report.to_dict()
        assert "summary" in d
        assert "results" in d
        assert d["summary"]["total_checks"] == 1
        assert d["summary"]["passed"] == 1
        assert d["summary"]["failed"] == 0
        assert d["summary"]["status"] == "PASS"

    def test_to_dict_fail_status(self):
        report = ValidationReport()
        report.add_result("TestValidator", False, {})
        d = report.to_dict()
        assert d["summary"]["status"] == "FAIL"

    def test_summary_returns_string(self):
        report = ValidationReport()
        report.add_result("TestValidator", True, {})
        s = report.summary()
        assert isinstance(s, str)
        assert "Validation Summary" in s
        assert "PASS" in s

    def test_to_json_writes_valid_file(self, tmp_path):
        report = ValidationReport()
        report.add_result("TestValidator", True, {"check": "ok"})
        p = tmp_path / "report.json"
        report.to_json(p)
        assert p.exists()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["status"] == "PASS"
        assert len(data["results"]) == 1

    def test_to_html_writes_file(self, tmp_path):
        report = ValidationReport()
        report.add_result("TestValidator", True, {})
        p = tmp_path / "report.html"
        report.to_html(p)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "PASS" in content
        assert "<html" in content


class TestRunAllValidations:
    def test_run_all_validations_orchestrates(self):
        df = pd.DataFrame(
            {
                "temp": [1.0, 2.0],
                "city": ["a", "b"],
            }
        )
        result = run_all_validations(df)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "results" in result
        assert result["summary"]["total_checks"] >= 2

    def test_run_all_validations_with_schema(self):
        cols = [
            ColumnSchema(name="temp", dtype="float64"),
            ColumnSchema(name="city", dtype="object"),
        ]
        schema = TableSchema(table_name="test", columns=cols)
        df = pd.DataFrame({"temp": [1.0], "city": ["a"]})
        result = run_all_validations(df, schema=schema)
        assert result["summary"]["total_checks"] >= 3

    def test_run_all_validations_writes_output(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        run_all_validations(df, output_dir=tmp_path)
        report_dir = tmp_path / "reports"
        assert (report_dir / "validation_report.json").exists()
        assert (report_dir / "validation_summary.html").exists()
