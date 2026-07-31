import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.validation.data_quality import DataQualityValidator
from src.validation.provenance_validator import ProvenanceValidator
from src.validation.range_validator import RangeValidator
from src.validation.schema_validator import SchemaValidator, TableSchema


class ValidationReport:
    def __init__(self):
        self.results: list[dict] = []

    def add_result(self, validator_name: str, passed: bool, details: dict):
        self.results.append(
            {
                "validator": validator_name,
                "passed": passed,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def to_dict(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        return {
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "status": "PASS" if failed == 0 else "FAIL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": self.results,
        }

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_html(self, path: Path):
        report = self.to_dict()
        status_color = "green" if report["summary"]["status"] == "PASS" else "red"

        rows_html = ""
        for r in report["results"]:
            color = "green" if r["passed"] else "red"
            rows_html += f"""
            <tr style="background-color: {"#e8f5e9" if r["passed"] else "#ffebee"}">
                <td>{r["validator"]}</td>
                <td style="color: {color}; font-weight: bold;">{"PASS" if r["passed"] else "FAIL"}</td>
                <td style="font-size: 0.9em;"><pre style="white-space: pre-wrap; margin: 0;">{json.dumps(r["details"], indent=2, default=str)}</pre></td>
                <td>{r["timestamp"]}</td>
            </tr>"""

        detail_sections = ""
        for r in report["results"]:
            color = "green" if r["passed"] else "red"
            detail_sections += f"""
            <div style="margin-bottom: 20px; padding: 10px; border-left: 4px solid {color}; background: #fafafa;">
                <h3 style="margin: 0 0 5px 0;">{r["validator"]} — <span style="color: {color};">{"PASS" if r["passed"] else "FAIL"}</span></h3>
                <pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{json.dumps(r["details"], indent=2, default=str)}</pre>
                <p style="font-size: 0.8em; color: #666;">{r["timestamp"]}</p>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; color: #333; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
th {{ background: #f0f0f0; }}
.status {{ font-size: 1.2em; font-weight: bold; padding: 8px 16px; border-radius: 4px; display: inline-block; }}
</style>
</head>
<body>
<h1>Validation Report</h1>
<p class="status" style="background: {status_color}; color: white;">{report["summary"]["status"]}</p>
<p>Generated: {report["summary"]["timestamp"]}</p>
<table>
<thead><tr><th>Validator</th><th>Status</th><th>Details</th><th>Timestamp</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>Per-Validator Details</h2>
{detail_sections}
</body>
</html>"""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    def summary(self) -> str:
        report = self.to_dict()
        s = report["summary"]
        lines = [
            f"Validation Summary: {s['status']}",
            f"  Total checks: {s['total_checks']}",
            f"  Passed: {s['passed']}",
            f"  Failed: {s['failed']}",
            f"  Timestamp: {s['timestamp']}",
        ]
        for r in report["results"]:
            lines.append(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['validator']}")
        return "\n".join(lines)


def run_all_validations(
    df,
    schema: TableSchema | None = None,
    metadata: dict | None = None,
    manifest: dict | None = None,
    lineage: dict | None = None,
    output_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    logger = logger or logging.getLogger(__name__)
    report = ValidationReport()

    if schema is not None:
        try:
            validator = SchemaValidator(schema=schema, logger=logger)
            result = validator.validate(df)
            report.add_result("SchemaValidator", result["valid"], result)
        except Exception as e:
            report.add_result("SchemaValidator", False, {"error": str(e)})

    dq_validator = DataQualityValidator(logger=logger)
    try:
        dq_result = dq_validator.validate(df)
        has_issues = dq_result["duplicate_rows"]["count"] > 0 or any(
            v["count"] > 0 for v in dq_result["missing_values"].values()
        )
        report.add_result("DataQualityValidator", not has_issues, dq_result)
    except Exception as e:
        report.add_result("DataQualityValidator", False, {"error": str(e)})

    range_validator = RangeValidator(logger=logger)
    try:
        range_result = range_validator.validate(df)
        has_violations = any(
            v.get("out_of_range_count", 0) > 0
            for k, v in range_result.items()
            if not k.startswith("_")
        )
        report.add_result("RangeValidator", not has_violations, range_result)
    except Exception as e:
        report.add_result("RangeValidator", False, {"error": str(e)})

    prov_validator = ProvenanceValidator(logger=logger)
    if metadata is not None:
        try:
            meta_result = prov_validator.validate_metadata(metadata)
            report.add_result("ProvenanceValidator-Metadata", meta_result["valid"], meta_result)
        except Exception as e:
            report.add_result("ProvenanceValidator-Metadata", False, {"error": str(e)})

    if lineage is not None:
        try:
            lineage_result = prov_validator.validate_lineage(lineage)
            report.add_result(
                "ProvenanceValidator-Lineage", lineage_result["valid"], lineage_result
            )
        except Exception as e:
            report.add_result("ProvenanceValidator-Lineage", False, {"error": str(e)})

    if output_dir is not None:
        reports_dir = Path(output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(reports_dir / "validation_report.json")
        report.to_html(reports_dir / "validation_summary.html")

    return report.to_dict()
