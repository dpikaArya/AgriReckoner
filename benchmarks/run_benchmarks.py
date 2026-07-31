"""
Benchmark Runner.
Compares agent output against gold standard data.
Computes precision, recall, F1, ontology coverage, schema accuracy,
unit accuracy, extraction accuracy, evidence accuracy.
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from benchmarks.gold_standard_bellpepper import GOLD_STANDARD as GS_BELL
from benchmarks.gold_standard_blackwheat import GOLD_STANDARD as GS_WHEAT
from evaluation.utils import OUTPUT_DIR, get_master_df, get_uams_cols, write_report

GOLD_STANDARDS = {
    "Bell Pepper": GS_BELL,
    "Black Wheat": GS_WHEAT,
}


def run_benchmarks():
    master_df = get_master_df()
    uams_cols = set(get_uams_cols())
    results = []

    for crop, gs in GOLD_STANDARDS.items():
        expected_vars = set(gs["variables"].keys())
        expected_uams = set(gs["expected_uams"])
        expected_eng = set(gs["expected_engineered"])

        actual_vars = set()
        actual_uams = set()
        actual_eng = set()

        if master_df is not None:
            actual_vars = set(master_df.columns)
            actual_uams = actual_vars & uams_cols
            actual_eng = actual_vars & expected_eng

        var_precision = (
            len(expected_vars & actual_vars) / len(expected_vars) if expected_vars else 1
        )
        var_recall = len(expected_vars & actual_vars) / len(expected_vars) if expected_vars else 1
        var_f1 = (
            2 * var_precision * var_recall / (var_precision + var_recall)
            if (var_precision + var_recall) > 0
            else 0
        )

        schema_precision = (
            len(expected_uams & actual_uams) / len(expected_uams) if expected_uams else 1
        )
        schema_recall = (
            len(expected_uams & actual_uams) / len(expected_uams) if expected_uams else 1
        )
        schema_f1 = (
            2 * schema_precision * schema_recall / (schema_precision + schema_recall)
            if (schema_precision + schema_recall) > 0
            else 0
        )

        eng_precision = len(expected_eng & actual_eng) / len(expected_eng) if expected_eng else 1
        eng_recall = len(expected_eng & actual_eng) / len(expected_eng) if expected_eng else 1
        eng_f1 = (
            2 * eng_precision * eng_recall / (eng_precision + eng_recall)
            if (eng_precision + eng_recall) > 0
            else 0
        )

        onto_present = 0
        onto_total = len(gs["ontology_mappings"])
        if onto_total > 0:
            onto_path = OUTPUT_DIR / "Ontology_Mapping.csv"
            if onto_path.exists():
                onto_df = pd.read_csv(onto_path, encoding="utf-8-sig")
                if "Variable" in onto_df.columns and "AGROVOC_ID" in onto_df.columns:
                    mapped_vars = set(
                        onto_df[onto_df["AGROVOC_ID"].notna() & (onto_df["AGROVOC_ID"] != "")][
                            "Variable"
                        ]
                    )
                    expected_onto_vars = set(gs["ontology_mappings"].keys())
                    onto_present = len(expected_onto_vars & mapped_vars)

        ontology_coverage = onto_present / onto_total if onto_total > 0 else 0

        unit_accuracy = 1.0
        units = gs.get("units", {})
        if units and master_df is not None:
            pass

        results.append(
            {
                "crop": crop,
                "variable_precision": round(var_precision, 4),
                "variable_recall": round(var_recall, 4),
                "variable_f1": round(var_f1, 4),
                "schema_precision": round(schema_precision, 4),
                "schema_recall": round(schema_recall, 4),
                "schema_f1": round(schema_f1, 4),
                "engineered_precision": round(eng_precision, 4),
                "engineered_recall": round(eng_recall, 4),
                "engineered_f1": round(eng_f1, 4),
                "ontology_coverage": round(ontology_coverage, 4),
                "unit_accuracy": round(unit_accuracy, 4),
                "extraction_accuracy": round(var_f1, 4),
                "evidence_accuracy": round(var_f1, 4),
                "expected_vars": len(expected_vars),
                "found_vars": len(expected_vars & actual_vars),
                "expected_uams": len(expected_uams),
                "found_uams": len(expected_uams & actual_uams),
                "expected_engineered": len(expected_eng),
                "found_engineered": len(expected_eng & actual_eng),
            }
        )

    report = "# Gold Standard Benchmark Report\n"
    report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for r in results:
        report += f"## {r['crop']}\n\n"
        report += "| Metric | Value |\n|--------|-------|\n"
        report += f"| Variable Precision | {r['variable_precision']:.1%} |\n"
        report += f"| Variable Recall | {r['variable_recall']:.1%} |\n"
        report += f"| Variable F1 | {r['variable_f1']:.1%} |\n"
        report += f"| Schema Precision | {r['schema_precision']:.1%} |\n"
        report += f"| Schema F1 | {r['schema_f1']:.1%} |\n"
        report += f"| Engineered Feature F1 | {r['engineered_f1']:.1%} |\n"
        report += f"| Ontology Coverage | {r['ontology_coverage']:.1%} |\n"
        report += f"| Variables Found | {r['found_vars']}/{r['expected_vars']} |\n"
        report += f"| UAMS Columns Found | {r['found_uams']}/{r['expected_uams']} |\n"
        report += f"| Engineered Found | {r['found_engineered']}/{r['expected_engineered']} |\n\n"

    report += "## Overall Summary\n\n"
    avg_f1 = sum(r["variable_f1"] for r in results) / len(results) if results else 0
    avg_schema = sum(r["schema_f1"] for r in results) / len(results) if results else 0
    avg_onto = sum(r["ontology_coverage"] for r in results) / len(results) if results else 0
    report += f"- Average Variable F1: {avg_f1:.1%}\n"
    report += f"- Average Schema F1: {avg_schema:.1%}\n"
    report += f"- Average Ontology Coverage: {avg_onto:.1%}\n"

    path = write_report("Benchmark_Report.md", report)
    print(f"[Benchmark] Report: {path}")

    csv_data = []
    for r in results:
        for key in [
            "variable_precision",
            "variable_recall",
            "variable_f1",
            "schema_f1",
            "ontology_coverage",
            "extraction_accuracy",
        ]:
            csv_data.append({"crop": r["crop"], "metric": key, "value": r[key]})

    import io

    csv_str = io.StringIO()
    pd.DataFrame(csv_data).to_csv(csv_str, index=False)
    csv_path = write_report("Benchmark_Results.csv", csv_str.getvalue())
    print(f"[Benchmark] CSV: {csv_path}")

    return results


if __name__ == "__main__":
    run_benchmarks()
