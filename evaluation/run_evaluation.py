"""
Main entry point for the ADES Evaluation and Benchmarking Framework.
Runs all evaluation stages and generates the complete report suite.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime


def run_all():
    start_time = time.time()
    all_results = {}

    print("=" * 60)
    print("ADES Evaluation and Benchmarking Framework")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Stage 01
    print("[1/14] Research Paper Ingestion Evaluation...", end=" ")
    try:
        from evaluation.stage01_ingestion import evaluate_ingestion
        results, path = evaluate_ingestion()
        all_results["ingestion"] = {
            "completeness": sum(r.get("text_extraction_completeness", 0) for r in results) / len(results) if results else 0,
            "papers_count": len(results),
            "report": path,
        }
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 02
    print("[2/14] Scientific Information Extraction Evaluation...", end=" ")
    try:
        from evaluation.stage02_extraction import evaluate_extraction
        results, path = evaluate_extraction()
        all_results["extraction"] = {
            "avg_f1": sum(r.get("f1_score", 0) for r in results) / len(results) if results else 0,
            "avg_precision": sum(r.get("precision", 0) for r in results) / len(results) if results else 0,
            "avg_recall": sum(r.get("recall", 0) for r in results) / len(results) if results else 0,
            "report": path,
        }
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 03
    print("[3/14] Ontology Mapping Evaluation...", end=" ")
    try:
        from evaluation.stage03_ontology import evaluate_ontology
        results, path = evaluate_ontology()
        coverage_vals = [v["coverage"] for v in results.values()] if results else [0]
        all_results["ontology"] = {
            "avg_coverage": sum(coverage_vals) / len(coverage_vals) if coverage_vals else 0,
            "report": path,
        }
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 04
    print("[4/14] Schema Mapping Evaluation...", end=" ")
    try:
        from evaluation.stage04_schema import evaluate_schema
        results, path = evaluate_schema()
        all_results["schema"] = results
        all_results["schema"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 05
    print("[5/14] Unit Harmonization Evaluation...", end=" ")
    try:
        from evaluation.stage05_unit import evaluate_unit
        results, path = evaluate_unit()
        all_results["unit"] = results
        all_results["unit"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 06
    print("[6/14] Quality Assurance Evaluation...", end=" ")
    try:
        from evaluation.stage06_quality import evaluate_quality
        results, path = evaluate_quality()
        all_results["qa"] = results
        all_results["qa"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 07
    print("[7/14] Feature Engineering Evaluation...", end=" ")
    try:
        from evaluation.stage07_feature import evaluate_features
        results, path = evaluate_features()
        all_results["features"] = results
        all_results["features"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 08
    print("[8/14] Leakage Detection Evaluation...", end=" ")
    try:
        from evaluation.stage08_leakage import evaluate_leakage
        results, path = evaluate_leakage()
        all_results["leakage"] = results
        all_results["leakage"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 09
    print("[9/14] Statistical Diagnostics Evaluation...", end=" ")
    try:
        from evaluation.stage09_statistics import evaluate_statistics
        results, path = evaluate_statistics()
        all_results["statistics"] = results
        all_results["statistics"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 10
    print("[10/14] Model Readiness Evaluation...", end=" ")
    try:
        from evaluation.stage10_model_readiness import evaluate_model_readiness
        results, path = evaluate_model_readiness()
        ready_count = sum(1 for v in results.values() if isinstance(v, dict) and v.get("readiness") == "ready")
        total = len(results)
        all_results["model_readiness"] = {
            "ready_ratio": ready_count / total if total > 0 else 0,
            "ready_count": ready_count,
            "total_models": total,
            "report": path,
        }
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Stage 11
    print("[11/14] Documentation Evaluation...", end=" ")
    try:
        from evaluation.stage11_documentation import evaluate_documentation
        results, path = evaluate_documentation()
        all_results["documentation"] = results
        all_results["documentation"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # End-to-End
    print("[12/14] End-to-End Pipeline Performance...", end=" ")
    try:
        from evaluation.e2e_performance import evaluate_e2e
        results, path = evaluate_e2e()
        all_results["e2e"] = results
        all_results["e2e"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Dataset Quality Score
    print("[13/14] Dataset Quality Score...", end=" ")
    try:
        from evaluation.dataset_quality_score import compute_quality_score
        results, path = compute_quality_score()
        all_results["quality_score"] = results
        all_results["quality_score"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Model Benchmark
    print("[14/14] Model Benchmark...", end=" ")
    try:
        from evaluation.model_benchmark import benchmark_models
        results, path = benchmark_models()
        trained = sum(1 for r in results if "rmse" in r)
        failed = sum(1 for r in results if "error" in r)
        all_results["model_benchmark"] = {
            "trained": trained,
            "failed": failed,
            "success_rate": trained / (trained + failed) if (trained + failed) > 0 else 0,
            "report": path,
        }
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Agent Benchmark
    print("[+] Agent Benchmark...", end=" ")
    try:
        from evaluation.agent_benchmark import benchmark_agents
        results, path = benchmark_agents()
        all_results["agent_benchmark"] = {"csv_path": path}
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Output Validation
    print("[+] Output Validation...", end=" ")
    try:
        from evaluation.output_validation import validate_output
        results, path = validate_output()
        all_results["output_validation"] = results
        all_results["output_validation"]["report"] = path
        print(f"OK → {path}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Dashboard
    print("[+] Final Dashboard Generation...", end=" ")
    try:
        from evaluation.dashboard import generate_dashboard
        dash_results = generate_dashboard(all_results)
        all_results["dashboard"] = dash_results
        print(f"OK → {dash_results.get('dashboard_html', '')}")
    except Exception as e:
        print(f"FAILED: {e}")

    # Results summary
    total_time = time.time() - start_time
    quality = all_results.get("quality_score", {}).get("overall", 0)
    print()
    print("=" * 60)
    print(f"EVALUATION COMPLETE in {total_time:.1f}s")
    print(f"Overall Quality Score: {quality}/100")
    print("=" * 60)
    print()
    print("Reports generated in: evaluation/reports/")
    for key in ["ingestion", "extraction", "ontology", "schema", "unit",
                "qa", "features", "leakage", "statistics", "model_readiness",
                "documentation"]:
        r = all_results.get(key, {})
        if r.get("report"):
            print(f"  - {Path(r['report']).name}")
    print()
    print("Additional outputs:")
    for key in ["e2e", "quality_score", "model_benchmark"]:
        r = all_results.get(key, {})
        if r.get("report"):
            print(f"  - {Path(r['report']).name}")
    for key in ["agent_benchmark", "dashboard"]:
        r = all_results.get(key, {})
        if r:
            for k, v in r.items():
                if v and isinstance(v, str) and v.endswith((".csv", ".md", ".html")):
                    print(f"  - {Path(v).name}")
    print()
    print("Dashboard:")
    dash = all_results.get("dashboard", {})
    if dash.get("dashboard_html"):
        print(f"  - {dash['dashboard_html']}")
    if dash.get("scorecard_md"):
        print(f"  - {dash['scorecard_md']}")
    if dash.get("executive_md"):
        print(f"  - {dash['executive_md']}")

    return all_results


if __name__ == "__main__":
    run_all()
