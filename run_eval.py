"""Run all evaluation stages and collect metrics."""
import sys, json, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

results = {}

try:
    from evaluation.stage04_schema import evaluate_schema
    r, p = evaluate_schema()
    results['schema'] = r
    print("Stage 04 Schema: OK")
except Exception as e:
    print(f"Stage 04 Schema: FAILED - {e}")

try:
    from evaluation.stage05_unit import evaluate_unit
    r, p = evaluate_unit()
    results['unit'] = r
    print("Stage 05 Unit: OK")
except Exception as e:
    print(f"Stage 05 Unit: FAILED - {e}")

try:
    from evaluation.stage06_quality import evaluate_quality
    r, p = evaluate_quality()
    results['quality'] = r
    print("Stage 06 Quality: OK")
except Exception as e:
    print(f"Stage 06 Quality: FAILED - {e}")

try:
    from evaluation.stage07_feature import evaluate_features
    r, p = evaluate_features()
    results['features'] = r
    print("Stage 07 Features: OK")
except Exception as e:
    print(f"Stage 07 Features: FAILED - {e}")

try:
    from evaluation.stage08_leakage import evaluate_leakage
    r, p = evaluate_leakage()
    results['leakage'] = r
    print("Stage 08 Leakage: OK")
except Exception as e:
    print(f"Stage 08 Leakage: FAILED - {e}")

try:
    from evaluation.stage09_statistics import evaluate_statistics
    r, p = evaluate_statistics()
    results['statistics'] = r
    print("Stage 09 Statistics: OK")
except Exception as e:
    print(f"Stage 09 Statistics: FAILED - {e}")

try:
    from evaluation.stage10_model_readiness import evaluate_model_readiness
    r, p = evaluate_model_readiness()
    results['model_readiness'] = r
    print("Stage 10 Model Readiness: OK")
except Exception as e:
    print(f"Stage 10 Model Readiness: FAILED - {e}")

try:
    from evaluation.stage11_documentation import evaluate_documentation
    r, p = evaluate_documentation()
    results['documentation'] = r
    print("Stage 11 Documentation: OK")
except Exception as e:
    print(f"Stage 11 Documentation: FAILED - {e}")

try:
    from evaluation.dataset_quality_score import compute_quality_score
    r, p = compute_quality_score()
    results['quality_score'] = r
    print(f"Quality Score: {r.get('overall', 0)}/100")
except Exception as e:
    print(f"Quality Score: FAILED - {e}")

try:
    from evaluation.e2e_performance import evaluate_e2e
    r, p = evaluate_e2e()
    results['e2e'] = r
    cr = r.get("completion_rate", 0)
    tt = r.get("total_time", 0)
    print(f"E2E: completion={cr:.0%}, time={tt:.1f}s")
except Exception as e:
    print(f"E2E: FAILED - {e}")

try:
    from evaluation.model_benchmark import benchmark_models
    r, p = benchmark_models()
    trained = sum(1 for x in r if "rmse" in x)
    failed = sum(1 for x in r if "error" in x)
    results['model_benchmark'] = {"trained": trained, "failed": failed}
    print(f"Model Benchmark: {trained} trained, {failed} failed")
    for m in r:
        if "rmse" in m:
            print(f"  {m['model']}: R2={m['r2']}, RMSE={m['rmse']}")
except Exception as e:
    print(f"Model Benchmark: FAILED - {e}")

try:
    from evaluation.dashboard import generate_dashboard
    dash = generate_dashboard(results)
    print(f"Dashboard: {dash.get('dashboard_html', '')}")
except Exception as e:
    print(f"Dashboard: FAILED - {e}")

print("\n=== RESULTS ===")
print(json.dumps(results, indent=2, default=str)[:5000])
