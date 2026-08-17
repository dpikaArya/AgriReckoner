"""
PerformanceTargets — Phase 16.

Defines and validates pipeline performance targets:
  - extraction_accuracy >= 95%
  - schema_mapping_accuracy >= 90%
  - missing_data_rate <= 10%
  - validation_pass_rate >= 99%
  - feature_coverage >= 95%
  - recommendation_confidence >= 90%
  - reckoner_completeness >= 95%
  - pipeline_success_rate >= 99%
"""

import json
from datetime import datetime
from pathlib import Path

TARGETS = {
    "extraction_accuracy": {"min": 0.95, "direction": ">="},
    "schema_mapping_accuracy": {"min": 0.90, "direction": ">="},
    "missing_data_rate": {"max": 0.10, "direction": "<="},
    "validation_pass_rate": {"min": 0.99, "direction": ">="},
    "feature_coverage": {"min": 0.95, "direction": ">="},
    "recommendation_confidence": {"min": 0.90, "direction": ">="},
    "reckoner_completeness": {"min": 0.95, "direction": ">="},
    "pipeline_success_rate": {"min": 0.99, "direction": ">="},
}


class PerformanceValidator:
    def __init__(self, targets: dict | None = None):
        self.targets = targets or TARGETS
        self._results: dict[str, dict] = {}

    @property
    def results(self) -> dict[str, dict]:
        return dict(self._results)

    def validate(self, metrics: dict[str, float]) -> dict:
        self._results = {}
        passed = 0
        total = len(self.targets)

        for metric_name, rule in self.targets.items():
            actual = metrics.get(metric_name, 0.0)
            direction = rule["direction"]

            if direction == ">=":
                threshold = rule["min"]
                is_pass = actual >= threshold
            elif direction == "<=":
                threshold = rule["max"]
                is_pass = actual <= threshold
            else:
                is_pass = False
                threshold = 0.0

            self._results[metric_name] = {
                "actual": round(actual, 4),
                "threshold": threshold,
                "direction": direction,
                "passed": is_pass,
                "gap": round(abs(actual - threshold), 4) if not is_pass else 0.0,
            }
            if is_pass:
                passed += 1

        return {
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "metrics": self._results,
        }

    def get_failures(self) -> list[dict]:
        failures = []
        for name, result in self._results.items():
            if not result["passed"]:
                failures.append(
                    {
                        "metric": name,
                        "actual": result["actual"],
                        "threshold": result["threshold"],
                        "direction": result["direction"],
                        "gap": result["gap"],
                    }
                )
        return failures

    def get_passed(self) -> list[str]:
        return [name for name, r in self._results.items() if r["passed"]]

    def summary_text(self) -> str:
        lines = ["=" * 70, "Performance Targets Validation", "=" * 70]
        for name, result in self._results.items():
            status = "PASS" if result["passed"] else "FAIL"
            threshold_str = f"{result['direction']} {result['threshold'] * 100:.1f}%"
            lines.append(
                f"  [{status}] {name:.<45s} {result['actual'] * 100:.1f}% (target: {threshold_str})"
            )
        passed = sum(1 for r in self._results.values() if r["passed"])
        total = len(self._results)
        lines.append(f"  Overall: {passed}/{total} targets met")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now().isoformat(),
            "targets": self.targets,
            "results": self._results,
            "passed": sum(1 for r in self._results.values() if r["passed"]),
            "total": len(self._results),
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path
