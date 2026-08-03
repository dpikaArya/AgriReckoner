"""Tests for the framework health dashboard generator."""

from __future__ import annotations

from types import SimpleNamespace

from agri_ai_agent.literature.dashboard import HealthMetrics, build_health_dashboard


class _StubHttp:
    def stats(self):
        return {
            "cache_hits": 60,
            "cache_misses": 40,
            "total_requests": 100,
            "avg_response_time_ms": 120.0,
        }


class _StubConnector:
    source_name = "Stub"
    enabled = True

    def __init__(self):
        self.http = _StubHttp()

    def get_cursor(self, key, default=None):
        return None


class _StubAgriManager:
    def __init__(self):
        self.connectors = []


class _StubState:
    def summary(self):
        return {"cached_records": 42, "cursors_by_connector": {}}

    def iter_sources(self):
        return []


class _StubReport:
    run_id = "RUN-1"
    started_at = "2026-01-01T00:00:00+00:00"
    completed_at = "2026-01-01T00:01:00+00:00"
    duration_sec = 60.0
    sync_results = []
    agri_sync_results = []
    total_fetched = 24
    verified_papers = 16
    duplicates = 3
    validated_dois = 16
    failed_pdfs = 2
    agri_records = 5
    errors = ["a harmless connector warning"]


def _pipeline(config, report):
    conn = _StubConnector()
    conn.http = _StubHttp()
    return SimpleNamespace(
        config=config,
        connectors=[conn],
        agri_manager=_StubAgriManager(),
        state=_StubState(),
    )


def test_health_metrics_collection(config_for_tests, tmp_path):
    metrics = HealthMetrics(_StubReport(), _pipeline(config_for_tests, _StubReport()), tmp_path)
    m = metrics.collect()
    assert m["papers_discovered"] == 24
    assert m["verified_original_experiments"] == 16
    assert m["duplicate_rate_pct"] == 12.5  # 3 / 24
    assert m["cache_utilization_pct"] == 60.0  # 60 / (60+40)
    assert m["api_avg_latency_ms"] == 120.0
    assert m["literature_enabled"] == 1
    assert m["total_errors"] == 1


def test_build_health_dashboard_renders_html(config_for_tests, tmp_path):
    (tmp_path / "universal_schema.csv").write_text(
        "Paper_ID,DOI,Journal,Year\np1,10.1/x,FCR,2021\n", encoding="utf-8"
    )
    path = build_health_dashboard(
        _StubReport(), _pipeline(config_for_tests, _StubReport()), tmp_path
    )
    html = path.read_text(encoding="utf-8")
    assert path.name == "framework_health_dashboard.html"
    for label in (
        "Papers Discovered",
        "Verified Original Experiments",
        "Schema Completeness",
        "Duplicate Rate",
        "Missing Values",
        "Training Dataset Size",
        "Best Validation R",
        "Cache Utilization",
        "Connector Failures",
    ):
        assert label in html
    assert "RUN-1" in html
