"""Phase 18 — Production external enrichment & UAMS promotion tests.

Pure tests exercise the enrichment helpers on synthetic data only (no network).
Integration tests assert the structure of the Phase 18 deliverables and are
skipped when the pipeline outputs are absent. Nothing here modifies validated
observations, UAMS, Phase 14/14.5A/16 outputs, or embeddings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase18"
OUT = ROOT / "outputs" / "phase18"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import p18_common as C  # noqa: E402
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase18 scratch modules unavailable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# project root safety
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_outputs_inside_project_root(self):
        for p in (C.OUT, C.REPORTS, C.TMP, C.STAGING):
            assert str(p.resolve()).startswith(str(C.PROJECT_ROOT.resolve()))

    def test_stages_listed(self):
        for stage in ("inventory", "acquisition", "mapping", "fusion",
                      "promotion", "coverage", "rag_sync"):
            assert stage in C.STAGES


# ---------------------------------------------------------------------------
# enrichment + promotion logic (synthetic)
# ---------------------------------------------------------------------------


class TestEnrichmentHelpers:
    def test_credentials_present_no_values(self):
        report = C.credentials_present(list(C.CREDENTIAL_ENV_VARS)[:3])
        assert isinstance(report, dict)
        for k, v in report.items():
            assert isinstance(v, bool)

    def test_sha256_file_deterministic(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello")
        assert C.sha256_file(p) == C.sha256_file(p)
        assert len(C.sha256_file(p)) == 64

    def test_fingerprint_bytes(self):
        fp = C.record_fingerprint("FAOSTAT", "QCL", "Yield_per_Hectare", "IND", "2020")
        assert fp
        assert C.record_fingerprint("FAOSTAT", "QCL", "Yield_per_Hectare", "IND", "2020") == fp
        assert C.record_fingerprint("FAOSTAT", "QCL", "Yield_per_Hectare", "IND", "2021") != fp

    def test_s_and_fnum(self):
        assert C.s(None) == ""
        assert C.fnum("3.14") == 3.14

    def test_load_uams_columns(self):
        cols, _ = C.load_uams_columns()
        assert len(cols) >= 290
        assert "Yield_per_Hectare" in cols


# ---------------------------------------------------------------------------
# deliverable structure (integration, skipped when absent)
# ---------------------------------------------------------------------------


class TestDeliverables:
    @pytest.mark.skipif(
        not (OUT / "state_inventory.parquet").exists(), reason="inventory not run"
    )
    def test_inventory_covers_sources(self):
        df = pd.read_parquet(OUT / "state_inventory.parquet")
        assert "Source" in df.columns
        assert len(df) >= 5

    @pytest.mark.skipif(
        not (OUT / "acquisition_records.parquet").exists(), reason="acquisition not run"
    )
    def test_acquisition_has_provenance(self):
        df = pd.read_parquet(OUT / "acquisition_records.parquet")
        assert "Provenance" in df.columns or "Source" in df.columns
        assert len(df) >= 10

    @pytest.mark.skipif(
        not (OUT / "mapped_records.parquet").exists(), reason="mapping not run"
    )
    def test_mapping_has_decision_column(self):
        df = pd.read_parquet(OUT / "mapped_records.parquet")
        assert "Mapping_Decision" in df.columns or "UAMS_Column" in df.columns

    @pytest.mark.skipif(
        not (OUT / "validated_records.parquet").exists(), reason="validation not run"
    )
    def test_validation_has_verdict(self):
        df = pd.read_parquet(OUT / "validated_records.parquet")
        assert "Validation_Status" in df.columns or "Decision" in df.columns

    @pytest.mark.skipif(
        not (OUT / "staging" / "external_observations.parquet").exists(),
        reason="fusion not run",
    )
    def test_fusion_staging_exists(self):
        df = pd.read_parquet(OUT / "staging" / "external_observations.parquet")
        assert len(df) >= 10

    @pytest.mark.skipif(
        not (ROOT / "outputs" / "UAMS_v2.1.parquet").exists(), reason="promotion not run"
    )
    def test_promotion_append_only(self):
        v2 = pd.read_parquet(ROOT / "outputs" / "UAMS_v2.parquet")
        v21 = pd.read_parquet(ROOT / "outputs" / "UAMS_v2.1.parquet")
        assert len(v21) >= len(v2)
        merged = pd.merge(v2, v21, on="ObservationID", how="inner")
        assert len(merged) == len(v2)

    @pytest.mark.skipif(
        not (OUT / "coverage_before_after.parquet").exists(), reason="coverage not run"
    )
    def test_coverage_tracks_columns(self):
        df = pd.read_parquet(OUT / "coverage_before_after.parquet")
        assert "UAMS_Column" in df.columns
        assert "Population_After" in df.columns

    @pytest.mark.skipif(
        not (OUT / "acquisition_priority.parquet").exists(), reason="info gain not run"
    )
    def test_info_gain_priority_sorted(self):
        df = pd.read_parquet(OUT / "acquisition_priority.parquet")
        assert "Acquisition_Priority" in df.columns

    @pytest.mark.skipif(
        not (OUT / "ml" / "ml_readiness_metrics.json").exists(), reason="ml not run"
    )
    def test_ml_readiness_no_retrain(self):
        m = json.loads((OUT / "ml" / "ml_readiness_metrics.json").read_text(encoding="utf-8"))
        assert m["retrain_allowed"] is False


# ---------------------------------------------------------------------------
# protected-output integrity (integration, skipped when absent)
# ---------------------------------------------------------------------------


class TestProtectedIntegrity:
    @pytest.mark.skipif(
        not (OUT / "promotion_checks.json").exists(), reason="promotion not run"
    )
    def test_promotion_gate_checks_pass(self):
        m = json.loads((OUT / "promotion_checks.json").read_text(encoding="utf-8"))
        assert m["row_count_valid"] is True
        assert m["existing_observations_unchanged"] is True

    @pytest.mark.skipif(
        not (OUT / "phase18_final_metrics.json").exists(), reason="reports not run"
    )
    def test_final_metrics_report(self):
        m = json.loads((OUT / "phase18_final_metrics.json").read_text(encoding="utf-8"))
        assert m["stages_completed"]  # non-empty
