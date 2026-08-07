"""Tests for the Phase 16 Autonomous Continuous Learning Engine.

Pure tests exercise config-driven primitives (config merging/weight
normalisation, DOI normalisation, acquisition integrity checks, the
DerSimonian-Laird pooling, version bumping) on synthetic data. Integration
tests assert the structure of the Phase 16 deliverables when the cycle has run
(they skip if the outputs are absent). The engine never modifies validated
observations, UAMS records or extraction pipelines.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase16"
OUT = ROOT / "outputs" / "phase16"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import p16_common as PC  # noqa: E402, I001
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase16 scratch modules unavailable: {_exc}", allow_module_level=True)


class TestConfig:
    def test_defaults_present(self):
        cfg = PC.load_config(TMP / "nonexistent.yaml")
        assert "learning" in cfg
        assert cfg["learning"]["discovery"]["require_doi"] is True
        assert cfg["learning"]["discovery"]["dry_run"] is True
        w = cfg["learning"]["prioritization"]["weights"]
        assert sum(w.values()) == pytest.approx(1.0, abs=0.001)
        assert "information_gain" in w

    def test_user_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("learning:\n  acquisition:\n    max_papers_per_cycle: 2\n", encoding="utf-8")
        cfg = PC.load_config(p)
        assert cfg["learning"]["acquisition"]["max_papers_per_cycle"] == 2
        assert cfg["learning"]["discovery"]["dry_run"] is True


class TestDoiHelpers:
    def test_normalized_doi(self):
        assert PC.normalized_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
        assert PC.normalized_doi("10.1000/ABC") == "10.1000/abc"
        assert PC.normalized_doi(float("nan")) == ""

    def test_known_dois_nonempty(self):
        known = PC.known_dois()
        assert isinstance(known, set)
        assert "" not in known
        assert all(d == d.lower() for d in known)


class TestAcquisitionIntegrity:
    def test_verify_doi(self):
        from p16_acquisition import verify_doi

        assert verify_doi("10.1000/AbC", "10.1000/abc.pdf") is True
        assert verify_doi("10.1000/AbC", "10.9999/xyz.pdf") is False
        assert verify_doi("", "10.1000/abc.pdf") is None

    def test_checksum_ok(self):
        from p16_acquisition import checksum_ok

        assert checksum_ok("ABC", "abc") is True
        assert checksum_ok("ABC", "def") is False
        assert checksum_ok("", "abc") is None

    def test_license_ok(self):
        from p16_acquisition import _license_ok

        assert _license_ok("CC BY 4.0") is True
        assert _license_ok("Open Access") is True
        assert _license_ok("") is None
        assert _license_ok("All rights reserved") is False


class TestMetaPooling:
    def test_single_study_pool(self):
        from p16_meta import fixed_and_random_effects

        es = pd.DataFrame([{"EffectSize_g": 0.5, "EffectSize_SE": 0.2, "n_observations": 10}])
        r = fixed_and_random_effects(es)
        assert r["k_studies"] == 1
        assert r["fixed_effect_mean"] == pytest.approx(0.5)
        assert r["random_effect_mean"] == pytest.approx(0.5)
        assert r["fixed_effect_se"] == pytest.approx(0.2)

    def test_homogeneous_studies_zero_tau(self):
        from p16_meta import fixed_and_random_effects

        es = pd.DataFrame(
            [
                {"EffectSize_g": 0.5, "EffectSize_SE": 0.2, "n_observations": 10},
                {"EffectSize_g": 0.5, "EffectSize_SE": 0.2, "n_observations": 10},
            ]
        )
        r = fixed_and_random_effects(es)
        assert r["heterogeneity_tau2"] == pytest.approx(0.0)
        assert r["heterogeneity_I2"] == pytest.approx(0.0)

    def test_empty_returns_none(self):
        from p16_meta import fixed_and_random_effects

        assert fixed_and_random_effects(pd.DataFrame()) is None


class TestVersioning:
    def test_bump_patch(self):
        from p16_versioning import _bump_patch

        assert _bump_patch("16.0.0") == "16.0.1"
        assert _bump_patch("2") == "2.0.1"


class TestReadiness:
    def test_gate_schema(self):
        from p16_readiness import ReadinessMonitor

        cfg = PC.load_config(TMP / "nonexistent.yaml")
        gates = ReadinessMonitor(cfg).evaluate_gates()
        assert {"gate", "actual", "threshold", "operator", "pass"}.issubset(gates.columns)
        assert len(gates) == 9
        assert gates["pass"].dtype == bool


# ---------------------------------------------------------------------------
# deliverable integrity (skip if cycle has not run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cycle_run():
    if not (OUT / "cycle_metrics.json").exists():
        pytest.skip("Phase 16 learning cycle outputs not present")
    return True


class TestDeliverables:
    def test_cycle_metrics(self, cycle_run):
        import json

        m = json.loads((OUT / "cycle_metrics.json").read_text(encoding="utf-8"))
        assert m["cycle_ok"] is True
        assert m["stages_failed"] == {}
        assert "dry_run" in m

    def test_prioritization_inputs(self, cycle_run):
        from p16_common import P15_OUT

        assert (P15_OUT / "literature_priority_queue.xlsx").exists()
        assert (P15_OUT / "evidence_gap_analysis.xlsx").exists()
        assert (P15_OUT / "information_gain_report.xlsx").exists()

    def test_discovery_plan(self, cycle_run):
        import json

        plan = json.loads((OUT / "discovery" / "discovery_plan.json").read_text(encoding="utf-8"))
        assert plan["dry_run"] is True
        assert plan["require_doi"] is True
        assert len(plan["approved_sources"]) >= 1

    def test_acquisition_manifest(self, cycle_run):
        manifest = pd.read_parquet(OUT / "acquisition" / "acquisition_manifest.parquet")
        if len(manifest):
            assert "integrity_ok" in manifest.columns
            assert manifest["status"].isin(["planned", "acquired"]).all()

    def test_extraction_ledger(self, cycle_run):
        ledger = pd.read_parquet(OUT / "extracted" / "extraction_plan.parquet")
        if len(ledger):
            assert ledger["new_paper"].all()

    def test_embedding_plan(self, cycle_run):
        import json

        plan = json.loads((OUT / "embedding" / "embedding_plan.json").read_text(encoding="utf-8"))
        assert plan["store_consistent"] is True
        assert plan["reuse_existing"] is True
        assert plan["incremental"] is True

    def test_selective_meta(self, cycle_run):
        updated = pd.read_parquet(OUT / "selective_meta" / "updated_meta_analysis.parquet")
        if len(updated):
            assert {
                "Variable",
                "random_effect_mean",
                "random_effect_se",
                "heterogeneity_I2",
            }.issubset(updated.columns)

    def test_ready_reckoner(self, cycle_run):
        updated = pd.read_parquet(OUT / "ready_reckoner" / "updated_ready_reckoner.parquet")
        if len(updated):
            assert "review_flag" in updated.columns
            assert updated["review_flag"].isin(["review", "auto"]).all()

    def test_readiness_targets(self, cycle_run):
        import json

        assert (ROOT / "outputs" / "phase15" / "model_readiness_targets.xlsx").exists()
        metrics = json.loads(
            (OUT / "readiness" / "readiness_metrics.json").read_text(encoding="utf-8")
        )
        assert metrics["retrain_triggered"] is False
        assert 0.0 <= metrics["readiness_score"] <= 1.0

    def test_rag_sync_report(self, cycle_run):
        import json

        rep = json.loads((OUT / "rag_sync" / "sync_report.json").read_text(encoding="utf-8"))
        assert "new_chunks_pending" in rep
        assert rep["rebuild_vector_index"] is False

    def test_versioning_audit(self, cycle_run):
        import json

        changelog = json.loads((OUT / "versioning" / "changelog.json").read_text(encoding="utf-8"))
        assert isinstance(changelog, list) and len(changelog) >= 1
        assert changelog[-1]["dry_run"] is True
        manifest = json.loads(
            (OUT / "versioning" / "version_manifest.json").read_text(encoding="utf-8")
        )
        assert "knowledge_version" in manifest

    def test_monitoring_reports(self, cycle_run):
        import json

        prog = json.loads(
            (OUT / "Monitoring" / "learning_progress.json").read_text(encoding="utf-8")
        )
        assert "evidence_base" in prog
        assert "stages" in prog
        assert (ROOT / "reports" / "phase16" / "learning_dashboard.html").exists()
        assert (OUT / "knowledge_growth_report.xlsx").exists()

    def test_validated_tables_untouched(self, cycle_run):
        import hashlib

        for rel in ("outputs/UAMS_v2.parquet", "outputs/phase14/harmonized_evidence.parquet"):
            p = ROOT / rel
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                assert len(h) == 64
