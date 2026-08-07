"""Tests for the Phase 14.5B Evidence Verification Layer.

Pure tests exercise config-driven primitives (config merging/weight
normalisation, identifier cleaning, duplicate detection, claim reference
splitting) on synthetic data. Integration tests assert the structure of the
Phase 14.5B deliverables when the pipeline has run (they skip if the outputs
are absent). The verification layer never modifies validated observations,
UAMS records or extraction pipelines.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase14_5b"
OUT = ROOT / "outputs" / "phase14_5b"
VERIFIED = OUT / "Verified_Evidence"
CHAINS = OUT / "Evidence_Chain"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import v14_5b_common as VC  # noqa: E402, I001
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase14_5b scratch modules unavailable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# pure primitives
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_present(self):
        cfg = VC.load_config(TMP / "nonexistent.yaml")
        assert "chain" in cfg["verification"]
        assert cfg["verification"]["chain"]["min_completeness"] > 0
        assert cfg["verification"]["citation"]["target_accuracy"] == 0.95
        w = cfg["verification"]["confidence"]["weights"]
        assert sum(w.values()) == pytest.approx(1.0, abs=0.001)
        assert set(w) == {
            "scientific",
            "retrieval",
            "extraction",
            "study_quality",
            "meta_weight",
            "observation_completeness",
            "provenance_completeness",
        }

    def test_user_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("verification:\n  chain:\n    min_completeness: 0.9\n", encoding="utf-8")
        cfg = VC.load_config(p)
        assert cfg["verification"]["chain"]["min_completeness"] == 0.9
        assert cfg["verification"]["citation"]["target_accuracy"] == 0.95


class TestCleaners:
    def test_clean_str_lowercases(self):
        assert VC.clean_str("PAPER_AbC") == "paper_abc"
        assert VC.clean_str(float("nan")) == ""

    def test_clean_id_preserves_case(self):
        assert VC.clean_id("PAPER_AbC") == "PAPER_AbC"
        assert VC.clean_id(" None ") == ""
        assert VC.clean_id(float("nan")) == ""


class TestDuplicates:
    def test_dedupe_key_stable(self):
        from v14_5b_duplicates import DuplicateDetector

        det = DuplicateDetector(VC.load_config(TMP / "nonexistent.yaml"))
        r1 = {
            "PaperID": "PAPER_X",
            "Variable": "Plant Height",
            "Value": 4.5678,
            "Unit": " cm ",
            "Crop": "Wheat",
        }
        r2 = {
            "PaperID": "PAPER_X",
            "Variable": "PlantHeight",
            "Value": 4.56781,
            "Unit": "cm",
            "Crop": "Wheat",
        }
        assert det.dedupe_key(r1) == det.dedupe_key(r2)

    def test_detect_groups_and_canonical(self):
        from v14_5b_duplicates import DuplicateDetector

        det = DuplicateDetector(VC.load_config(TMP / "nonexistent.yaml"))
        df = pd.DataFrame(
            [
                {
                    "ChunkID": "a",
                    "ChunkType": "observation",
                    "PaperID": "PAPER_X",
                    "Variable": "Plant Height",
                    "Value": 4.5,
                    "Unit": "cm",
                    "Crop": "Wheat",
                    "EvidenceGrade": "B",
                    "Confidence": 0.7,
                },
                {
                    "ChunkID": "b",
                    "ChunkType": "measurement",
                    "PaperID": "PAPER_X",
                    "Variable": "PlantHeight",
                    "Value": 4.5,
                    "Unit": "cm",
                    "Crop": "Wheat",
                    "EvidenceGrade": "A",
                    "Confidence": 0.9,
                },
            ]
        )
        out = det.detect(df)
        assert len(out) == 2
        assert (out["is_canonical"] == [True, False]).all()
        assert out.loc[1, "duplicate_of"] == "b"


class TestClaims:
    def test_split_refs(self):
        from v14_5b_claims import _split_refs

        assert _split_refs("EV-001;EV-002;") == ["EV-001", "EV-002"]
        assert _split_refs(" nan ") == []


class TestStatsHelpers:
    def test_ci_order_ok_and_fail(self):
        from v14_5b_stats import StatisticalValidator

        cfg = VC.load_config(TMP / "nonexistent.yaml")
        v = StatisticalValidator(cfg)
        assert v._ci_order(1.0, 2.0, 3.0, "t", "k", "x")
        assert not v._ci_order(3.0, 2.0, 1.0, "t", "k", "x")
        flags = pd.DataFrame(v.rows)
        assert (flags["status"] == "fail").any()


# ---------------------------------------------------------------------------
# deliverable integrity (skip if pipeline not yet run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run():
    if not (OUT / "verification_metrics.json").exists():
        pytest.skip("Phase 14.5B pipeline outputs not present")
    return True


class TestDeliverables:
    def test_metrics_present(self, pipeline_run):
        import json

        m = json.loads((OUT / "verification_metrics.json").read_text(encoding="utf-8"))
        for key in (
            "chain",
            "provenance",
            "duplicates",
            "statistics",
            "claims",
            "confidence",
            "citation_accuracy",
        ):
            assert key in m
        assert 0.0 <= m["citation_accuracy"]["accuracy"] <= 1.0
        assert m["citation_accuracy"]["target"] == 0.95

    def test_chain_validation(self, pipeline_run):
        import json

        chains = pd.read_parquet(CHAINS / "evidence_chains.parquet")
        assert len(chains) > 100
        assert set(["ChunkID", "chain_status", "completeness", "traceable"]).issubset(
            chains.columns
        )
        assert chains["chain_status"].isin(["valid", "incomplete", "inconsistent"]).all()
        status = json.loads((CHAINS / "chain_status.json").read_text(encoding="utf-8"))
        assert status["inconsistent"] == 0

    def test_provenance(self, pipeline_run):
        prov = pd.read_parquet(VERIFIED / "provenance_verification.parquet")
        assert "provenance_completeness" in prov.columns
        assert prov["provenance_completeness"].between(0, 1).all()

    def test_duplicates_never_delete(self, pipeline_run):
        import json

        chunks = pd.read_parquet(
            ROOT / "outputs" / "phase14_5a" / "Chunk_Metadata" / "chunks.parquet"
        )
        pd.read_parquet(VERIFIED / "canonical_evidence.parquet")
        dups = json.loads((VERIFIED / "duplicate_metrics.json").read_text(encoding="utf-8"))
        assert dups["canonical_evidence_records"] <= len(chunks)
        assert dups["groups"] >= 0

    def test_statistical_validation(self, pipeline_run):
        flags = pd.read_parquet(VERIFIED / "statistical_validation.parquet")
        assert {"source", "record_key", "field", "status"}.issubset(flags.columns)
        assert flags["status"].isin(["ok", "fail", "na"]).all()

    def test_claims_supported_references(self, pipeline_run):
        claims = pd.read_parquet(VERIFIED / "verified_claims.parquet")
        assert {"artifact", "claim_id", "status", "support_coverage"}.issubset(claims.columns)
        meta = claims[claims["artifact"] == "meta_analysis"]
        assert (meta["status"] == "supported").all()

    def test_confidence_scores(self, pipeline_run):
        scores = pd.read_parquet(VERIFIED / "evidence_confidence.parquet")
        assert scores["evidence_confidence_score"].between(0, 1).all()
        assert scores["verified_grade"].isin(["A", "B", "C", "D"]).all()

    def test_reports_present(self, pipeline_run):
        assert (ROOT / "reports" / "phase14_5b" / "verification_dashboard.html").exists()
        assert (ROOT / "reports" / "phase14_5b" / "citation_accuracy_report.xlsx").exists()
        assert (ROOT / "reports" / "phase14_5b" / "duplicate_evidence_report.xlsx").exists()
        assert (ROOT / "reports" / "phase14_5b" / "unsupported_claims.xlsx").exists()
        assert (ROOT / "reports" / "phase14_5b" / "Phase14_5B_Summary.html").exists()
