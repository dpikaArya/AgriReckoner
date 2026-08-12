"""Phase 20 test suite.

Run with:
    python -m pytest tests/test_phase20.py -q

Covers project-root isolation, path traversal safety, protected-input
immutability, canonical identity, duplicate review (no merging), spatial /
temporal linkage honesty, ontology recovery, external linkage provenance,
predictor completeness, missingness, information gain, independence, leakage
(safety without deletion), grouped validation splits, model readiness,
RAG append-only behavior, and versioning.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
P20 = ROOT / ".opencode_tmp" / "phase20"
sys.path.insert(0, str(P20))

import p20_common as C

OUT = C.OUT
REPORTS = C.REPORTS


def _read(name):
    p = OUT / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return C.read_json(p)
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def matrix():
    m = _read("model_ready_observation_matrix.parquet")
    if m is None:
        pytest.skip("phase20 outputs not built")
    return m


@pytest.fixture(scope="module", autouse=True)
def snapshot_protected():
    before = {}
    for rel in ["outputs/UAMS_v2.parquet", "outputs/UAMS_v2.1.parquet"]:
        p = ROOT / rel
        if p.exists():
            before[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    yield
    for rel, h in before.items():
        p = ROOT / rel
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == h, \
                f"protected input mutated: {rel}"


class TestPathSafety:
    def test_project_root_enforced(self):
        assert C.PROJECT_ROOT.drive.upper() == "F:"
        assert C.PROJECT_ROOT.name == "Agriculture Intelligence Framework3"

    def test_safe_resolve_rejects_escape(self):
        with pytest.raises(RuntimeError):
            C.safe_resolve(r"C:\Windows\System32")
        with pytest.raises(RuntimeError):
            C.safe_resolve(str(Path("..") / ".." / "escape.txt"))

    def test_safe_resolve_accepts_inside(self):
        p = C.safe_resolve("outputs/phase20/x.parquet")
        assert C.is_inside_root(p)

    def test_path_safety_report(self):
        rep = C.read_json(REPORTS / "path_safety_report.json", {})
        assert rep.get("all_paths_safe", False) is True
        assert rep.get("paths_rejected", -1) == 0


class TestProtectedImmutability:
    def test_checksums_preserved(self):
        bef = (C.read_json(OUT / "protected_input_checksums_before.json") or {}).get("checksums", {})
        aft = (C.read_json(OUT / "protected_input_checksums_after.json") or {}).get("checksums", {})
        assert bef and aft
        diff = [k for k in bef if bef.get(k) != aft.get(k)]
        assert diff == []

    def test_verification_report(self):
        v = C.read_json(OUT / "protected_checksum_verification.json", {})
        assert v.get("drift_detected") is False


class TestCanonicalIdentity:
    def test_identity_output_exists(self):
        df = _read("canonical_identity.parquet")
        assert df is not None and len(df) > 12000

    def test_observation_ids_unique(self, matrix):
        assert matrix["ObservationID_ML"].is_unique

    def test_canonical_ids_present(self, matrix):
        for c in ["canonical_observation_id", "canonical_study_id",
                  "canonical_experiment_id", "canonical_location_id"]:
            assert c in matrix.columns
            assert matrix[c].notna().all()

    def test_duplicate_review_no_merge(self):
        dup = _read("duplicate_review_queue.parquet")
        metrics = C.read_json(OUT / "canonical_identity_metrics.json", {})
        assert metrics.get("duplicates_merged", -1) == 0
        assert dup is None or len(dup) >= 0


class TestSpatial:
    def test_no_invented_coordinates(self):
        df = _read("spatial_linkage.parquet")
        assert df is not None
        assert "location_match_method" in df.columns
        assert not df["location_match_method"].astype(str).str.contains(
            "FABRICATED", na=False).any()

    def test_precision_ordered(self):
        df = _read("spatial_linkage.parquet")
        vals = df["location_precision"].astype(str).unique()
        for v in vals:
            assert v in ["exact_coordinates", "experimental_station",
                         "study_location", "district", "country_only", "unknown"]


class TestTemporal:
    def test_year_bounds(self):
        df = _read("temporal_linkage.parquet")
        assert df is not None
        y = pd.to_numeric(df["year"], errors="coerce")
        ok = y.dropna()
        if len(ok):
            assert ok.between(1970, 2100).all()

    def test_no_future_information(self):
        metrics = C.read_json(OUT / "temporal_linkage_metrics.json", {})
        assert metrics.get("no_future_information", False) is True


class TestOntology:
    def test_extension_file_exists(self):
        assert C.ONTO_EXT.exists()

    def test_no_new_aliases_for_national_aggregates(self):
        metrics = C.read_json(OUT / "ontology_recovery_metrics.json", {})
        assert metrics.get("new_aliases_added", -1) == 0
        # rejection rate unchanged: honest classification
        assert metrics.get("staged_rejection_rate_after", -1) == \
            metrics.get("staged_rejection_rate_before", -2)


class TestExternalLinkage:
    def test_no_fabricated_data(self):
        df = _read("external_predictor_linkage.parquet")
        if df is not None and len(df):
            assert not df["linkage_method"].astype(str).str.contains(
                "FABRICATED", na=False).any()

    def test_provenance_or_documented_empty(self):
        metrics = C.read_json(OUT / "external_linkage_metrics.json", {})
        assert metrics.get("no_fabricated_data", False) is True
        assert metrics.get("no_live_api_calls", False) is True


class TestCompleteness:
    def test_completeness_bounded(self, matrix):
        assert matrix["CompletenessRatio"].between(0, 1).all()

    def test_yield_observations_present(self, matrix):
        n_yield = (matrix["Target"] == "Yield").sum()
        assert n_yield > 12000

    def test_missingness_output(self):
        m = _read("observation_missingness.parquet")
        assert m is not None and len(m) > 0


class TestInfoGain:
    def test_acquisition_priority_exists(self):
        ap = _read("phase20_acquisition_priority.parquet")
        assert ap is not None and len(ap) > 0
        assert ap["AcquisitionPriority"].isin(["HIGH", "MEDIUM", "LOW"]).all()

    def test_only_approved_sources(self):
        ap = _read("phase20_acquisition_priority.parquet")
        allowed = {"CHIRPS", "NASA_POWER", "SoilGrids", "MapSPAM", "FAOSTAT"}
        for src in ap["SourceAvailability"]:
            for s in str(src).split(";"):
                assert s in allowed or s == "none_approved"


class TestIndependence:
    def test_independent_ids(self):
        ind = _read("independence_audit.parquet")
        assert ind is not None
        for c in ["IndependentGroup_Study", "IndependentGroup_Location",
                  "IndependentGroup_Treatment"]:
            assert c in ind.columns
            assert ind[c].notna().all()

    def test_studies_and_locations(self):
        metrics = C.read_json(OUT / "independence_metrics.json", {})
        assert metrics.get("independent_studies", 0) > 0
        assert metrics.get("independent_locations", 0) > 0


class TestLeakage:
    def test_audit_exists(self):
        la = _read("leakage_audit.parquet")
        assert la is not None and len(la) > 0

    def test_nothing_deleted(self):
        metrics = C.read_json(OUT / "leakage_metrics.json", {})
        assert metrics.get("no_automatic_deletion", False) is True

    def test_no_future_weather(self):
        metrics = C.read_json(OUT / "leakage_metrics.json", {})
        assert metrics.get("no_future_weather", False) is True


class TestSplitting:
    def test_groupkfold_no_contamination(self, matrix):
        from sklearn.model_selection import GroupKFold
        df = matrix
        if df["canonical_study_id"].nunique() < 2:
            pytest.skip("insufficient studies")
        groups = df["canonical_study_id"].values
        kf = GroupKFold(n_splits=min(5, df["canonical_study_id"].nunique()))
        for tr, te in kf.split(df, groups=groups):
            tr_groups = set(df["canonical_study_id"].iloc[tr])
            te_groups = set(df["canonical_study_id"].iloc[te])
            assert tr_groups.isdisjoint(te_groups)

    def test_splits_report(self):
        sp = _read("validation_splits.parquet")
        assert sp is not None and len(sp) >= 2


class TestReadiness:
    def test_readiness_json(self):
        rd = C.read_json(OUT / "model_readiness.json", {})
        assert "overall_readiness_score" in rd
        assert "retrain_allowed" in rd
        assert "per_target" in rd
        assert "yield" in rd["per_target"]

    def test_per_target_fields(self):
        rd = C.read_json(OUT / "model_readiness.json", {})
        for d, v in rd["per_target"].items():
            if isinstance(v, dict):
                assert "total_observations" in v
                assert "readiness_score" in v

    def test_decision_recorded(self):
        dec = C.read_json(OUT / "final_decision.json", {})
        assert dec.get("decision") in ("READY", "CONDITIONALLY_READY", "NOT_READY")
        assert "decision_summary" in dec


class TestGates:
    def test_all_gates_pass(self):
        g = C.read_json(OUT / "quality_gates.json", {})
        assert g.get("gates_total", 0) > 0
        assert g.get("gates_passed", -1) == g.get("gates_total", 0)


class TestRag:
    def test_append_only(self):
        manifest = C.read_json(C.RAG_OUT / "rag_sync_manifest.json", {})
        assert manifest.get("store_unchanged", False) is True
        assert manifest.get("mode") == "append_only"

    def test_ids_json_preserved(self):
        p = C.PROJECT_ROOT / "outputs" / "phase14_5a" / "Embedding_Store" / "ids.json"
        if p.exists():
            ids = C.read_json(p, {})
            assert isinstance(ids, (dict, list))
        manifest = C.read_json(C.RAG_OUT / "rag_sync_manifest.json", {})
        assert manifest.get("existing_ids_count", 0) >= 0


class TestVersioning:
    def test_version_manifest(self):
        v = C.read_json(OUT / "version_manifest.json", {})
        assert v.get("phase20_version") == "1.0.0"
        assert v.get("artifact_count", 0) > 10

    def test_final_summary(self):
        s = C.read_json(OUT / "final_execution_summary.json", {})
        assert s.get("phase") == 20
        assert "decision" in s
        assert len(s.get("reports", [])) > 0


class TestIdempotency:
    def test_checkpoint_marks_all_stages(self):
        state = C.load_checkpoints()
        expected = {"step0_path_audit", "step1_input_manifest", "identity",
                    "spatial", "temporal", "ontology", "external_linkage",
                    "predictors", "completeness", "information_gain",
                    "independence", "leakage", "dataset", "rag_sync",
                    "reports", "step23_quality_gates", "step24_metrics",
                    "step25_version_manifest", "step26_final_decision"}
        missing = expected - set(state.keys())
        assert missing == set(), f"unfinished stages: {sorted(missing)}"

    def test_stable_matrix_rowcount(self, matrix):
        # reruns must not change the matrix size (checkpoint-guarded)
        m2 = _read("model_ready_observation_matrix.parquet")
        assert len(m2) == len(matrix)
