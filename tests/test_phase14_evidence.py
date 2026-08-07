"""Tests for the Phase 14 evidence-integration statistics and pipeline outputs.

The pure-statistics tests exercise the helpers in ``p14_common`` on synthetic
data; the integrity tests assert the structure of the Phase 14 deliverables when
the pipeline has been run (they skip if the outputs are absent).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP14 = ROOT / ".opencode_tmp" / "phase14"
OUT14 = ROOT / "outputs" / "phase14"
REP14 = ROOT / "reports" / "phase14"

pytest.importorskip("statsmodels")
pytest.importorskip("scipy")
try:
    if str(TMP14) not in sys.path:
        sys.path.insert(0, str(TMP14))
    import p14_common as Q  # noqa: E402, I001
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase14 scratch modules unavailable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# pure statistics helpers
# ---------------------------------------------------------------------------


class TestParseTreatmentDoses:
    def test_organic_dose(self):
        d = Q.parse_treatment_doses("Compost @ 5 ton ha-1 + EM + Bordeaux")
        assert d.get("Organic") == 5.0

    def test_fertilizer_dose(self):
        d = Q.parse_treatment_doses("N 120 P 60 K 40 kg/ha")
        assert d.get("N") == 120.0
        assert d.get("P") == 60.0
        assert d.get("K") == 40.0

    def test_irrigation_dose(self):
        d = Q.parse_treatment_doses("450 mm irrigation")
        assert d.get("Irrigation") == 450.0

    def test_empty_label(self):
        assert Q.parse_treatment_doses("") == {}


class TestValuePlausible:
    def test_negative_impossible_variable_rejected(self):
        ok, _ = Q.value_plausible("Phosphorus", -12.0)
        assert ok is False

    def test_yield_range(self):
        ok, _ = Q.value_plausible("Yield_per_Hectare", 4500.0)
        assert ok is True


class TestMetaCombine:
    def test_basic_combine(self):
        studies = [
            {"n": 6, "mean": 10.0, "sd": 2.0},
            {"n": 6, "mean": 12.0, "sd": 2.0},
            {"n": 6, "mean": 11.0, "sd": 2.0},
        ]
        out = Q.meta_combine(studies)
        assert out is not None
        assert out["k_studies"] == 3
        assert out["n_observations"] == 18
        assert (
            out["random_effect_ci_lower"]
            <= out["random_effect_mean"]
            <= out["random_effect_ci_upper"]
        )
        assert 0.0 <= out["heterogeneity_I2"] <= 1.0
        assert "effect_size_g" in out

    def test_too_few_studies(self):
        assert Q.meta_combine([{"n": 2, "mean": 1.0, "sd": 0.1}]) is None


class TestResponseCurve:
    def test_linear_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 + 3.0 * x
        coeff, band = Q.response_curve(x, y)
        assert coeff is not None
        assert coeff["model"] == "linear"
        assert coeff["slope"] == pytest.approx(3.0, abs=1e-4)
        assert coeff["intercept"] == pytest.approx(2.0, abs=1e-4)
        assert abs(coeff["r2"] - 1.0) < 1e-6
        assert list(band.columns) == ["x", "y_pred", "ci_lower", "ci_upper"]
        assert (band["ci_lower"] <= band["y_pred"]).all()
        assert (band["y_pred"] <= band["ci_upper"]).all()

    def test_insufficient_points(self):
        coeff, band = Q.response_curve(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert coeff is None
        assert band is None

    def test_constant_x_rejected(self):
        coeff, band = Q.response_curve(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
        assert coeff is None


# ---------------------------------------------------------------------------
# deliverable integrity (skips when pipeline outputs are missing)
# ---------------------------------------------------------------------------


def _require(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Phase 14 outputs not generated (missing: {missing})")


def test_meta_analysis_output_structure():
    _require(OUT14 / "meta_analysis_results.parquet", OUT14 / "effect_size_database.parquet")
    meta = pd.read_parquet(OUT14 / "meta_analysis_results.parquet")
    es = pd.read_parquet(OUT14 / "effect_size_database.parquet")
    for col in (
        "Crop",
        "Variable",
        "PublicationCount",
        "random_effect_mean",
        "heterogeneity_I2",
        "effect_size_g",
        "SupportingEvidenceIDs",
    ):
        assert col in meta.columns
    assert (meta["PublicationCount"] >= 3).all()
    assert len(es) >= len(meta)


def test_response_curves_output_structure():
    _require(OUT14 / "response_curves.parquet")
    curves = pd.read_parquet(OUT14 / "response_curves.parquet")
    for col in ("CurveID", "DoseVariable", "ResponseVariable", "Status", "N_points", "N_studies"):
        assert col in curves.columns
    fit = curves[curves["Status"] != "insufficient_data"]
    if len(fit):
        assert (fit["N_points"] >= 3).all()


def test_ready_reckoner_traceability():
    _require(OUT14 / "ready_reckoner_knowledge_base.parquet")
    rr = pd.read_parquet(OUT14 / "ready_reckoner_knowledge_base.parquet")
    for col in (
        "RecommendationID",
        "EvidenceType",
        "Variable",
        "ConfidenceLevel",
        "SupportingStudies",
    ):
        assert col in rr.columns
    ve = rr[rr["EvidenceType"] == "variable_estimate"]
    if len(ve):
        assert (ve["SupportingStudies"].astype(str) != "").all()
        assert (ve["SupportingEvidenceIDs"].astype(str) != "").all()


def test_feature_validation_output_structure():
    _require(REP14 / "feature_validation_report.xlsx", OUT14 / "feature_validation.parquet")
    feat = pd.read_parquet(OUT14 / "feature_validation.parquet")
    for col in ("Feature", "suitability", "best_target", "best_abs_r", "best_p", "n_papers"):
        assert col in feat.columns
    assert set(feat["suitability"]) <= {"recommended", "promising", "limited_evidence"}


def test_study_quality_output_structure():
    _require(REP14 / "study_quality_scores.xlsx", OUT14 / "study_quality_scores.parquet")
    sq = pd.read_parquet(OUT14 / "study_quality_scores.parquet")
    assert "quality_score" in sq.columns
    assert sq["quality_score"].between(0, 1).all()


def test_knowledge_graph_output_structure():
    _require(OUT14 / "knowledge_graph.json", OUT14 / "knowledge_graph_edges.parquet")
    import json

    with open(OUT14 / "knowledge_graph.json", encoding="utf-8") as fh:
        graph = json.load(fh)
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
    node_ids = {n["id"] for n in graph["nodes"]}
    for e in graph["edges"]:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


def test_metrics_json_present():
    _require(OUT14 / "Phase14_Metrics.json")
    import json

    with open(OUT14 / "Phase14_Metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["pipeline"] == "phase14"
    assert metrics["evidence"]["usable_rows"] > 0
    assert metrics["meta_analysis"]["groups"] > 0


def test_no_validated_observation_modification():
    """Recovered rows must be additive: original model observation count is intact."""
    _require(OUT14 / "harmonized_evidence.parquet")
    ev = pd.read_parquet(OUT14 / "harmonized_evidence.parquet")
    assert "EvidenceID" in ev.columns
    src = ev["Source"].astype(str)
    n_model = int((src == "model").sum())
    assert n_model == 2320, f"model observations changed: {n_model} != 2320"
    n_rec = int(src.str.startswith("recovered").sum())
    assert n_rec > 0
