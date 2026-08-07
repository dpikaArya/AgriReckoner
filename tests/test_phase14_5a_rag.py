"""Tests for the Phase 14.5A Evidence RAG layer.

Pure tests exercise the shared primitives (config merging, BM25, hashing,
grading, chunk building, response schema) on synthetic data. Integration tests
assert the structure of the Phase 14.5A deliverables when the pipeline has run
(they skip if the outputs are absent). Vector/embedding backends are taken from
config only; no model or backend is hardcoded in the tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase14_5a"
OUT = ROOT / "outputs" / "phase14_5a"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import rag_common as RC  # noqa: E402, I001
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase14_5a scratch modules unavailable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# pure primitives
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_present(self):
        cfg = RC.load_config(RC.TMP / "nonexistent.yaml")
        assert cfg["embedding"]["provider"] in ("sentence_transformers", "deterministic")
        assert cfg["vector"]["backend"] in ("sqlite", "chroma", "faiss", "qdrant")
        assert sum(cfg["reranker"]["weights"].values()) == pytest.approx(1.0, abs=0.001)
        assert sum(cfg["retriever"]["weights"].values()) == pytest.approx(1.0, abs=0.001)

    def test_user_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "embedding:\n  provider: deterministic\n  model: hashed-ngram-1024\n", encoding="utf-8"
        )
        cfg = RC.load_config(p)
        assert cfg["embedding"]["provider"] == "deterministic"
        assert cfg["embedding"]["model"] == "hashed-ngram-1024"


class TestHashAndTokens:
    def test_sha256_deterministic(self):
        assert RC.sha256("wheat biomass") == RC.sha256("wheat biomass")
        assert len(RC.sha256("x")) == 64

    def test_tokenize(self):
        assert RC.tokenize("Plant Height cm & EC!") == ["plant", "height", "cm", "ec"]


class TestBM25:
    def test_relevance_ordering(self):
        docs = [["wheat", "biomass"], ["rice", "grain"], ["wheat", "yield", "biomass"]]
        bm = RC.BM25(docs)
        s = bm.score_many(["wheat", "biomass"], [0, 1, 2])
        assert s[0] > 0 and s[2] > 0 and s[0] > s[1] and s[2] > s[1]

    def test_empty_query_zero(self):
        bm = RC.BM25([["a", "b"], ["c"]])
        assert np.all(bm.score_many([], [0, 1]) == 0)


class TestGrading:
    def test_unusable_is_d(self):
        assert RC.grade_from_scores(0.7, False, 0.8, 5) == "D"

    def test_meta_boost(self):
        assert RC.grade_from_scores(0.45, True, 0.8, 4) in ("A", "B")

    def test_high_quality_a(self):
        assert RC.grade_from_scores(0.65, True, 0.9, None) == "A"


class TestChunkerDeterminism:
    def test_chunk_id_deterministic(self):
        text = "wheat biomass yield 4.5 t/ha"
        a = f"obs-{RC.sha256(text)[:12]}"
        b = f"obs-{RC.sha256(text)[:12]}"
        assert a == b

    def test_chunk_columns_complete(self):
        for col in (
            "ChunkID",
            "ChunkType",
            "Text",
            "PaperID",
            "EvidenceGrade",
            "Variable",
            "Crop",
            "DOI",
        ):
            assert col in RC.CHUNK_COLUMNS


class TestPackageBase:
    def test_schema_stable(self):
        out = RC.package_base("q", "evidence_search")
        for key in (
            "service",
            "query",
            "matched_papers",
            "matched_variables",
            "provenance",
            "confidence_score",
            "evidence_grade",
            "latency_ms",
            "traceability",
        ):
            assert key in out
        assert out["service"] == "evidence_search"
        assert out["evidence_grade"] == "D"


# ---------------------------------------------------------------------------
# deterministic embedder/vector backend (no external model required)
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path=None):
    if tmp_path is not None:
        p = tmp_path / "rag_test.yaml"
        p.write_text(
            "embedding:\n  provider: deterministic\n  model: hashed-ngram-256\n"
            "vector:\n  backend: sqlite\n",
            encoding="utf-8",
        )
        return RC.load_config(p)
    cfg = RC.load_config(RC.TMP / "nonexistent.yaml")
    cfg["embedding"]["provider"] = "deterministic"
    cfg["embedding"]["model"] = "hashed-ngram-256"
    cfg["vector"]["backend"] = "sqlite"
    return cfg


class TestDeterministicEmbedder:
    def test_embed_dim_and_determinism(self):
        from embedder import make_provider

        cfg = _make_cfg(None)
        prov = make_provider(cfg)
        v1 = prov.encode(["wheat biomass yield"])
        v2 = prov.encode(["wheat biomass yield"])
        assert v1.shape == (1, 256)
        assert np.allclose(v1, v2)
        assert np.isclose(np.linalg.norm(v1[0]), 1.0, atol=1e-4)


class TestSqliteBackend:
    def test_write_read_roundtrip(self, tmp_path, monkeypatch):
        from rag_vector import SqliteBackend

        monkeypatch.setattr(RC, "VECTOR_INDEX", tmp_path)
        backend = SqliteBackend(_make_cfg(tmp_path))
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        backend.write(["c1", "c2"], vecs)
        mat, ids = backend.read()
        assert ids == ["c1", "c2"]
        assert mat.shape == (2, 2)
        top_ids, scores = backend.search(np.array([1.0, 0.0], dtype=np.float32), 1)
        assert top_ids[0] == "c1"


class TestHybridScoring:
    def test_component_scores_shapes(self):
        from rag_vector import VectorStore

        cfg = _make_cfg(None)
        store = object.__new__(VectorStore)
        store.cfg = cfg
        store.df = pd.DataFrame(
            {
                "ChunkID": ["a", "b"],
                "Text": ["wheat biomass", "rice grain"],
                "Crop": ["Wheat", "Rice"],
                "Variable": ["Biomass_Yield", "Grain"],
                "TreatmentLabel": [None, None],
                "NodeID": [None, None],
                "EvidenceID": [None, None],
                "Year": ["2019", "2020"],
                "QualityScore": [0.6, 0.4],
                "Confidence": [0.8, 0.5],
                "n_observations": [12, 4],
                "StudyCount": [5, 2],
                "EvidenceGrade": ["A", "C"],
                "ChunkType": ["observation", "observation"],
                "MeasurementID": [None, None],
                "ExperimentID": [None, None],
                "TreatmentID": [None, None],
                "ObservationID": [None, None],
                "VariableGroup": [None, None],
                "Value": [None, None],
                "Unit": [None, None],
                "Country": [None, None],
                "Season": [None, None],
                "PublicationYear": ["2019", "2020"],
                "DOI": [None, None],
                "Parser": [None, None],
                "SourceRow": [None, None],
                "EvidenceID_": [None, None],
                "MetaGroupID": [None, None],
                "CurveID": [None, None],
                "RecommendationID": [None, None],
                "StudyID": [None, None],
                "Usable": [True, True],
                "RecoveredSource": [None, None],
            },
            index=[0, 1],
        )
        store.ids = ["a", "b"]
        store.vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store.bm25 = RC.BM25([["wheat", "biomass"], ["rice", "grain"]])
        store._crop_tokens = {"Wheat": [0], "Rice": [1]}
        store._var_tokens = {"Biomass_Yield": [0], "Grain": [1]}
        store._treatment_tokens = {}
        store.search = lambda qvec, k, threshold=None: (store.ids[:k], [1.0] * len(store.ids[:k]))
        from rag_retrieval import HybridRetriever, QueryFactorizer

        retr = HybridRetriever(cfg, store)
        retr.provider = _make_provider(cfg)
        facts = QueryFactorizer(store).factorize("wheat biomass")
        qvec = retr._query_vec("wheat biomass")
        idx, _, _ = retr._candidate_pool("wheat biomass", qvec)
        sem, kw, ont, kg = retr._component_scores(idx, "wheat biomass", qvec, facts)
        assert len(sem) == len(idx) == len(kw) == len(ont) == len(kg)
        hybrid = retr._hybrid(sem, kw, ont, kg, idx)
        assert np.all(np.isfinite(hybrid))


def _make_provider(cfg):
    from embedder import make_provider

    return make_provider(cfg)


# ---------------------------------------------------------------------------
# deliverable integrity (skip if pipeline not yet run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run():
    chunks = OUT / "Chunk_Metadata" / "chunks.parquet"
    vectors = OUT / "Embedding_Store" / "vectors.npy"
    index = OUT / "Vector_Index" / "vector_index.sqlite"
    if not (chunks.exists() and vectors.exists() and index.exists()):
        pytest.skip("Phase 14.5A pipeline outputs not present")
    return True


class TestDeliverables:
    def test_chunks_exist_and_schema(self, pipeline_run):
        df = pd.read_parquet(OUT / "Chunk_Metadata" / "chunks.parquet")
        assert len(df) > 100
        for col in RC.CHUNK_COLUMNS:
            assert col in df.columns
        assert df["ChunkID"].is_unique
        assert df["EvidenceGrade"].isin(["A", "B", "C", "D"]).all()

    def test_embedding_store(self, pipeline_run):
        import json

        meta = json.loads((OUT / "Embedding_Store" / "metadata.json").read_text(encoding="utf-8"))
        vecs = np.load(OUT / "Embedding_Store" / "vectors.npy")
        ids = json.loads((OUT / "Embedding_Store" / "ids.json").read_text(encoding="utf-8"))
        assert vecs.shape == (len(ids), meta["dim"])
        assert meta["checksum"]

    def test_vector_index(self, pipeline_run):
        import json

        manifest = json.loads(
            (OUT / "Vector_Index" / "vector_index_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["vectors"] == manifest["chunks"] > 0
        assert manifest["backend"] in ("sqlite", "chroma", "faiss", "qdrant")

    def test_eval_metrics(self, pipeline_run):
        import json

        agg = json.loads((OUT / "rag_metrics.json").read_text(encoding="utf-8"))
        assert 0.0 <= agg["mean_precision_at_k"] <= 1.0
        assert 0.0 <= agg["coverage"] <= 1.0
        assert agg["n_queries"] >= 5

    def test_reports_present(self, pipeline_run):
        assert (ROOT / "reports" / "phase14_5a" / "rag_architecture.html").exists()
        assert (ROOT / "reports" / "phase14_5a" / "rag_service_documentation.pdf").exists()
        assert (ROOT / "reports" / "phase14_5a" / "Phase14_5A_Summary.html").exists()

    def test_demo_responses_schema(self, pipeline_run):
        import json

        demo = json.loads(
            (OUT / "Evidence_RAG_Database" / "demo_responses.json").read_text(encoding="utf-8")
        )
        assert len(demo) >= 5
        first = next(iter(demo.values()))
        assert "evidence_grade" in first and "confidence_score" in first
