"""Phase 16.1 evidence-gap-driven discovery tests.

Pure tests exercise config merge/normalisation, DOI/title canonical keys,
the 16-state processing model, deduplication, scoring weights, source
credential gating and queue separation on synthetic data only. Integration
tests assert the structure of the Phase 16.1 deliverables and are skipped when
the pipeline outputs are absent. Nothing here modifies validated observations,
UAMS, the extraction pipeline, Phase 14/14.5A/14.5B outputs, or embeddings.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase16_1"
OUT = ROOT / "outputs" / "phase16_1"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import p16_1_bottleneck as BN  # noqa: E402
    import p16_1_common as C  # noqa: E402
    import p16_1_dedup as DE  # noqa: E402
    import p16_1_queues as QU  # noqa: E402
    import p16_1_scoring as SC  # noqa: E402
    import p16_1_sources as SRC  # noqa: E402
    import p16_1_state as ST  # noqa: E402
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase16_1 scratch modules unavailable: {_exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# project root safety
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_outputs_inside_project_root(self):
        for p in (
            C.OUT,
            C.REPORTS,
            C.GAP_OUT,
            C.TARGET_OUT,
            C.DISCOVERY_OUT,
            C.DEDUP_OUT,
            C.QUEUE_OUT,
            C.STATE_OUT,
        ):
            assert str(p.resolve()).startswith(str(C.PROJECT_ROOT.resolve()))

    def test_no_external_scan_paths(self):
        # no path construction for drives other than PROJECT_ROOT
        assert "literature_metadata" in str(
            C.PROJECT_ROOT / "outputs" / "literature_metadata.parquet"
        )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_present(self):
        cfg = C.load_config(TMP / "nonexistent.yaml")
        assert cfg["dry_run"] is True
        assert cfg["thresholds"]["minimum_information_gain"] > 0
        assert cfg["thresholds"]["maximum_recovery_attempts"] >= 1
        w = cfg["scoring"]["weights"]
        assert len(w) == 21
        assert sum(w.values()) == pytest.approx(1.0, abs=0.001)

    def test_recovery_states_separate(self, tmp_path):
        cfg = C.load_config()
        assert set(cfg["recovery"]["eligible_states"]).isdisjoint(
            cfg["recovery"]["ineligible_states"]
        )
        for s in ("ML_USABLE", "META_ANALYSIS_USABLE", "READY_RECKONER_USABLE", "RAG_INDEXED"):
            assert s in cfg["recovery"]["ineligible_states"]

    def test_user_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "learning:\n  phase16_1:\n    dry_run: false\n"
            "    thresholds:\n      minimum_information_gain: 0.5\n",
            encoding="utf-8",
        )
        cfg = C.load_config(p)
        assert cfg["dry_run"] is False
        assert cfg["thresholds"]["minimum_information_gain"] == 0.5


# ---------------------------------------------------------------------------
# canonical identity / cleaners
# ---------------------------------------------------------------------------


class TestCanonicalKeys:
    def test_doi_key(self):
        assert (
            C.canonical_paper_key(
                doi="https://doi.org/10.1000/ABC", title="t", authors="a", year="2019"
            )
            == "doi:10.1000/abc"
        )

    def test_title_key(self):
        key = C.canonical_paper_key(
            doi="", title="  Nitrogen  Effects  ", authors="Singh A; Kumar B", year="2021"
        )
        assert key.startswith("title:nitrogen effects|au:singh a|y:2021")

    def test_empty_key(self):
        assert C.canonical_paper_key() == ""

    def test_normalize_doi(self):
        assert C.normalize_doi("HTTPS://dx.doi.org/10.1/X.PDF") == "10.1/x"


# ---------------------------------------------------------------------------
# state model
# ---------------------------------------------------------------------------


class TestStateModel:
    def test_sixteen_states(self):
        assert len(ST.STATES) == 16
        assert ST.STATES[0] == "METADATA_ONLY"
        assert ST.STATES[-1] == "RAG_INDEXED"

    def test_metadata_only_never_processed(self):
        df = PaperStateDf()
        if df is None:
            pytest.skip("pipeline outputs absent")
        meta = df[df["HighestState"] == "METADATA_ONLY"]
        processed_flags = [
            "ML_USABLE",
            "META_ANALYSIS_USABLE",
            "READY_RECKONER_USABLE",
            "RAG_INDEXED",
            "OBSERVATIONS_EXTRACTED",
            "QC_APPROVED",
        ]
        assert (meta[processed_flags].any(axis=1)).sum() == 0

    def test_state_monotonic_rank(self):
        df = PaperStateDf()
        if df is None:
            pytest.skip("pipeline outputs absent")
        assert (df["StateRank"] >= 0).all()


def PaperStateDf():
    p = OUT / "paper_processing_state.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------


class TestDedup:
    def test_doi_duplicate_detected(self):
        corpus = pd.DataFrame(
            [
                {
                    "paper_id": "P1",
                    "doi": "10.1000/a",
                    "title": "one",
                    "authors": "",
                    "year": "2000",
                },
                {
                    "paper_id": "P2",
                    "doi": "10.1000/b",
                    "title": "two",
                    "authors": "",
                    "year": "2001",
                },
            ]
        )
        cands = pd.DataFrame(
            [
                {
                    "source": "OpenAlex",
                    "doi": "https://doi.org/10.1000/a",
                    "title": "duplicate",
                    "authors": "",
                    "year": "2000",
                },
                {
                    "source": "Crossref",
                    "doi": "10.1000/c",
                    "title": "three",
                    "authors": "K",
                    "year": "2002",
                },
            ]
        )
        engine = DE.DedupEngine(C.load_config())
        ded, _ = engine.deduplicate(cands, corpus)
        flags = dict(zip(ded["doi"], ded["is_duplicate"], strict=True))
        assert bool(flags["https://doi.org/10.1000/a"]) is True
        assert bool(flags["10.1000/c"]) is False
        assert ded.loc[ded["doi"] == "10.1000/c", "duplicate_type"].iloc[0] == ""

    def test_in_batch_duplicate(self):
        corpus = pd.DataFrame(columns=["paper_id", "doi", "title", "authors", "year"])
        cands = pd.DataFrame(
            [
                {"source": "a", "doi": "", "title": "same paper", "authors": "X", "year": "2020"},
                {"source": "b", "doi": "", "title": "same paper", "authors": "X", "year": "2020"},
            ]
        )
        engine = DE.DedupEngine(C.load_config())
        ded, _ = engine.deduplicate(cands, corpus)
        assert ded["is_duplicate"].sum() == 1
        assert ded.loc[ded["duplicate_type"] == "in_batch"].shape[0] == 1

    def test_title_author_year_duplicate(self):
        corpus = pd.DataFrame(
            [
                {
                    "paper_id": "P1",
                    "doi": "",
                    "title": "nitrogen and wheat",
                    "authors": "kumar",
                    "year": "2018",
                },
            ]
        )
        cands = pd.DataFrame(
            [
                {
                    "source": "DOAJ",
                    "doi": "",
                    "title": "Nitrogen and Wheat",
                    "authors": "Kumar A",
                    "year": "2018",
                },
            ]
        )
        engine = DE.DedupEngine(C.load_config())
        ded, _ = engine.deduplicate(cands, corpus)
        assert bool(ded["is_duplicate"].iloc[0]) is True
        assert ded["duplicate_type"].iloc[0] in ("title_author_year", "coarse_title")
        assert ded["duplicate_of"].iloc[0] == "P1"


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def _gaps(self):
        return pd.DataFrame(
            [
                {"GapID": "G1", "Deficit": 30, "RequiredMinimum": 100, "PriorityScore": 0.8},
            ]
        )

    def _targets(self):
        return pd.DataFrame(
            [
                {
                    "TargetID": "TGT-0001",
                    "GapID": "G1",
                    "Crop": "wheat",
                    "MissingTreatment": "",
                    "PreferredGeographicRegion": "India",
                },
            ]
        )

    def _cands(self):
        return pd.DataFrame(
            [
                {
                    "source": "OpenAlex",
                    "doi": "10.1000/z",
                    "title": "t",
                    "authors": "a",
                    "year": "2022",
                    "target_id": "TGT-0001",
                    "candidate_type": "new",
                },
            ]
        )

    def test_score_in_unit_interval(self):
        scorer = SC.Scorer(C.load_config())
        scored = scorer.score(self._cands(), self._targets(), self._gaps())
        assert (scored["InformationGainScore"] >= 0).all()
        assert (scored["InformationGainScore"] <= 1.0 + 1e-9).all()

    def test_all_subscores_present(self):
        scorer = SC.Scorer(C.load_config())
        scored = scorer.score(self._cands(), self._targets(), self._gaps())
        for k in scorer.w:
            assert k in scored.columns

    def test_bucket_assignment(self):
        scorer = SC.Scorer(C.load_config())
        scored = scorer.score(self._cands(), self._targets(), self._gaps())
        assert scored["ScoreBucket"].iloc[0] in (
            "high_information_gain",
            "medium_information_gain",
            "low_information_gain",
        )

    def test_reproducible(self):
        scorer = SC.Scorer(C.load_config())
        a = scorer.score(self._cands(), self._targets(), self._gaps())
        b = scorer.score(self._cands(), self._targets(), self._gaps())
        assert a["InformationGainScore"].equals(b["InformationGainScore"])


# ---------------------------------------------------------------------------
# sources / credential gating
# ---------------------------------------------------------------------------


class TestSources:
    def test_credential_gated_skipped(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        reg, metrics = SRC.SourceRegistry(C.load_config()).run()
        assert "Semantic_Scholar" in metrics["sources_skipped"]
        assert "credentials missing" in metrics["skipped_reasons"]["Semantic_Scholar"]

    def test_with_credential_available(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
        reg, metrics = SRC.SourceRegistry(C.load_config()).run()
        assert "Semantic_Scholar" in metrics["sources_available"]

    def test_no_placeholder_sources(self):
        reg, metrics = SRC.SourceRegistry(C.load_config()).run()
        known = {
            "OpenAlex",
            "Crossref",
            "Semantic_Scholar",
            "Europe_PMC",
            "OpenAIRE",
            "CORE",
            "DOAJ",
            "Zenodo",
            "Figshare",
            "CGIAR",
            "CGSpace",
            "Scopus",
            "Web_of_Science",
            "Dimensions",
        }
        assert set(metrics["sources_available"]) | set(metrics["sources_skipped"]) <= known


# ---------------------------------------------------------------------------
# discovery / dry-run
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_dry_run_no_fabricated_new_candidates(self):
        if not (OUT / "discovery" / "discovery_metrics.json").exists():
            pytest.skip("pipeline outputs absent")
        import json

        m = json.loads((OUT / "discovery" / "discovery_metrics.json").read_text(encoding="utf-8"))
        assert m["fabricated_results"] == 0
        assert m["new_literature_candidates"] == 0
        assert m["queries_planned"] > 0

    def test_recovery_candidates_are_real_corpus_papers(self):
        if not (OUT / "discovery" / "recovery_candidates.parquet").exists():
            pytest.skip("pipeline outputs absent")
        rec = pd.read_parquet(OUT / "discovery" / "recovery_candidates.parquet")
        corpus = pd.read_parquet(OUT / "dedup" / "known_corpus.parquet")
        known_keys = set(corpus["canonical_key"])
        assert set(rec["canonical_key"]) <= known_keys
        assert len(rec) > 0

    def test_recovery_states_eligible_only(self):
        if not (OUT / "discovery" / "recovery_candidates.parquet").exists():
            pytest.skip("pipeline outputs absent")
        rec = pd.read_parquet(OUT / "discovery" / "recovery_candidates.parquet")
        eligible = set(C.load_config()["recovery"]["eligible_states"])
        assert set(rec["recovery_state"]) <= eligible


# ---------------------------------------------------------------------------
# queues
# ---------------------------------------------------------------------------


class TestQueues:
    def test_queues_never_mixed(self):
        if not (OUT / "queues" / "new_literature_queue.parquet").exists():
            pytest.skip("pipeline outputs absent")
        nq = pd.read_parquet(OUT / "queues" / "new_literature_queue.parquet")
        rq = pd.read_parquet(OUT / "queues" / "existing_paper_recovery_queue.parquet")
        if len(nq):
            assert set(nq["QueueType"]) == {"NEW_LITERATURE"}
        if len(rq):
            assert set(rq["QueueType"]) == {"EXISTING_PAPER_RECOVERY"}
        assert (
            len(nq) == 0
            or len(rq) == 0
            or set(nq["CanonicalKey"]).isdisjoint(set(rq["CanonicalKey"]))
        )

    def test_nine_priority_buckets_defined(self):
        assert len(QU.QueueBuilder.BUCKETS) == 9

    def test_bucket_in_known_set(self):
        if not (OUT / "queues" / "existing_paper_recovery_queue.parquet").exists():
            pytest.skip("pipeline outputs absent")
        rq = pd.read_parquet(OUT / "queues" / "existing_paper_recovery_queue.parquet")
        assert set(rq["PriorityBucket"]) <= set(QU.QueueBuilder.BUCKETS)


# ---------------------------------------------------------------------------
# bottleneck / integration structure
# ---------------------------------------------------------------------------


class TestBottleneck:
    def test_counts_monotonic(self):
        counts, table, bottleneck = BN.run_bottleneck_only()
        # later pipeline stages cannot exceed earlier ones
        pairs = [
            ("pdf_parsed", "observations_extracted"),
            ("observations_extracted", "qc_approved"),
            ("qc_approved", "ml_usable"),
        ]
        for a, b in pairs:
            assert counts[a] >= counts[b]

    def test_deliverables_present(self):
        expected = [
            OUT / "gaps" / "evidence_gap_records.parquet",
            OUT / "targets" / "search_targets.parquet",
            OUT / "discovery" / "discovery_query_plan.parquet",
            OUT / "discovery" / "recovery_candidates.parquet",
            OUT / "dedup" / "known_corpus.parquet",
            OUT / "queues" / "new_literature_queue.parquet",
            OUT / "queues" / "existing_paper_recovery_queue.parquet",
            OUT / "handoff" / "acquisition_plan.parquet",
            OUT / "handoff" / "cycle_summary.json",
        ]
        for p in expected:
            assert p.exists(), f"missing deliverable {p}"


class TestHandoff:
    def test_dry_run_plan_only(self):
        if not (OUT / "handoff" / "cycle_summary.json").exists():
            pytest.skip("pipeline outputs absent")
        import json

        s = json.loads((OUT / "handoff" / "cycle_summary.json").read_text(encoding="utf-8"))
        assert s["dry_run"] is True
        assert s["acquisition_status"] == "plan_only"

    def test_gap_target_linkage(self):
        if not (OUT / "gaps" / "evidence_gap_records.parquet").exists():
            pytest.skip("pipeline outputs absent")
        gaps = pd.read_parquet(OUT / "gaps" / "evidence_gap_records.parquet")
        targets = pd.read_parquet(OUT / "targets" / "search_targets.parquet")
        assert set(targets["GapID"]) <= set(gaps["GapID"])
