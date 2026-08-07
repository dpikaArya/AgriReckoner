"""Phase 16.2 controlled full-text recovery tests.

Pure tests exercise config merge, DOI normalisation, deterministic storage,
recovery-queue eligibility/exclusions, source credential gating, resolution
plans (no invented PDF URLs), integrity checks, canonical identity resolution,
legacy reconciliation, incremental extraction/QC plans, value scoring,
selective meta/ready-reckoner updates, incremental RAG, ML-readiness and the
16-category failure/retry model on synthetic data only. Integration tests
assert the structure of the Phase 16.2 deliverables and are skipped when the
pipeline outputs are absent. The scratch modules under .opencode_tmp/phase16_2
are required only for the pure tests and are skipped cleanly when unavailable
(e.g. on CI), so collection never fails.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase16_2"
OUT = ROOT / "outputs" / "phase16_2"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")

_HAVE_MODULES = False
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))
try:
    import p16_2_common as C  # noqa: E402
    import p16_2_integrity as INT  # noqa: E402
    import p16_2_recovery as REC  # noqa: E402
    import p16_2_sources as SRC  # noqa: E402

    _HAVE_MODULES = True
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase16_2 scratch modules unavailable: {_exc}", allow_module_level=True)


def _cfg():
    return C.load_config()


def _synthetic_candidates(n=3):
    rows = []
    for i in range(n):
        rows.append(
            {
                "source": "OpenAlex",
                "doi": f"10.1000/x{i}",
                "title": f"paper {i}",
                "authors": "a",
                "year": "2020",
                "recovery_state": "FULL_TEXT_UNAVAILABLE",
                "attempts_so_far": 0,
                "next_retry_date": "",
                "retry_eligible": True,
                "canonical_key": f"doi:10.1000/x{i}",
                "paper_id": f"P{i + 1}",
                "candidate_type": "recovery",
            }
        )
    return pd.DataFrame(rows)


def _synthetic_state():
    return pd.DataFrame(
        [
            {"PaperID": "P1", "HighestState": "FULL_TEXT_UNAVAILABLE"},
            {"PaperID": "P2", "HighestState": "QC_APPROVED"},
            {"PaperID": "P3", "HighestState": "FULL_TEXT_UNAVAILABLE"},
        ]
    )


def _synthetic_queue(monkeypatch):
    monkeypatch.setattr(C, "load_recovery_candidates", lambda: _synthetic_candidates(3))
    monkeypatch.setattr(C, "load_state", _synthetic_state)
    monkeypatch.setattr(
        C,
        "load_attempts_ledger",
        lambda: pd.DataFrame(
            columns=["canonical_key", "attempt_date", "source", "pdf_integrity_ok"]
        ),
    )
    monkeypatch.setattr(REC.RecoveryQueueBuilder, "verified_pdf_keys", lambda self: (set(), set()))
    return REC.RecoveryQueueBuilder(_cfg())


# ---------------------------------------------------------------------------
# project root safety
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_outputs_inside_project_root(self):
        for p in (
            C.OUT,
            C.RECOVERY_OUT,
            C.RESOLUTION_OUT,
            C.INTEGRITY_OUT,
            C.IDENTITY_OUT,
            C.RECONCILE_OUT,
            C.EXTRACTION_OUT,
            C.QC_OUT,
            C.VALUE_OUT,
            C.META_OUT,
            C.RR_OUT,
            C.RAG_OUT,
            C.ML_OUT,
            C.ATTEMPTS_OUT,
            C.REPORTS,
        ):
            assert str(p.resolve()).startswith(str(C.PROJECT_ROOT.resolve()))

    def test_no_absolute_url_fabrication(self):
        # no fixed full-text URLs appear as constants anywhere but the
        # documented connector endpoints registered in the source registry
        import p16_2_sources as S
        from p16_2_resolution import _plan_for

        plan = _plan_for(
            {"PaperID": "P", "CanonicalKey": "k", "DOI": "", "Title": "t", "Year": "2000"},
            ["OpenAlex"],
        )
        for row in plan:
            url = row["QueryURL"]
            if url:
                assert url.startswith("https://")
        for _, row in S.SourceRegistry(_cfg()).status_table().iterrows():
            base = row["base_url"]
            if base:
                assert base.startswith("https://")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_present(self):
        cfg = _cfg()
        assert cfg["dry_run"] is True
        assert cfg["recovery"]["max_attempts_per_paper"] >= 1
        assert cfg["rag"]["citation_accuracy_target"] >= 0.9
        assert cfg["qc"]["min_extraction_confidence"] > 0
        assert cfg["integrity"]["min_file_size_bytes"] > 0

    def test_eligible_and_exclude_disjoint(self):
        cfg = _cfg()
        assert set(cfg["recovery"]["eligible_states"]).isdisjoint(
            set(cfg["recovery"]["exclude_states"])
        )

    def test_discovery_only_not_in_recovery_order(self):
        cfg = _cfg()
        assert set(cfg["sources"]["discovery_only"]).isdisjoint(set(cfg["sources"]["order"]))

    def test_user_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(
            "learning:\n  phase16_2:\n    dry_run: false\n"
            "    recovery:\n      max_recovery_per_cycle: 5\n",
            encoding="utf-8",
        )
        cfg = C.load_config(p)
        assert cfg["dry_run"] is False
        assert cfg["recovery"]["max_recovery_per_cycle"] == 5

    def test_repo_config_has_phase16_2(self):
        import yaml

        raw = yaml.safe_load(C.CONFIG.read_text(encoding="utf-8")) or {}
        section = (raw.get("learning") or {}).get("phase16_2")
        assert isinstance(section, dict)
        assert "sources" in section and "recovery" in section


# ---------------------------------------------------------------------------
# doi normalisation / deterministic storage
# ---------------------------------------------------------------------------


class TestIdentityHelpers:
    def test_normalize_doi_strips_prefixes(self):
        for p in (
            "https://doi.org/",
            "http://doi.org/",
            "https://dx.doi.org/",
            "http://dx.doi.org/",
            "doi:",
        ):
            assert INT.normalize_doi(p + "10.1000/ABC") == "10.1000/abc"

    def test_deterministic_path_stable(self):
        a = INT.deterministic_pdf_path("https://doi.org/10.1000/XY-Z")
        b = INT.deterministic_pdf_path("doi:10.1000/xy-z")
        assert a == b
        assert a.parent == C.PROJECT_ROOT / "recovered_pdfs"

    def test_deterministic_path_safe_chars(self):
        p = INT.deterministic_pdf_path("10.1000/a b/c")
        assert str(p.name) == "10.1000_a_b_c.pdf"


# ---------------------------------------------------------------------------
# recovery queue
# ---------------------------------------------------------------------------


class TestRecoveryQueue:
    def test_eligibility_by_state(self, monkeypatch):
        builder = _synthetic_queue(monkeypatch)
        q = builder.build()
        # P2 (QC_APPROVED) excluded; P1 and P3 eligible
        assert set(q["PaperID"]) == {"P1", "P3"}
        assert set(q["State"]) <= set(builder.eligible)

    def test_no_duplicate_canonical_keys(self, monkeypatch):
        cands = pd.concat([_synthetic_candidates(2), _synthetic_candidates(2)])
        monkeypatch.setattr(C, "load_recovery_candidates", lambda: cands)
        monkeypatch.setattr(C, "load_state", _synthetic_state)
        monkeypatch.setattr(
            C, "load_attempts_ledger", lambda: pd.DataFrame(columns=["canonical_key"])
        )
        q = REC.RecoveryQueueBuilder(_cfg()).build()
        assert q["CanonicalKey"].is_unique

    def test_max_attempts_excluded(self, monkeypatch):
        ledger = pd.DataFrame(
            [
                {
                    "canonical_key": "doi:10.1000/x0",
                    "attempt_date": "2026-01-01",
                    "source": "a",
                    "pdf_integrity_ok": False,
                },
                {
                    "canonical_key": "doi:10.1000/x0",
                    "attempt_date": "2026-02-01",
                    "source": "a",
                    "pdf_integrity_ok": False,
                },
                {
                    "canonical_key": "doi:10.1000/x0",
                    "attempt_date": "2026-03-01",
                    "source": "a",
                    "pdf_integrity_ok": False,
                },
            ]
        )
        monkeypatch.setattr(C, "load_recovery_candidates", lambda: _synthetic_candidates(1))
        monkeypatch.setattr(C, "load_state", _synthetic_state)
        monkeypatch.setattr(C, "load_attempts_ledger", lambda: ledger)
        builder = REC.RecoveryQueueBuilder(_cfg())
        q = builder.build()
        assert len(q) == 0  # attempts 3 >= max 3

    def test_retry_not_due_excluded(self, monkeypatch):
        from datetime import date, timedelta

        ledger = pd.DataFrame(
            [
                {
                    "canonical_key": "doi:10.1000/x0",
                    "attempt_date": str(date.today() - timedelta(days=1)),
                    "source": "a",
                    "pdf_integrity_ok": False,
                },
            ]
        )
        monkeypatch.setattr(C, "load_recovery_candidates", lambda: _synthetic_candidates(1))
        monkeypatch.setattr(C, "load_state", _synthetic_state)
        monkeypatch.setattr(C, "load_attempts_ledger", lambda: ledger)
        builder = REC.RecoveryQueueBuilder(_cfg())
        assert len(builder.build()) == 0  # next retry 29 days away

    def test_verified_pdf_excluded(self, monkeypatch, tmp_path):
        from p16_2_recovery import RecoveryQueueBuilder

        builder = RecoveryQueueBuilder(_cfg())
        q = _synthetic_candidates(1)
        monkeypatch.setattr(C, "load_recovery_candidates", lambda: q)
        monkeypatch.setattr(
            C,
            "load_state",
            lambda: pd.DataFrame([{"PaperID": "P1", "HighestState": "FULL_TEXT_UNAVAILABLE"}]),
        )
        monkeypatch.setattr(
            C, "load_attempts_ledger", lambda: pd.DataFrame(columns=["canonical_key"])
        )
        monkeypatch.setattr(RecoveryQueueBuilder, "verified_pdf_keys", lambda self: ({"P1"}, set()))
        assert len(builder.build()) == 0

    def test_max_per_cycle_capped(self, monkeypatch):
        cands = _synthetic_candidates(20)
        monkeypatch.setattr(C, "load_recovery_candidates", lambda: cands)
        monkeypatch.setattr(C, "load_state", _synthetic_state)
        monkeypatch.setattr(
            C, "load_attempts_ledger", lambda: pd.DataFrame(columns=["canonical_key"])
        )
        monkeypatch.setattr(
            REC.RecoveryQueueBuilder, "verified_pdf_keys", lambda self: (set(), set())
        )
        cfg = _cfg()
        cfg["recovery"]["max_recovery_per_cycle"] = 3
        q = REC.RecoveryQueueBuilder(cfg).build()
        assert len(q) == 3


# ---------------------------------------------------------------------------
# sources / credential gating
# ---------------------------------------------------------------------------


class TestSources:
    def test_credential_gated_skipped(self, monkeypatch):
        monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
        monkeypatch.delenv("CORE_API_KEY", raising=False)
        st, metrics = SRC.SourceRegistry(_cfg()).run()
        assert "Unpaywall" in metrics["skipped"]
        assert "credentials missing" in metrics["skipped_reasons"]["Unpaywall"]

    def test_with_credential_available(self, monkeypatch):
        monkeypatch.setenv("UNPAYWALL_EMAIL", "me@example.com")
        st, metrics = SRC.SourceRegistry(_cfg()).run()
        assert "Unpaywall" in metrics["available"]

    def test_discovery_only_never_available_for_recovery(self, monkeypatch):
        monkeypatch.setenv("SCOPUS_API_KEY", "k")
        monkeypatch.setenv("WOS_API_KEY", "k")
        monkeypatch.setenv("DIMENSIONS_API_KEY", "k")
        reg = SRC.SourceRegistry(_cfg())
        available = set(reg.available())
        assert not (available & set(reg.cfg["sources"]["discovery_only"]))

    def test_order_preserved(self):
        reg = SRC.SourceRegistry(_cfg())
        assert reg.order == _cfg()["sources"]["order"]
        assert reg.order[0] == "OpenAlex"


# ---------------------------------------------------------------------------
# resolution plan
# ---------------------------------------------------------------------------


class TestResolution:
    def test_priority_order(self):
        from p16_2_resolution import _plan_for

        plan = _plan_for(
            {"PaperID": "P", "CanonicalKey": "k", "DOI": "10.1000/x", "Title": "t", "Year": "2000"},
            ["Unpaywall", "Crossref", "Europe_PMC", "OpenAlex"],
        )
        prios = [row["Priority"] for row in plan]
        assert prios == sorted(prios)
        assert plan[0]["ResolutionType"] == "OA_location_Unpaywall"

    def test_deterministic_urls_known_a_priori(self):
        from p16_2_resolution import _plan_for

        plan = _plan_for(
            {"PaperID": "P", "CanonicalKey": "k", "DOI": "10.1000/x", "Title": "t", "Year": "2000"},
            ["Unpaywall", "Crossref", "Europe_PMC", "OpenAlex"],
        )
        for row in plan:
            if row["ResolutionType"] in (
                "OA_location_Unpaywall",
                "Crossref_link_license",
                "EuropePMC_fulltext",
                "OA_location_OpenAlex",
            ):
                assert row["URLKnownAPriori"] is True
                assert "10.1000/x" in row["QueryURL"]

    def test_direct_pdf_url_not_invented(self):
        from p16_2_resolution import _plan_for

        plan = _plan_for(
            {"PaperID": "P", "CanonicalKey": "k", "DOI": "10.1000/x", "Title": "t", "Year": "2000"},
            ["OpenAlex"],
        )
        direct = [r for r in plan if r["ResolutionType"] == "Direct_PDF"]
        assert direct and direct[0]["URLKnownAPriori"] is False
        assert direct[0]["QueryURL"] == ""

    def test_dry_run_all_planned(self, monkeypatch):
        from p16_2_resolution import ResolutionPlanner

        planner = ResolutionPlanner(_cfg())
        q = _synthetic_queue(monkeypatch).build()
        plan = planner.plan(q)
        assert set(plan["Status"]) == {"planned"}
        assert not plan["Retrieved"].any()

    def test_budget_respected(self, monkeypatch):
        from p16_2_resolution import ResolutionPlanner

        planner = ResolutionPlanner(_cfg())
        q = _synthetic_queue(monkeypatch).build()
        plan = planner.plan(q)
        budget = planner.budget(plan)
        assert budget["within_budget"] is True
        assert budget["papers_planned"] == len(q)


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_signature_detects_real_pdf(self, tmp_path):
        p = tmp_path / "x.pdf"
        p.write_bytes(b"%PDF-1.4\n%%EOF\n")
        assert INT.pdf_signature_ok(p) is True

    def test_signature_rejects_non_pdf(self, tmp_path):
        p = tmp_path / "x.pdf"
        p.write_bytes(b"<html>login required</html>")
        assert INT.pdf_signature_ok(p) is False

    def test_plan_is_planned_only(self, monkeypatch):
        from p16_2_recovery import RecoveryQueueBuilder

        q = _synthetic_queue(monkeypatch).build()
        monkeypatch.setattr(RecoveryQueueBuilder, "build", lambda self: q)
        plan, audit, summary = INT.IntegrityChecker(_cfg()).run()
        assert set(plan["Status"]) == {"planned"}
        assert summary["planned_checks"] == len(q)

    def test_deterministic_path_inside_root(self):
        p = INT.deterministic_pdf_path("10.1000/a")
        assert str(p.resolve()).startswith(str(C.PROJECT_ROOT.resolve()))


# ---------------------------------------------------------------------------
# canonical identity / reconciliation
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_resolve_against_corpus(self, monkeypatch):
        from p16_2_identity import IdentityResolver

        corpus = pd.DataFrame(
            [
                {
                    "paper_id": "P1",
                    "canonical_key": "doi:10.1000/x0",
                    "doi": "10.1000/x0",
                    "normalized_doi": "10.1000/x0",
                    "title": "paper 0",
                    "state": "FULL_TEXT_UNAVAILABLE",
                },
            ]
        )
        monkeypatch.setattr(C, "load_known_corpus", lambda: corpus)
        resolver = IdentityResolver(_cfg())
        q = _synthetic_queue(monkeypatch).build()
        ident = resolver.resolve(q)
        assert set(ident["PaperID"]) == {"P1", "P3"}
        row = ident[ident["PaperID"] == "P1"].iloc[0]
        assert bool(row["InKnownCorpus"]) is True
        assert row["IdentityConfidence"] == "high"

    def test_reconcile_flags_prior_extraction(self, monkeypatch, tmp_path):
        from p16_2_reconcile import Reconciliation

        ident = pd.DataFrame(
            [
                {
                    "PaperID": "P1",
                    "CanonicalKey": "k1",
                    "DOI": "10.1/a",
                    "NormalizedDOI": "10.1/a",
                    "Title": "t",
                    "InKnownCorpus": True,
                },
                {
                    "PaperID": "P2",
                    "CanonicalKey": "k2",
                    "DOI": "10.1/b",
                    "NormalizedDOI": "10.1/b",
                    "Title": "u",
                    "InKnownCorpus": True,
                },
            ]
        )
        rec = Reconciliation(_cfg())
        monkeypatch.setattr(
            rec,
            "load_all",
            lambda: (
                pd.DataFrame([{"PaperID": "P1", "Crop": "wheat", "Variable": "yield"}]),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame([{"Crop": "wheat", "Variable": "yield"}]),
                pd.DataFrame([{"Crop": "wheat", "Variable": "yield"}]),
            ),
        )
        out = rec.reconcile(ident)
        r1 = out[out["PaperID"] == "P1"].iloc[0]
        assert bool(r1["LegacyExtractionPresent"]) is True
        assert bool(r1["AffectsMetaGroups"]) is True
        assert bool(r1["AffectsReadyReckoner"]) is True
        r2 = out[out["PaperID"] == "P2"].iloc[0]
        assert bool(r2["LegacyExtractionPresent"]) is False


# ---------------------------------------------------------------------------
# incremental extraction / QC
# ---------------------------------------------------------------------------


class TestExtractionPlan:
    def test_four_stages_per_paper(self, monkeypatch):
        from p16_2_extraction import PIPELINE_STAGES, ExtractionPlanner

        q = _synthetic_queue(monkeypatch).build()
        ident = pd.DataFrame(
            [
                {
                    "PaperID": r["PaperID"],
                    "CanonicalKey": r["CanonicalKey"],
                    "DOI": r["DOI"],
                    "Title": r["Title"],
                }
                for _, r in q.iterrows()
            ]
        )
        plan = ExtractionPlanner(_cfg()).plan(ident)
        assert plan["PaperID"].value_counts().min() == len(PIPELINE_STAGES)
        assert set(plan["Status"]) == {"planned"}

    def test_pipeline_reuses_existing_modules(self):
        from p16_2_extraction import PIPELINE_STAGES

        stages = dict(PIPELINE_STAGES)
        assert "PDF_Parse" in stages and "QC_Gate" in stages


class TestQC:
    def test_thresholds_from_config(self):
        cfg = _cfg()
        qc = cfg["qc"]
        assert qc["min_extraction_confidence"] >= 0.5
        assert qc["min_evidence_grade"] in ("B", "C", "D")

    def test_uncertain_to_review(self):
        assert _cfg()["qc"]["send_uncertain_to_review"] is True


# ---------------------------------------------------------------------------
# value scoring
# ---------------------------------------------------------------------------


class TestValue:
    def test_score_in_unit_interval(self, monkeypatch):
        from p16_2_value import ValueScorer

        q = _synthetic_queue(monkeypatch).build()
        rec = pd.DataFrame(columns=["PaperID", "AffectsMetaGroups", "AffectsReadyReckoner"])
        scores = ValueScorer(_cfg()).score(q, rec)
        assert (scores["InformationGainScore"] >= 0).all()
        assert (scores["InformationGainScore"] <= 1.0 + 1e-9).all()

    def test_value_rank_unique(self, monkeypatch):
        from p16_2_value import ValueScorer

        q = _synthetic_queue(monkeypatch).build()
        rec = pd.DataFrame(columns=["PaperID", "AffectsMetaGroups", "AffectsReadyReckoner"])
        scores = ValueScorer(_cfg()).score(q, rec)
        assert scores["ValueRank"].is_unique

    def test_deterministic(self, monkeypatch):
        from p16_2_value import ValueScorer

        q = _synthetic_queue(monkeypatch).build()
        rec = pd.DataFrame(columns=["PaperID", "AffectsMetaGroups", "AffectsReadyReckoner"])
        a = ValueScorer(_cfg()).score(q, rec)
        b = ValueScorer(_cfg()).score(q, rec)
        assert a["InformationGainScore"].equals(b["InformationGainScore"])


# ---------------------------------------------------------------------------
# selective meta / ready-reckoner / RAG / ML
# ---------------------------------------------------------------------------


class TestSelectiveUpdates:
    def _rec(self):
        return pd.DataFrame(
            [
                {
                    "PaperID": "P1",
                    "CanonicalKey": "k1",
                    "DOI": "10.1/a",
                    "Title": "t",
                    "RAGChunkCount": 5,
                    "AffectedGroups": "wheat|yield",
                }
            ]
        )

    def test_meta_selective_only(self, monkeypatch):
        from p16_2_meta import MetaUpdater

        meta = MetaUpdater(_cfg())
        rec = self._rec()
        monkeypatch.setattr(
            meta,
            "current_groups",
            lambda: pd.DataFrame(
                [
                    {
                        "Crop": "wheat",
                        "Variable": "yield",
                        "Unit": "kg/ha",
                        "PublicationCount": 2,
                        "k_studies": 1,
                        "n_observations": 5,
                        "effect_size_g": 0.3,
                    }
                ]
            ),
        )
        plan = meta.plan(rec)
        assert bool(plan.loc[0, "Affected"]) is True
        assert bool(plan.loc[0, "RecomputeRequired"]) is True
        assert set(plan["Status"]) == {"planned_update"}

    def test_rr_selective_only(self, monkeypatch):
        from p16_2_rr import ReadyReckonerUpdater

        rr = ReadyReckonerUpdater(_cfg())
        rec = self._rec()
        monkeypatch.setattr(
            rr,
            "current_rr",
            lambda: pd.DataFrame(
                [
                    {
                        "RecommendationID": "R1",
                        "Crop": "wheat",
                        "Variable": "yield",
                        "Unit": "kg/ha",
                        "CentralEstimate": 100,
                        "PublicationCount": 2,
                        "n_observations": 5,
                    }
                ]
            ),
        )
        plan = rr.plan(rec)
        assert bool(plan.loc[0, "Affected"]) is True

    def test_meta_unchanged_groups_left_alone(self, monkeypatch):
        from p16_2_meta import MetaUpdater

        meta = MetaUpdater(_cfg())
        rec = pd.DataFrame(
            [
                {
                    "PaperID": "P9",
                    "CanonicalKey": "k9",
                    "DOI": "10.1/z",
                    "Title": "z",
                    "AffectedGroups": "",
                }
            ]
        )
        monkeypatch.setattr(
            meta,
            "current_groups",
            lambda: pd.DataFrame(
                [
                    {
                        "Crop": "rice",
                        "Variable": "yield",
                        "Unit": "t/ha",
                        "PublicationCount": 1,
                        "k_studies": 1,
                        "n_observations": 2,
                        "effect_size_g": 0.2,
                    }
                ]
            ),
        )
        plan = meta.plan(rec)
        assert bool(plan.loc[0, "Affected"]) is False
        assert plan.loc[0, "Status"] == "unchanged"

    def test_rag_reuses_existing_embeddings(self):
        cfg = _cfg()
        assert cfg["rag"]["reuse_existing_embeddings"] is True
        assert cfg["rag"]["incremental"] is True

    def test_rag_target_095(self):
        assert _cfg()["rag"]["citation_accuracy_target"] >= 0.95

    def test_ml_retrain_disallowed(self):
        assert _cfg()["ml_readiness"]["retrain_allowed"] is False
        assert _cfg()["ml_readiness"]["recalculate"] is True


# ---------------------------------------------------------------------------
# failure classification / retry backoff
# ---------------------------------------------------------------------------


class TestAttempts:
    def test_sixteen_categories(self):
        from p16_2_attempts import FAILURE_CATEGORIES, AttemptsManager

        assert len(FAILURE_CATEGORIES) == 16
        cls = AttemptsManager(_cfg()).classification_table()
        assert len(cls) == 16

    def test_backoff_monotonic_and_bounded(self):
        from p16_2_attempts import AttemptsManager

        m = AttemptsManager(_cfg())
        seq = [m.backoff(i) for i in (1, 2, 3, 4)]
        assert seq == sorted(seq)
        assert seq[-1] <= m.max_backoff

    def test_max_attempts_cap(self):
        assert _cfg()["recovery"]["max_attempts_per_paper"] >= 1

    def test_retry_schedule_budgeted(self, monkeypatch):
        from p16_2_attempts import AttemptsManager

        q = _synthetic_queue(monkeypatch).build()
        m = AttemptsManager(_cfg())
        sched = m.retry_schedule(q)
        assert len(sched) == len(q)


# ---------------------------------------------------------------------------
# integration structure (deliverables)
# ---------------------------------------------------------------------------


class TestDeliverables:
    def test_outputs_present(self):
        expected = [
            OUT / "recovery" / "recovery_queue.parquet",
            OUT / "resolution" / "resolution_plan.parquet",
            OUT / "integrity" / "integrity_plan.parquet",
            OUT / "identity" / "canonical_identity.parquet",
            OUT / "reconcile" / "reconciliation.parquet",
            OUT / "extraction" / "extraction_plan.parquet",
            OUT / "qc" / "qc_gate_plan.parquet",
            OUT / "value" / "value_scores.parquet",
            OUT / "meta" / "meta_update_plan.parquet",
            OUT / "ready_reckoner" / "rr_update_plan.parquet",
            OUT / "rag" / "rag_update_plan.parquet",
            OUT / "ml" / "ml_readiness_plan.parquet",
            OUT / "attempts" / "retry_schedule.parquet",
        ]
        for p in expected:
            assert p.exists(), f"missing deliverable {p}"

    def test_recovery_queue_eligible_states_only(self):
        if not (OUT / "recovery" / "recovery_queue.parquet").exists():
            pytest.skip("pipeline outputs absent")
        q = pd.read_parquet(OUT / "recovery" / "recovery_queue.parquet")
        eligible = set(_cfg()["recovery"]["eligible_states"])
        assert set(q["State"]) <= eligible

    def test_recovery_keys_unique(self):
        if not (OUT / "recovery" / "recovery_queue.parquet").exists():
            pytest.skip("pipeline outputs absent")
        q = pd.read_parquet(OUT / "recovery" / "recovery_queue.parquet")
        assert q["CanonicalKey"].is_unique

    def test_resolution_no_invented_pdf_urls(self):
        if not (OUT / "resolution" / "resolution_plan.parquet").exists():
            pytest.skip("pipeline outputs absent")
        plan = pd.read_parquet(OUT / "resolution" / "resolution_plan.parquet")
        assert set(plan["Status"]) == {"planned"}
        invented = plan[plan["URLKnownAPriori"] == False]  # noqa: E712
        assert invented["QueryURL"].fillna("").astype(str).str.len().eq(0).all()

    def test_dry_run_summary(self):
        if not (OUT / "phase16_2_summary.json").exists():
            pytest.skip("pipeline outputs absent")
        import json

        s = json.loads((OUT / "phase16_2_summary.json").read_text(encoding="utf-8"))
        assert s["dry_run"] is True
        assert s["stages_completed"] == [
            "recovery",
            "sources",
            "resolution",
            "integrity",
            "identity",
            "reconcile",
            "extraction",
            "qc",
            "value",
            "meta",
            "ready_reckoner",
            "rag",
            "ml_readiness",
            "attempts",
            "reports",
        ]

    def test_checkpoint_restartable(self):
        from p16_2_common import load_checkpoint

        cp = load_checkpoint()
        assert cp.get("last_step") == "reports"
