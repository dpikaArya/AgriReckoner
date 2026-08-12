"""Phase 17 — UAMS coverage repair & external source acquisition tests.

Pure tests exercise the resolver, unit conversion, alias ontology and
prioritisation logic on synthetic data only (no network). Integration tests
assert the structure of the Phase 17 deliverables and are skipped when the
pipeline outputs are absent. Nothing here modifies validated observations,
UAMS, Phase 14/14.5A/14.5B/16/16.1 outputs, or embeddings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".opencode_tmp" / "phase17"
OUT = ROOT / "outputs" / "phase17"

pytest.importorskip("yaml")
pytest.importorskip("pyarrow")
try:
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    import p17_common as C  # noqa: E402
    import p17_mapper as MAP  # noqa: E402
    import p17_ontology as ONT  # noqa: E402
    import p17_priority as PR  # noqa: E402
    import p17_units as UNITS  # noqa: E402
except (ImportError, FileNotFoundError, RuntimeError) as _exc:  # pragma: no cover
    pytest.skip(f"phase17 scratch modules unavailable: {_exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# project root safety
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_outputs_inside_project_root(self):
        for p in (C.OUT, C.REPORTS, C.TMP, C.LOGS):
            assert str(p.resolve()).startswith(str(C.PROJECT_ROOT.resolve()))

    def test_stages_listed(self):
        assert "audit" in C.STAGES
        assert "reports" in C.STAGES


# ---------------------------------------------------------------------------
# unit conversion (synthetic)
# ---------------------------------------------------------------------------


class TestUnits:
    def test_kg_ha_variants(self):
        assert UNITS.normalize_unit("kg ha-1") == "kg/ha"
        assert UNITS.normalize_unit("kg/ha") == "kg/ha"

    def test_t_ha_to_kg_ha(self):
        res = UNITS.convert(2.5, "t/ha", "kg/ha")
        assert res["normalized_value"] == pytest.approx(2500.0)

    def test_percent_to_mg_kg(self):
        res = UNITS.convert(1.5, "%", "mg/kg")
        assert res["normalized_value"] == pytest.approx(15000.0)

    def test_invalid_conversion_none(self):
        assert UNITS.convert(5, "g/plant", "kg/ha") is None

    def test_temperature_offset(self):
        res = UNITS.convert(25.0, "degC", "K")
        assert res["normalized_value"] == pytest.approx(298.15)

    def test_detect_unit_in_header(self):
        assert UNITS.detect_unit("grain yield (kg/ha)") == "kg/ha"


# ---------------------------------------------------------------------------
# alias ontology (synthetic)
# ---------------------------------------------------------------------------


class TestOntology:
    def test_curated_aliases_cover_core_variables(self):
        for col in ("Yield_per_Hectare", "Rainfall", "Soil_pH", "Nitrogen"):
            assert ONT.CURATED_ALIASES.get(col), f"missing curated aliases for {col}"

    def test_abbreviations_exist(self):
        for abbr in ("ph", "gdd", "nue", "bcr", "ndvi"):
            assert abbr in ONT.ABBREVIATIONS

    def test_spelling_pairs_symmetric(self):
        assert ("sulphur", "sulfur") in ONT.SPELLING_PAIRS


# ---------------------------------------------------------------------------
# resolver (synthetic labels)
# ---------------------------------------------------------------------------


class TestResolver:
    @classmethod
    def _resolver(cls):
        return MAP.Resolver()

    def test_exact_canonical(self):
        res = self._resolver().resolve("SPAD")
        assert res["matched"] and res["canonical"] == "SPAD"
        assert res["grade"] == "A"

    def test_curated_alias(self):
        res = self._resolver().resolve("grain yield kg ha-1")
        assert res["matched"] and res["canonical"] == "Yield_per_Hectare"

    def test_unit_aware_alias(self):
        res = self._resolver().resolve("Grain Yield (kg/ha)")
        assert res["matched"] and res["canonical"] == "Yield_per_Hectare"

    def test_abbreviation(self):
        res = self._resolver().resolve("NUE")
        assert res["matched"] and res["canonical"] == "Nitrogen_Use_Efficiency"

    def test_bogus_label_missed(self):
        res = self._resolver().resolve("zzznotarealvar")
        assert not res["matched"]
        assert res["grade"] == "MANUAL_REVIEW"

    def test_confidence_grades_ordered(self):
        r = self._resolver()
        assert r.grade_a >= r.grade_b >= r.grade_c


# ---------------------------------------------------------------------------
# prioritisation logic (synthetic)
# ---------------------------------------------------------------------------


class TestPriority:
    def test_gap_scores_ordered(self):
        assert PR.GAP_SCORE["HIGH"] > PR.GAP_SCORE["LOW"]

    def test_dimension_mapping_has_key_columns(self):
        for dim in ("fertilizer_dose", "irrigation", "yield_measurement"):
            assert PR.DIMENSION_TO_UAMS.get(dim), f"missing mapping for {dim}"


# ---------------------------------------------------------------------------
# deliverable structure (integration, skipped when absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def have_deliverables():
    return (OUT / "uams_coverage_matrix.parquet").exists() and (
        OUT / "ontology_aliases.parquet"
    ).exists()


class TestDeliverables:
    @pytest.mark.skipif(
        not (OUT / "uams_coverage_matrix.parquet").exists(), reason="coverage not run"
    )
    def test_coverage_matrix_covers_all_columns(self):
        df = pd.read_parquet(OUT / "uams_coverage_matrix.parquet")
        cols, _ = C.load_uams_columns()
        assert df["UAMS_Column"].nunique() == len(cols)

    @pytest.mark.skipif(
        not (OUT / "ontology_aliases.parquet").exists(), reason="ontology not run"
    )
    def test_ontology_has_canonical_per_column(self):
        df = pd.read_parquet(OUT / "ontology_aliases.parquet")
        cols, _ = C.load_uams_columns()
        canon = df[df["Source"] == "canonical"]["UAMS_Column"].tolist()
        assert set(canon) == set(cols)

    @pytest.mark.skipif(
        not (OUT / "connector_status.parquet").exists(), reason="audit not run"
    )
    def test_audit_has_expected_columns(self):
        df = pd.read_parquet(OUT / "connector_status.parquet")
        for col in ("Source", "Module", "Importable", "Status", "Failure_Category"):
            assert col in df.columns

    @pytest.mark.skipif(
        not (OUT / "external_data_priority.parquet").exists(), reason="priority not run"
    )
    def test_priority_sorted_desc(self):
        df = pd.read_parquet(OUT / "external_data_priority.parquet")
        scores = df["Information_Gain_Score"].tolist()
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(
        not (OUT / "acquisition_records.parquet").exists(), reason="pilot not run"
    )
    def test_acquisition_has_provenance(self):
        df = pd.read_parquet(OUT / "acquisition_records.parquet")
        assert "Provenance" in df.columns
        assert df["Provenance"].notna().all()

    @pytest.mark.skipif(
        not (OUT / "phase17_final_metrics.json").exists(), reason="reports not run"
    )
    def test_final_metrics_report(self):
        m = json.loads((OUT / "phase17_final_metrics.json").read_text(encoding="utf-8"))
        assert m["stages_completed"]  # non-empty
        assert m["key_results"]["protected_outputs_unchanged"] is True


# ---------------------------------------------------------------------------
# protected-output integrity (integration, skipped when absent)
# ---------------------------------------------------------------------------


class TestProtectedIntegrity:
    @pytest.mark.skipif(
        not (OUT / "integration_metrics.json").exists(), reason="integration not run"
    )
    def test_protected_outputs_unchanged(self):
        m = json.loads((OUT / "integration_metrics.json").read_text(encoding="utf-8"))
        assert m["protected_outputs_unchanged"] is True
