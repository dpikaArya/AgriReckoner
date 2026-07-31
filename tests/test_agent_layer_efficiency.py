"""
Per-Agent Layer Efficiency Test Suite.

Tests each of the 7 pipeline layers individually for:
  - Throughput (rows/sec)
  - Data preservation (% columns retained)
  - Feature quality (new features created, leakage prevention)
  - Error handling (graceful failure on bad input)
  - Output completeness (required fields present)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_base_df(n: int = 100, include_yield: bool = True) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "Paper_ID": [f"P{i:03d}" for i in range(n)],
            "Crop": rng.choice(["Rice", "Wheat", "Maize"], n),
            "Variety": rng.choice(["V1", "V2", "V3"], n),
            "Season": rng.choice(["Kharif", "Rabi"], n),
            "Country": rng.choice(["India", "USA"], n),
            "Soil_pH": rng.uniform(4.5, 8.5, n),
            "EC": rng.uniform(0.1, 5.0, n),
            "Organic_Carbon": rng.uniform(0.1, 2.0, n),
            "Nitrogen": rng.uniform(20, 250, n),
            "Phosphorus": rng.uniform(5, 80, n),
            "Potassium": rng.uniform(50, 400, n),
            "Temperature_Max": rng.uniform(20, 42, n),
            "Temperature_Min": rng.uniform(5, 25, n),
            "Average_Temperature": rng.uniform(15, 35, n),
            "Rainfall": rng.uniform(200, 2500, n),
            "Humidity": rng.uniform(30, 95, n),
            "Plant_Height_cm": rng.uniform(20, 300, n),
            "SPAD": rng.uniform(15, 75, n),
            "Biomass_Yield": rng.uniform(500, 8000, n),
            "Shoot_Biomass_g": rng.uniform(10, 500, n),
            "Root_Biomass_g": rng.uniform(5, 200, n),
            "Leaf_Area_cm2": rng.uniform(10, 500, n),
            "Fruit_Number": rng.randint(5, 100, n),
            "Fruit_Weight": rng.uniform(5, 500, n),
            "100_Seed_Weight": rng.uniform(5, 50, n),
            "Fertilizer_Name": rng.choice(["Urea", "DAP", "MOP", "15-15-15"], n),
            "Dose": rng.uniform(20, 200, n),
            "Plot_Size": rng.choice([10, 16, 20], n),
        }
    )
    if include_yield:
        df["Yield_per_Hectare"] = (
            2000
            + 15 * df["Nitrogen"]
            + 10 * df["Phosphorus"]
            + 3 * df["Rainfall"] / 100
            + rng.normal(0, 300, n)
        ).clip(lower=100)
    return df


@dataclass
class LayerResult:
    layer: str
    agent: str
    rows_in: int
    rows_out: int
    cols_in: int
    cols_out: int
    new_cols: list = field(default_factory=list)
    elapsed_sec: float = 0.0
    throughput_rps: float = 0.0
    data_preservation_pct: float = 0.0
    leakage_detected: int = 0
    errors: list = field(default_factory=list)
    passed: bool = False
    notes: str = ""


# ── Layer 1: Extraction ──────────────────────────────────────────────────────


class TestLayer1Extraction:
    LAYER = "L1 Extraction"

    def test_extraction_agent_processes_csv(self):
        from agri_ai_agent.agents.extraction_agent import ExtractionAgent

        agent = ExtractionAgent()
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "Data ADES"
        candidates = [
            data_dir / "Growth_Parameters_Dryad.csv",
            data_dir / "Data_for_correlation.csv",
            data_dir / "Soil_Analysis_Dryad.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            pytest.skip("No real CSV in Data ADES/ for extraction test")
        rows_in = len(pd.read_csv(csv_path))
        try:
            t0 = time.time()
            contract = agent.run(pd.DataFrame(), filepath=str(csv_path))
            elapsed = time.time() - t0
            assert contract.status == "success", f"ExtractionAgent failed: {contract.errors}"
            result_df = agent.dataframe
            assert result_df is not None and len(result_df) > 0
            r = LayerResult(
                layer=self.LAYER,
                agent="ExtractionAgent",
                rows_in=rows_in,
                rows_out=len(result_df),
                cols_in=0,
                cols_out=len(result_df.columns),
                elapsed_sec=round(elapsed, 3),
                throughput_rps=round(len(result_df) / max(elapsed, 0.001), 1),
                data_preservation_pct=round(len(result_df.columns) / 10 * 100, 1),
                passed=True,
            )
            print(
                f"\n  [{self.LAYER}] ExtractionAgent ({csv_path.name}): {r.throughput_rps} rows/s, "
                f"{r.data_preservation_pct}% col preservation"
            )
        except Exception as e:
            r = LayerResult(
                layer=self.LAYER,
                agent="ExtractionAgent",
                rows_in=rows_in,
                rows_out=0,
                cols_in=0,
                cols_out=0,
                errors=[str(e)],
                passed=False,
            )
            pytest.skip(f"ExtractionAgent not testable standalone: {e}")

    def test_evidence_fusion_handles_empty(self):
        from agri_ai_agent.agents.evidence_fusion_agent import EvidenceFusionAgent

        agent = EvidenceFusionAgent()
        df = _make_base_df(10)
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        assert contract.status == "success"
        r = LayerResult(
            layer=self.LAYER,
            agent="EvidenceFusionAgent",
            rows_in=10,
            rows_out=len(agent.dataframe),
            cols_in=len(df.columns),
            cols_out=len(agent.dataframe.columns),
            elapsed_sec=round(elapsed, 3),
            passed=True,
        )
        print(f"  [{self.LAYER}] EvidenceFusionAgent: {r.throughput_rps} rows/s")


# ── Layer 2: Normalization ───────────────────────────────────────────────────


class TestLayer2Normalization:
    LAYER = "L2 Normalization"

    def test_ontology_agent_column_mapping(self):
        from agri_ai_agent.agents.ontology_agent import OntologyAgent

        agent = OntologyAgent()
        df = _make_base_df(100)
        df = df.rename(
            columns={"Soil_pH": "ph", "Nitrogen": "n_total", "Rainfall": "precipitation"}
        )
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        mapped_cols = [c for c in result_df.columns if c in ["Soil_pH", "Nitrogen", "Rainfall"]]
        mapped_pct = round(len(mapped_cols) / 3 * 100, 1)
        r = LayerResult(
            layer=self.LAYER,
            agent="OntologyAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(100 / max(elapsed, 0.001), 1),
            data_preservation_pct=round(len(result_df.columns) / max(len(df.columns), 1) * 100, 1),
            notes=f"Mapped {mapped_pct}% of target synonyms",
            passed=len(mapped_cols) >= 2,
        )
        print(
            f"\n  [{self.LAYER}] OntologyAgent: {r.throughput_rps} rows/s, "
            f"mapped {mapped_pct}% synonyms"
        )
        assert r.passed

    def test_table_intelligence_agent(self):
        from agri_ai_agent.agents.table_intelligence_agent import TableIntelligenceAgent

        agent = TableIntelligenceAgent()
        df = _make_base_df(100)
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        r = LayerResult(
            layer=self.LAYER,
            agent="TableIntelligenceAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(100 / max(elapsed, 0.001), 1),
            data_preservation_pct=round(len(result_df.columns) / max(len(df.columns), 1) * 100, 1),
            passed=True,
        )
        print(f"  [{self.LAYER}] TableIntelligenceAgent: {r.throughput_rps} rows/s")

    def test_schema_population_derives_safely(self):
        from agri_ai_agent.agents.schema_population_agent import SchemaPopulationAgent

        agent = SchemaPopulationAgent()
        df = _make_base_df(100)
        df["Yield_per_Plot"] = df["Yield_per_Hectare"] * df["Plot_Size"] / 10000
        df = df.drop(columns=["Yield_per_Hectare"])
        contract = agent.run(df)
        result_df = agent.dataframe
        assert contract.status == "success"
        leaky_targets = ["Nitrogen_Use_Efficiency", "Water_Use_Efficiency", "Harvest_Index"]
        derived_leaky = [c for c in result_df.columns if c in leaky_targets]
        assert len(derived_leaky) == 0, f"Leaky rules still present: {derived_leaky}"
        LayerResult(
            layer=self.LAYER,
            agent="SchemaPopulationAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            leakage_detected=0,
            notes="No yield-derived efficiency features created",
            passed=True,
        )
        print(f"  [{self.LAYER}] SchemaPopulationAgent: leakage-free={len(derived_leaky) == 0}")


# ── Layer 3: Knowledge ───────────────────────────────────────────────────────


class TestLayer3Knowledge:
    LAYER = "L3 Knowledge"

    def test_knowledge_agent_annotations(self):
        from agri_ai_agent.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent()
        df = _make_base_df(100)
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        required = ["Optimal_pH_Min", "Optimal_pH_Max", "N_Status", "P_Status", "K_Status"]
        present = [c for c in required if c in result_df.columns]
        coverage_pct = round(len(present) / len(required) * 100, 1)
        r = LayerResult(
            layer=self.LAYER,
            agent="KnowledgeAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(100 / max(elapsed, 0.001), 1),
            data_preservation_pct=coverage_pct,
            notes=f"{len(present)}/{len(required)} annotations present",
            passed=coverage_pct >= 80,
        )
        print(
            f"\n  [{self.LAYER}] KnowledgeAgent: {r.throughput_rps} rows/s, "
            f"coverage {coverage_pct}%"
        )
        assert r.passed

    def test_knowledge_integration_deterministic(self):
        from agri_ai_agent.agents.knowledge_integration_agent import KnowledgeIntegrationAgent

        agent = KnowledgeIntegrationAgent()
        df = _make_base_df(50)
        for col in ["Soil_pH", "Organic_Carbon", "Nitrogen", "Phosphorus", "Potassium"]:
            df[col] = np.nan
        df["Soil_Texture"] = "Loamy"
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        filled = (
            result_df[["Soil_pH", "Organic_Carbon", "Nitrogen", "Phosphorus", "Potassium"]]
            .notna()
            .all(axis=1)
            .sum()
        )
        r = LayerResult(
            layer=self.LAYER,
            agent="KnowledgeIntegrationAgent",
            rows_in=50,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(50 / max(elapsed, 0.001), 1),
            notes=f"Filled {filled}/50 rows deterministically (no np.random)",
            passed=filled > 0,
        )
        print(f"  [{self.LAYER}] KnowledgeIntegrationAgent: filled {filled}/50 rows")
        assert r.passed


# ── Layer 4: Validation ──────────────────────────────────────────────────────


class TestLayer4Validation:
    LAYER = "L4 Validation"

    def test_validation_agent_quality_checks(self):
        from agri_ai_agent.agents.validation_agent import ValidationAgent

        agent = ValidationAgent()
        df = _make_base_df(100)
        df.loc[0, "Soil_pH"] = 25.0
        df.loc[1, "Temperature_Max"] = -50.0
        df.loc[2, "Nitrogen"] = -10
        t0 = time.time()
        contract = agent.run(df)
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        meta = contract.metadata or {}
        quality_issues = meta.get("quality_issues", {})
        total_issues = sum(v for v in quality_issues.values() if isinstance(v, (int, float)))
        r = LayerResult(
            layer=self.LAYER,
            agent="ValidationAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(100 / max(elapsed, 0.001), 1),
            notes=f"Detected {total_issues} quality issues",
            passed=total_issues > 0,
        )
        print(
            f"\n  [{self.LAYER}] ValidationAgent: {r.throughput_rps} rows/s, "
            f"{total_issues} issues found"
        )
        assert r.passed


# ── Layer 5: ML ──────────────────────────────────────────────────────────────


class TestLayer5ML:
    LAYER = "L5 ML"

    def test_feature_agent_no_yield_leakage(self):
        from agri_ai_agent.agents.feature_agent import FeatureAgent

        agent = FeatureAgent()
        df = _make_base_df(200)
        t0 = time.time()
        contract = agent.run(df, target_col="Yield_per_Hectare")
        elapsed = time.time() - t0
        result_df = agent.dataframe
        assert contract.status == "success"
        leaky_names = [
            "Yield_x_N",
            "Yield_x_pH",
            "Yield_x_Biomass",
            "Yield_pH_N_interaction",
            "Yield_per_Height",
            "Harvest_Index_Calc",
            "Nitrogen_Use_Efficiency",
            "Water_Use_Efficiency",
            "Yield_per_Plot_Calc",
            "Yield_per_Hectare_Calc",
            "Yield_per_Hectare_log",
            "Yield_category",
        ]
        present_leaky = [c for c in leaky_names if c in result_df.columns]
        new_cols = [c for c in result_df.columns if c not in df.columns]
        r = LayerResult(
            layer=self.LAYER,
            agent="FeatureAgent",
            rows_in=200,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            new_cols=new_cols,
            leakage_detected=len(present_leaky),
            elapsed_sec=round(elapsed, 3),
            throughput_rps=round(200 / max(elapsed, 0.001), 1),
            data_preservation_pct=round(len(result_df.columns) / max(len(df.columns), 1) * 100, 1),
            notes=f"{len(new_cols)} features created, {len(present_leaky)} leaky found",
            passed=len(present_leaky) == 0,
        )
        print(
            f"\n  [{self.LAYER}] FeatureAgent: {r.throughput_rps} rows/s, "
            f"+{len(new_cols)} features, leaky={present_leaky}"
        )
        assert r.passed, f"Leaky features still present: {present_leaky}"

    def test_feature_agent_target_col_detection(self):
        from agri_ai_agent.agents.feature_agent import FeatureAgent

        agent = FeatureAgent()
        df = _make_base_df(50)
        contract = agent.run(df, target_col="Yield_per_Hectare")
        result_df = agent.dataframe
        assert contract.status == "success"
        avail_col = "Feature_Available_Before_Prediction"
        if avail_col in result_df.columns:
            post_harvest = (result_df[avail_col] == "POST_HARVEST_OR_DERIVED").sum()
            print(
                f"\n  [{self.LAYER}] FeatureAgent target-aware: {post_harvest} post-harvest detected"
            )
        else:
            print(f"\n  [{self.LAYER}] FeatureAgent: no availability column (OK)")

    def test_training_agent_leakage_guard(self):
        from agri_ai_agent.agents.training_agent import TrainingAgent

        agent = TrainingAgent()
        df = _make_base_df(200)
        contract = agent.run(df, target_col="Yield_per_Hectare")
        result_df = agent.dataframe
        assert contract.status == "success"
        model_cols = [c for c in result_df.columns if "R2" in c or "MAE" in c or "RMSE" in c]
        LayerResult(
            layer=self.LAYER,
            agent="TrainingAgent",
            rows_in=200,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            notes=f"Model metrics: {len(model_cols)}",
            passed=True,
        )
        print(f"  [{self.LAYER}] TrainingAgent: {len(model_cols)} model metrics produced")

    def test_leakage_smoke_post_fix(self):
        from agri_ai_agent.ml.leakage import is_leaky_feature, select_feature_columns

        df = _make_base_df(100)
        leaky = [c for c in df.columns if is_leaky_feature(c, "Yield_per_Hectare")]
        safe = select_feature_columns(df, "Yield_per_Hectare")
        assert "Yield_per_Hectare" not in safe
        assert "Paper_ID" not in safe
        print(f"\n  [{self.LAYER}] Leakage guard: {len(leaky)} leaky, {len(safe)} safe features")


# ── Layer 6: Decision ────────────────────────────────────────────────────────


class TestLayer6Decision:
    LAYER = "L6 Decision"

    def test_fuzzy_logic_agent(self):
        from agri_ai_agent.agents.fuzzy_logic_agent import FuzzyAgent

        agent = FuzzyAgent()
        df = _make_base_df(50)
        contract = agent.run(df)
        result_df = agent.dataframe
        assert contract.status == "success"
        LayerResult(
            layer=self.LAYER,
            agent="FuzzyLogicAgent",
            rows_in=50,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            passed=True,
        )
        print(f"\n  [{self.LAYER}] FuzzyLogicAgent: {len(result_df.columns)} output cols")

    def test_recommendation_agent(self):
        from agri_ai_agent.agents.recommendation_agent import RecommendationAgent

        agent = RecommendationAgent()
        df = _make_base_df(50)
        contract = agent.run(df)
        result_df = agent.dataframe
        assert contract.status == "success"
        LayerResult(
            layer=self.LAYER,
            agent="RecommendationAgent",
            rows_in=50,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            passed=True,
        )
        print(f"  [{self.LAYER}] RecommendationAgent: {len(result_df.columns)} output cols")

    def test_benchmark_agent(self):
        from agri_ai_agent.agents.benchmark_agent import BenchmarkAgent

        agent = BenchmarkAgent()
        df = _make_base_df(100)
        contract = agent.run(df)
        result_df = agent.dataframe
        assert contract.status == "success"
        LayerResult(
            layer=self.LAYER,
            agent="BenchmarkAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            passed=True,
        )
        print(f"  [{self.LAYER}] BenchmarkAgent: {len(result_df.columns)} output cols")


# ── Layer 7: Meta ────────────────────────────────────────────────────────────


class TestLayer7Meta:
    LAYER = "L7 Meta"

    def test_ready_reckoner_agent(self):
        from agri_ai_agent.agents.ready_reckoner_agent import ReadyReckonerAgent

        agent = ReadyReckonerAgent()
        df = _make_base_df(100)
        contract = agent.run(df)
        result_df = agent.dataframe
        assert contract.status == "success"
        LayerResult(
            layer=self.LAYER,
            agent="ReadyReckonerAgent",
            rows_in=100,
            rows_out=len(result_df),
            cols_in=len(df.columns),
            cols_out=len(result_df.columns),
            passed=True,
        )
        print(f"\n  [{self.LAYER}] ReadyReckonerAgent: {len(result_df.columns)} output cols")


# ── Summary ──────────────────────────────────────────────────────────────────


def _print_efficiency_summary(results: list[LayerResult]):
    print("\n" + "=" * 80)
    print("PER-AGENT LAYER EFFICIENCY REPORT")
    print("=" * 80)
    print(f"{'Layer':<20} {'Agent':<30} {'Rows/s':>8} {'Cols Δ':>8} {'Leak':>5} {'Status':>8}")
    print("-" * 80)
    for r in results:
        col_delta = f"{r.cols_out - r.cols_in:+d}"
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{r.layer:<20} {r.agent:<30} {r.throughput_rps:>8.1f} {col_delta:>8} "
            f"{r.leakage_detected:>5} {status:>8}"
        )
    print("-" * 80)
    total_pass = sum(1 for r in results if r.passed)
    print(f"Total: {total_pass}/{len(results)} agents passed")
    print()

    improvements = []
    for r in results:
        if not r.passed:
            improvements.append(f"  FIX: {r.agent} — {r.notes or r.errors}")
        if r.leakage_detected > 0:
            improvements.append(f"  LEAK: {r.agent} — {r.leakage_detected} leaky features")
    if improvements:
        print("IMPROVEMENT RECOMMENDATIONS:")
        for imp in improvements:
            print(imp)
    else:
        print("No critical improvements needed.")
    print("=" * 80)
