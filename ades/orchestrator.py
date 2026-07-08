"""
Orchestrator Agent
Manages execution order, passes outputs between agents, retries failures,
logs operations, maintains provenance, supports checkpoint recovery and
incremental execution for the full 13-agent pipeline.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ades.config.settings import ADESSettings
from ades.contracts.messages import (
    AgentContract,
    OrchestratorState,
)
from ades.utils.logging_utils import get_logger

from ades.agents import (
    DatasetIngestionAgent,
    DocumentUnderstandingAgent,
    ScientificInformationExtractionAgent,
    EvidenceValidationAgent,
    EvidenceTraceabilityAgent,
    OntologyMappingAgent,
    SchemaMappingAgent,
    UnitHarmonizationAgent,
    QualityAssuranceAgent,
    FeatureEngineeringAgent,
    LeakageDetectionAgent,
    EncodingAgent,
    StatisticalDiagnosticsAgent,
    ModelReadinessAgent,
    DocumentationAgent,
    ExportAgent,
    TrainingAgent,
    PredictionAgent,
    FuzzyAgent,
    KnowledgeAgent,
    RecommendationAgent,
    ReadyReckonerAgent,
    ContinuousLearningAgent,
)


PIPELINE_STEPS = [
    ("ingestion", DatasetIngestionAgent, "Dataset Ingestion"),
    ("document_understanding", DocumentUnderstandingAgent, "Document Understanding"),
    ("knowledge", KnowledgeAgent, "Domain Knowledge"),
    ("scientific_extraction", ScientificInformationExtractionAgent, "Scientific Information Extraction"),
    ("evidence_validation", EvidenceValidationAgent, "Evidence Validation"),
    ("evidence_traceability", EvidenceTraceabilityAgent, "Evidence Traceability"),
    ("ontology_mapping", OntologyMappingAgent, "Ontology Mapping"),
    ("schema_mapping", SchemaMappingAgent, "Schema Mapping"),
    ("unit_harmonization", UnitHarmonizationAgent, "Unit Harmonization"),
    ("quality_assurance", QualityAssuranceAgent, "Quality Assurance"),
    ("feature_engineering", FeatureEngineeringAgent, "Feature Engineering"),
    ("leakage_detection", LeakageDetectionAgent, "Leakage Detection"),
    ("encoding", EncodingAgent, "Encoding"),
    ("statistical_diagnostics", StatisticalDiagnosticsAgent, "Statistical Diagnostics"),
    ("model_readiness", ModelReadinessAgent, "Model Readiness"),
    ("documentation", DocumentationAgent, "Documentation"),
    ("export", ExportAgent, "Export"),
    ("training", TrainingAgent, "Model Training"),
    ("fuzzy", FuzzyAgent, "Fuzzy Expert System"),
    ("prediction", PredictionAgent, "ML Prediction"),
    ("recommendation", RecommendationAgent, "Generate Recommendations"),
    ("ready_reckoner", ReadyReckonerAgent, "Ready Reckoner & Exports"),
    ("continuous_learning", ContinuousLearningAgent, "Continuous Learning"),
]


class Orchestrator:
    def __init__(self, settings: Optional[ADESSettings] = None):
        self.settings = settings or ADESSettings()
        self.settings.ensure_dirs()
        self.log = get_logger("Orchestrator")
        self.state = OrchestratorState(
            pipeline_id=datetime.now().strftime("ADES_%Y%m%d_%H%M%S"),
            started_at=datetime.now(),
        )
        self.dataframe: Optional[pd.DataFrame] = None
        self.results: dict[str, AgentContract] = {}
        self.contracts_dir = self.settings.CONTRACTS_DIR
        self.checkpoint_dir = self.settings.CHECKPOINT_DIR

    def run(self, filepath: Optional[str] = None, df: Optional[pd.DataFrame] = None,
            papers_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
        self.log.info("=" * 60)
        self.log.info("ADES v2.0 Pipeline [%s]", self.state.pipeline_id)
        self.log.info("=" * 60)

        if filepath:
            self.log.info("Input file: %s", filepath)
        elif df is not None:
            self.log.info("Input DataFrame (%d rows)", len(df))
        if papers_dir:
            self.log.info("Papers directory: %s", papers_dir)
        if not filepath and df is None and not papers_dir:
            raise ValueError("Either filepath, df, or papers_dir must be provided")

        self.state.status = "running"

        for step_key, agent_cls, step_name in PIPELINE_STEPS:
            if self.settings.INCREMENTAL_MODE and self._checkpoint_exists(step_key):
                self.log.info("[%s] Skipping (checkpoint exists)", step_name)
                self.dataframe = self._load_checkpoint(step_key)
                continue

            self.state.current_agent = step_key
            self.log.info("\n[STEP] %s (%s)", step_name, step_key)

            try:
                agent = agent_cls(settings=self.settings)

                if step_key == "ingestion":
                    contract = agent.run(df=df, filepath=filepath)
                elif step_key in ("document_understanding", "scientific_extraction"):
                    contract = agent.run(filepath=papers_dir or filepath, df=self.dataframe)
                elif step_key == "continuous_learning":
                    contract = agent.run(df=self.dataframe, papers_dir=papers_dir)
                elif step_key in ("evidence_validation", "evidence_traceability"):
                    contract = agent.run(df=self.dataframe)
                else:
                    contract = agent.run(df=self.dataframe)

                self.results[step_key] = contract

                if contract.status == "success":
                    if hasattr(agent, 'dataframe') and agent.dataframe is not None:
                        self.dataframe = agent.dataframe
                    self.state.completed_agents.append(step_key)
                    if self.settings.CHECKPOINT_ENABLED:
                        self._save_checkpoint(step_key)
                    self.log.info("[%s] OK (%.2fs)", step_name, contract.execution_time_sec)
                else:
                    self.state.failed_agents.append({
                        "step": step_key,
                        "name": step_name,
                        "errors": contract.errors,
                    })
                    self.log.error("[%s] FAILED after %d attempts", step_name, contract.retry_count + 1)
                    if not self._should_continue_on_failure(step_key):
                        self.state.status = "failed"
                        self.state.completed_at = datetime.now()
                        raise RuntimeError(
                            f"Pipeline failed at {step_name}: "
                            f"{contract.errors[-1] if contract.errors else 'unknown'}"
                        )

            except Exception as e:
                self.log.error("[%s] Exception: %s", step_name, e)
                self.state.failed_agents.append({
                    "step": step_key,
                    "name": step_name,
                    "errors": [traceback.format_exc()],
                })
                if not self._should_continue_on_failure(step_key):
                    self.state.status = "failed"
                    self.state.completed_at = datetime.now()
                    raise

        self.state.status = "completed"
        self.state.completed_at = datetime.now()
        self._write_provenance()

        self.log.info("\n" + "=" * 60)
        self.log.info("ADES Pipeline Complete!")
        self.log.info("Output files in: %s", self.settings.OUTPUT_DIR)
        self.log.info("=" * 60)

        return self.dataframe if self.dataframe is not None else pd.DataFrame()

    def _save_checkpoint(self, step_key: str):
        if self.dataframe is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{step_key}.parquet"
        self.dataframe.to_parquet(path, index=False)
        self.log.debug("Checkpoint saved: %s", path)

    def _load_checkpoint(self, step_key: str) -> pd.DataFrame:
        path = self.checkpoint_dir / f"{step_key}.parquet"
        if path.exists():
            self.log.info("Loading checkpoint: %s", path)
            return pd.read_parquet(path)
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    def _checkpoint_exists(self, step_key: str) -> bool:
        return (self.checkpoint_dir / f"{step_key}.parquet").exists()

    def _should_continue_on_failure(self, step_key: str) -> bool:
        critical = {"ingestion", "schema_mapping", "export"}
        return step_key not in critical

    def _write_provenance(self):
        provenance = {
            "pipeline_id": self.state.pipeline_id,
            "pipeline_version": "2.0.0",
            "status": self.state.status,
            "started_at": str(self.state.started_at),
            "completed_at": str(self.state.completed_at),
            "completed_agents": self.state.completed_agents,
            "failed_agents": self.state.failed_agents,
            "results": {
                k: {
                    "status": v.status,
                    "execution_time_sec": v.execution_time_sec,
                    "retry_count": v.retry_count,
                    "errors": v.errors,
                    "artifacts": v.artifacts,
                }
                for k, v in self.results.items()
            },
        }
        path = self.settings.OUTPUT_DIR / "Pipeline_Provenance.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, default=str)
        self.log.info("Provenance written: %s", path)
