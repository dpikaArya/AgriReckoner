"""
Orchestrator
Manages execution order of the 11-agent pipeline, passes outputs,
retries failures, logs operations, and supports checkpoint recovery.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.contracts.messages import (
    AgentContract,
    OrchestratorState,
)
from agri_ai_agent.utils.logging_utils import get_logger
from agri_ai_agent import __version__

from agri_ai_agent.agents import (
    KnowledgeAgent,
    ExtractionAgent,
    ValidationAgent,
    EvidenceFusionAgent,
    OntologyAgent,
    TableIntelligenceAgent,
    SchemaPopulationAgent,
    FeatureAgent,
    ModelSelectionAgent,
    TrainingAgent,
    PredictionAgent,
    FuzzyAgent,
    RecommendationAgent,
    BenchmarkAgent,
    ExplainabilityAgent,
    ReadyReckonerAgent,
    ContinuousLearningAgent,
    KnowledgeIntegrationAgent,
    ExternalDataSourceAgent,
)

# Ordered so that external data ingestion runs first (PHASE -1), then
# extraction + schema normalization run BEFORE the knowledge agents
# (which look up canonical UAMS columns), and fuzzy runs AFTER prediction (it consumes
# the model's Yield_Prediction). See the refactor notes for the wiring rationale.
PIPELINE_STEPS = [
    ("external_data", ExternalDataSourceAgent, "External Data Source Layer"),
    ("extraction", ExtractionAgent, "Extract & Schema"),
    ("evidence_fusion", EvidenceFusionAgent, "Evidence Fusion & Provenance"),
    ("ontology", OntologyAgent, "Ontology Mapping & Normalization"),
    ("table_intelligence", TableIntelligenceAgent, "Table Intelligence & Statistics"),
    ("schema_population", SchemaPopulationAgent, "Schema Population & Inference"),
    ("knowledge_integration", KnowledgeIntegrationAgent, "Knowledge Integration & Missing Value Fill"),
    ("knowledge", KnowledgeAgent, "Domain Knowledge"),
    ("validation", ValidationAgent, "Validate & Harmonize"),
    ("feature", FeatureAgent, "Feature Engineering"),
    ("model_selection", ModelSelectionAgent, "Adaptive Model Selection & CV"),
    ("training", TrainingAgent, "Model Training"),
    ("prediction", PredictionAgent, "ML Prediction"),
    ("recommendation", RecommendationAgent, "Generate Recommendations"),
    ("fuzzy", FuzzyAgent, "Fuzzy Expert System"),
    ("benchmark", BenchmarkAgent, "Pipeline Benchmarking"),
    ("explainability", ExplainabilityAgent, "Prediction Explainability"),
    ("ready_reckoner", ReadyReckonerAgent, "Ready Reckoner & Exports"),
]


class Orchestrator:
    def __init__(self, settings: Optional[AgriAISettings] = None):
        self.settings = settings or AgriAISettings()
        self.settings.ensure_dirs()
        self.log = get_logger("Orchestrator")
        self.state = OrchestratorState(
            pipeline_id=datetime.now().strftime("AGRI_%Y%m%d_%H%M%S"),
            started_at=datetime.now(),
        )
        self.dataframe: Optional[pd.DataFrame] = None
        self.results: dict[str, AgentContract] = {}
        self.checkpoint_dir = self.settings.CHECKPOINT_DIR

    def run(self, filepath: Optional[str] = None, df: Optional[pd.DataFrame] = None,
            papers_dir: Optional[str] = None, **kwargs) -> pd.DataFrame:
        self.log.info("=" * 60)
        self.log.info("AgriAI Pipeline [%s]", self.state.pipeline_id)
        self.log.info("=" * 60)

        if filepath:
            self.log.info("Input file: %s", filepath)
        elif df is not None:
            self.log.info("Input DataFrame (%d rows)", len(df))
            self.dataframe = df

        self.state.status = "running"
        self._write_run_manifest(filepath=filepath, papers_dir=papers_dir)

        for step_key, agent_cls, step_name in PIPELINE_STEPS:
            if self.settings.INCREMENTAL_MODE and self._checkpoint_exists(step_key):
                self.log.info("[%s] Skipping (checkpoint exists)", step_name)
                self.dataframe = self._load_checkpoint(step_key)
                continue

            self.state.current_agent = step_key
            self.log.info("\n[STEP] %s (%s)", step_name, step_key)

            try:
                agent = agent_cls(settings=self.settings)

                if step_key == "extraction":
                    contract = agent.run(df=self.dataframe, filepath=filepath, papers_dir=papers_dir)
                elif step_key == "benchmark":
                    contract = agent.run(df=self.dataframe, pipeline_results=self.results)
                elif step_key == "explainability":
                    contract = agent.run(df=self.dataframe, pipeline_results=self.results)
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
                        "step": step_key, "name": step_name,
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
                    "step": step_key, "name": step_name,
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
        self.log.info("AgriAI Pipeline Complete!")
        self.log.info("=" * 60)

        return self.dataframe if self.dataframe is not None else pd.DataFrame()

    def run_continuous(self, papers_dir: str, **kwargs) -> pd.DataFrame:
        self.log.info("Starting continuous learning cycle...")
        agent = ContinuousLearningAgent(settings=self.settings)
        contract = agent.run(df=self.dataframe, papers_dir=papers_dir, **kwargs)
        self.dataframe = agent.dataframe
        self.results["continuous_learning"] = contract
        return self.dataframe

    def _save_checkpoint(self, step_key: str):
        if self.dataframe is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{step_key}.parquet"
        self.dataframe.to_parquet(path, index=False)

    def _load_checkpoint(self, step_key: str) -> pd.DataFrame:
        path = self.checkpoint_dir / f"{step_key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    def _checkpoint_exists(self, step_key: str) -> bool:
        return (self.checkpoint_dir / f"{step_key}.parquet").exists()

    def _should_continue_on_failure(self, step_key: str) -> bool:
        critical = {"extraction", "training"}
        return step_key not in critical

    def _write_run_manifest(self, filepath=None, papers_dir=None):
        """Snapshot the run's identity for reproducibility: version, git SHA, config, input."""
        run_dir = self.settings.OUTPUT_DIR / "runs" / self.state.pipeline_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "pipeline_id": self.state.pipeline_id,
            "package_version": __version__,
            "git_sha": self._git_sha(),
            "started_at": str(self.state.started_at),
            "input": {
                "filepath": filepath,
                "papers_dir": papers_dir,
                "df_rows": None if self.dataframe is None else len(self.dataframe),
            },
            "settings": {
                "incremental_mode": self.settings.INCREMENTAL_MODE,
                "checkpoint_enabled": self.settings.CHECKPOINT_ENABLED,
                "output_dir": str(self.settings.OUTPUT_DIR),
                "llm_model": getattr(self.settings, "LLM_MODEL", None),
            },
        }
        with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    @staticmethod
    def _git_sha():
        """Best-effort current git SHA; None if unavailable."""
        import subprocess
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or None
        except Exception:
            return None

    def _write_provenance(self):
        provenance = {
            "pipeline_id": self.state.pipeline_id,
            "pipeline_version": __version__,
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
