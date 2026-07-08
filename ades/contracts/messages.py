"""
Structured JSON contracts for inter-agent communication.
Every agent accepts an input contract and returns an output contract.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
import json


def _serialize(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


@dataclass
class AgentContract:
    agent_name: str
    status: str = "pending"
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_sec: float = 0.0
    retry_count: int = 0

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_serialize))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_serialize)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentContract":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DocumentUnderstandingResult(AgentContract):
    agent_name: str = "DocumentUnderstandingAgent"
    papers_processed: int = 0
    pages_total: int = 0
    sections_detected: dict = field(default_factory=dict)
    tables_detected: int = 0
    figures_detected: int = 0
    references_detected: int = 0
    document_structures: list[dict] = field(default_factory=list)
    structured_json_path: str = ""
    extraction_time_sec: float = 0.0


@dataclass
class ScientificExtractionResult(AgentContract):
    agent_name: str = "ScientificInformationExtractionAgent"
    papers_extracted: int = 0
    variables_extracted: list[dict] = field(default_factory=list)
    extraction_json_path: str = ""
    extraction_csv_path: str = ""
    report_path: str = ""


@dataclass
class EvidenceValidationResult(AgentContract):
    agent_name: str = "EvidenceValidationAgent"
    facts_validated: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    rejected_facts: int = 0
    report_path: str = ""


@dataclass
class EvidenceTraceabilityResult(AgentContract):
    agent_name: str = "EvidenceTraceabilityAgent"
    variables_traced: int = 0
    traceability_path: str = ""
    provenance_path: str = ""


@dataclass
class IngestionResult(AgentContract):
    agent_name: str = "DatasetIngestionAgent"
    sheet_names: list[str] = field(default_factory=list)
    detected_encoding: str = "utf-8"
    detected_delimiter: str = ","
    row_count: int = 0
    column_count: int = 0
    data_types: dict = field(default_factory=dict)


@dataclass
class SchemaMappingResult(AgentContract):
    agent_name: str = "SchemaMappingAgent"
    mapping: dict = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    schema_version: str = "1.0"


@dataclass
class OntologyMappingResult(AgentContract):
    agent_name: str = "OntologyMappingAgent"
    ontology_map: list[dict] = field(default_factory=list)
    ontology_csv_path: str = ""
    coverage_report_path: str = ""


@dataclass
class UnitHarmonizationResult(AgentContract):
    agent_name: str = "UnitHarmonizationAgent"
    conversions_applied: list[dict] = field(default_factory=list)
    report_path: str = ""


@dataclass
class QualityAssuranceResult(AgentContract):
    agent_name: str = "QualityAssuranceAgent"
    duplicate_rows: int = 0
    duplicate_columns: list[str] = field(default_factory=list)
    impossible_values: list[str] = field(default_factory=list)
    outliers_detected: int = 0
    missing_identifiers: list[str] = field(default_factory=list)
    quality_report_path: str = ""
    validation_report_path: str = ""


@dataclass
class FeatureEngineeringResult(AgentContract):
    agent_name: str = "FeatureEngineeringAgent"
    features_added: list[str] = field(default_factory=list)
    feature_count: int = 0


@dataclass
class LeakageDetectionResult(AgentContract):
    agent_name: str = "LeakageDetectionAgent"
    leaked_features: list[str] = field(default_factory=list)
    safe_features: list[str] = field(default_factory=list)


@dataclass
class EncodingResult(AgentContract):
    agent_name: str = "EncodingAgent"
    encoded_columns: list[str] = field(default_factory=list)
    encoding_map_path: str = ""


@dataclass
class StatisticalDiagnosticsResult(AgentContract):
    agent_name: str = "StatisticalDiagnosticsAgent"
    profile_path: str = ""
    correlation_path: str = ""
    vif_path: str = ""
    high_vif_features: list[str] = field(default_factory=list)


@dataclass
class ModelReadinessResult(AgentContract):
    agent_name: str = "ModelReadinessAgent"
    compatible_models: list[str] = field(default_factory=list)
    incompatible_models: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    report_path: str = ""


@dataclass
class DocumentationResult(AgentContract):
    agent_name: str = "DocumentationAgent"
    generated_files: list[str] = field(default_factory=list)


@dataclass
class TrainingResult(AgentContract):
    agent_name: str = "TrainingAgent"
    models_trained: list[dict] = field(default_factory=list)
    target_variable: str = ""
    training_report_path: str = ""


@dataclass
class PredictionResult(AgentContract):
    agent_name: str = "PredictionAgent"
    predicted_columns: list[str] = field(default_factory=list)
    rows_predicted: int = 0
    models_loaded: int = 0


@dataclass
class ExportResult(AgentContract):
    agent_name: str = "ExportAgent"
    exported_files: list[str] = field(default_factory=list)
    master_xlsx_path: str = ""


@dataclass
class OrchestratorState:
    pipeline_id: str = ""
    status: str = "initialized"
    current_agent: str = ""
    completed_agents: list[str] = field(default_factory=list)
    failed_agents: list[dict] = field(default_factory=list)
    checkpoint_path: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_serialize))
