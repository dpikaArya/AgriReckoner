# UAMS Data Contracts — JSON Schema Specification

**Version:** 1.0  
**Purpose:** Structured contracts for inter-agent communication in the ADES pipeline

---

## Contract Architecture

Every agent in the ADES pipeline communicates exclusively through `AgentContract` objects. These contracts ensure:

- **Deterministic interfaces**: Input/output contracts are strictly typed
- **Provenance tracking**: Every operation is logged with timestamps, retry counts, and error details
- **Error recovery**: Failed agent outputs include structured error information
- **Serialization**: Contracts are JSON-serializable for persistence and debugging

---

## AgentContract (Base Schema)

```json
{
  "agent_name": "string",
  "status": "pending | running | success | failed",
  "input_data": {},
  "output_data": {},
  "artifacts": ["filepath1", "filepath2"],
  "errors": ["error description"],
  "warnings": ["warning description"],
  "started_at": "ISO datetime",
  "completed_at": "ISO datetime",
  "execution_time_sec": 0.0,
  "retry_count": 0
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_name` | string | Yes | Name of the agent that produced this contract |
| `status` | enum | Yes | Current execution status |
| `input_data` | object | No | Input parameters |
| `output_data` | object | No | Execution result summary |
| `artifacts` | [string] | No | File paths of generated artifacts |
| `errors` | [string] | No | Error messages from failed attempts |
| `warnings` | [string] | No | Non-fatal warning messages |
| `started_at` | datetime (ISO 8601) | No | Execution start time |
| `completed_at` | datetime (ISO 8601) | No | Execution completion time |
| `execution_time_sec` | number | No | Wall-clock execution time |
| `retry_count` | integer | No | Number of retry attempts |

---

## Agent-Specific Contracts

### 1. IngestionResult

```json
{
  "agent_name": "DatasetIngestionAgent",
  "status": "success",
  "sheet_names": ["Sheet1"],
  "detected_encoding": "utf-8",
  "detected_delimiter": ",",
  "row_count": 1000,
  "column_count": 50,
  "data_types": {"col1": "float64", "col2": "object"},
  "output_data": {"rows": 1000, "columns": ["col1", "col2"], "column_count": 50}
}
```

### 2. SchemaMappingResult

```json
{
  "agent_name": "SchemaMappingAgent",
  "status": "success",
  "mapping": {"tmax_c": "Temperature_Max"},
  "unmapped_columns": ["custom_var"],
  "schema_version": "1.0",
  "output_data": {"rows": 1000, "columns": ["Temperature_Max", ...], "column_count": 128}
}
```

### 3. OntologyMappingResult

```json
{
  "agent_name": "OntologyMappingAgent",
  "status": "success",
  "ontology_map": [
    {"Variable": "Crop", "Ontology_ID": "AGROVOC:c_2556", "Preferred_Label": "Crop", "Synonyms": "crop type", "Source": "AGROVOC"}
  ],
  "ontology_csv_path": "outputs/Ontology_Mapping.csv"
}
```

### 4. UnitHarmonizationResult

```json
{
  "agent_name": "UnitHarmonizationAgent",
  "status": "success",
  "conversions_applied": [
    {"from": "Yield_per_Acre", "to": "Yield_per_Hectare", "factor": 2.47105, "rows_affected": 500}
  ],
  "report_path": "outputs/Unit_Conversion_Report.md"
}
```

### 5. QualityAssuranceResult

```json
{
  "agent_name": "QualityAssuranceAgent",
  "status": "success",
  "duplicate_rows": 0,
  "duplicate_columns": [],
  "impossible_values": ["Soil_pH: 3 value(s) outside [0, 14]"],
  "outliers_detected": 45,
  "missing_identifiers": [],
  "quality_report_path": "outputs/Quality_Report.md",
  "validation_report_path": "outputs/Validation_Report.csv"
}
```

### 6. FeatureEngineeringResult

```json
{
  "agent_name": "FeatureEngineeringAgent",
  "status": "success",
  "features_added": ["Growing_Degree_Days", "Heat_Units", "NUE", "WUE"],
  "feature_count": 16
}
```

### 7. LeakageDetectionResult

```json
{
  "agent_name": "LeakageDetectionAgent",
  "status": "success",
  "leaked_features": ["Protein", "Ash", "Yield_per_Plot"],
  "safe_features": ["Crop", "Season", "Temperature_Max"]
}
```

### 8. EncodingResult

```json
{
  "agent_name": "EncodingAgent",
  "status": "success",
  "encoded_columns": ["Crop_Code", "Season_Code"],
  "encoding_map_path": "outputs/Encoding_Map.csv"
}
```

### 9. StatisticalDiagnosticsResult

```json
{
  "agent_name": "StatisticalDiagnosticsAgent",
  "status": "success",
  "profile_path": "outputs/Statistical_Profile.csv",
  "correlation_path": "outputs/Correlation_Matrix.csv",
  "vif_path": "outputs/VIF_Report.csv",
  "high_vif_features": ["Nitrogen", "Organic_Carbon"]
}
```

### 10. ModelReadinessResult

```json
{
  "agent_name": "ModelReadinessAgent",
  "status": "success",
  "compatible_models": ["Random Forest", "XGBoost", "CatBoost"],
  "incompatible_models": [
    {"model": "LSTM", "issues": ["no time/date column for indexing"]}
  ],
  "issues": ["Unencoded categorical variables present"],
  "report_path": "outputs/Model_Readiness_Report.md"
}
```

### 11. DocumentationResult

```json
{
  "agent_name": "DocumentationAgent",
  "status": "success",
  "generated_files": [
    "outputs/Feature_Dictionary.csv",
    "outputs/Missing_Data_Report.md",
    "outputs/Pipeline_Log.md"
  ]
}
```

### 12. ExportResult

```json
{
  "agent_name": "ExportAgent",
  "status": "success",
  "exported_files": [
    "outputs/Universal_Agricultural_ML_Master.csv",
    "outputs/Universal_Agricultural_ML_Master.parquet",
    "outputs/Universal_Agricultural_Machine_Learning_Schema_v1.xlsx"
  ],
  "master_xlsx_path": "outputs/Universal_Agricultural_Machine_Learning_Schema_v1.xlsx"
}
```

---

## OrchestratorState

```json
{
  "pipeline_id": "ADES_20260704_171500",
  "status": "completed",
  "current_agent": "",
  "completed_agents": ["ingestion", "schema_mapping", "ontology_mapping", ...],
  "failed_agents": [],
  "checkpoint_path": ".checkpoints/",
  "started_at": "2026-07-04T17:15:00",
  "completed_at": "2026-07-04T17:16:30"
}
```

---

## Pipeline Provenance Record

```json
{
  "pipeline_id": "ADES_20260704_171500",
  "status": "completed",
  "started_at": "2026-07-04T17:15:00",
  "completed_at": "2026-07-04T17:16:30",
  "completed_agents": ["ingestion", ...],
  "failed_agents": [],
  "results": {
    "ingestion": {
      "status": "success",
      "execution_time_sec": 1.2,
      "retry_count": 0,
      "errors": [],
      "artifacts": ["outputs/..."],
      "output_data": {"rows": 1000, "columns": 50}
    }
  }
}
```

---

## Contract Validation Rules

1. **Required fields**: `agent_name` and `status` must always be present
2. **Status transitions**: `pending → running → success | failed`
3. **Timestamps**: `started_at` and `completed_at` must be valid ISO 8601
4. **Execution time**: Must be non-negative
5. **Retry count**: Must be non-negative, less than configured max
6. **Artifacts**: All artifact paths must be resolvable files on disk
7. **Error format**: Each error should be a human-readable string with context

## Implementation

Contracts are implemented as Python dataclasses in `agri_ai_agent/contracts/messages.py`:

```python
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

    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
```

Subclasses extend `AgentContract` with agent-specific fields while inheriting the base contract structure.
