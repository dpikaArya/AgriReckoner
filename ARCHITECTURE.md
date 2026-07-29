# Architecture — Agricultural Intelligence Framework

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCE LAYER                        │
│  CGIAR · FAOSTAT · NASA POWER · SoilGrids · ISRIC · Zenodo · Mendeley   │
│  Kaggle · ICAR · State Agricultural Universities · Future connectors    │
│  Plugin-based: ExternalDataConnector interface                          │
│  connect() → discover() → download() → validate() → register()         │
│  Output: standardized DatasetPackage objects                            │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                    CORE PIPELINE (23 sequential agents)                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 0: INGESTION & NORMALIZATION                              │    │
│  │  RepositorySync → ExternalData → DatasetNormalization →          │    │
│  │  DatasetIngestionBridge (ID assignment)                          │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: EXTRACTION & KNOWLEDGE                                │    │
│  │  Extraction (6 readers) → EvidenceFusion → Ontology →           │    │
│  │  TableIntelligence → SchemaPopulation → KnowledgeIntegration →  │    │
│  │  Knowledge → ObservationGeneration                              │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: VALIDATION & FEATURES                                 │    │
│  │  Validation → FeatureStore (gate) → Feature (engineering)       │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: ML TRAINING & PREDICTION                              │    │
│  │  ModelSelection → Training → Prediction                         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 4: RECOMMENDATION & EXPORT                               │    │
│  │  Recommendation → Fuzzy (221 rules) → Benchmark →               │    │
│  │  Explainability → ReadyReckoner                                 │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  CONTINUOUS LEARNING (run_continuous)                            │    │
│  │  IncrementalEngine · ChangeDetector · RepositoryRegistry ·      │    │
│  │  VersionHistory · DependencyGraph                                │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
agri_ai_agent/                  # Core agent framework
├── agents/                     # 29 agent implementations
│   ├── base_agent.py           # Abstract base with retry + artifact saving
│   ├── extraction_agent.py     # 6-reader hybrid PDF extraction
│   ├── evidence_fusion_agent.py # Confidence-weighted multi-reader merge
│   ├── ontology_agent.py       # Column name → UAMS normalisation
│   ├── table_intelligence_agent.py  # Table type classification
│   ├── schema_population_agent.py   # Derived column computation
│   ├── knowledge_integration_agent.py  # Domain-knowledge imputation
│   ├── knowledge_agent.py      # PDF ingestion + paper registry
│   ├── validation_agent.py     # Range checks, unit harmonisation, quality
│   ├── feature_agent.py        # 16+ composite feature engineering
│   ├── feature_store_agent.py  # Validation gate + parquet persistence
│   ├── model_selection_agent.py # Adaptive pool + GridSearchCV + nested CV
│   ├── training_agent.py       # Model training + evaluation + export
│   ├── prediction_agent.py     # Load best models, predict targets
│   ├── recommendation_agent.py  # Top-3 treatments with confidence
│   ├── fuzzy_logic_agent.py    # 221 Mamdani rules
│   ├── benchmark_agent.py      # Phase-by-phase timing + metrics
│   ├── explainability_agent.py  # Feature-attribution explanations
│   ├── ready_reckoner_agent.py  # Excel/CSV/HTML/JSON exports
│   ├── continuous_learning_agent.py  # Drift detection + retraining
│   ├── repository_sync_agent.py  # Change detection for data sources
│   ├── external_data_source_agent.py # Plugin-based source ingestion
│   ├── dataset_normalization_agent.py # Normalise to DatasetPackage
│   ├── dataset_ingestion_bridge_agent.py # ID hierarchy assignment
│   ├── observation_generation_agent.py # Observation hierarchy build
│   ├── provenance_agent.py     # Per-cell lineage (optional)
│   └── llm_extraction_agent.py # OpenAI-backed extraction (optional)
├── config/
│   ├── schema.py               # UAMS v2.0: 154+ columns, 26 groups (A-Z)
│   │                           # VARIANT_MAP: 700+ name variants
│   └── settings.py             # AgriAISettings: all paths + params
├── contracts/
│   └── messages.py             # AgentContract dataclass + 20+ subclasses
├── external_data/
│   ├── connector.py            # ExternalDataConnector ABC (7 methods)
│   ├── connector_manager.py    # Orchestrates multi-source parallel fetch
│   ├── dataset_package.py      # Standardised DatasetPackage dataclass
│   ├── registry.py             # ConnectorRegistry (pkgutil discovery)
│   ├── registry_db.py          # DatasetRegistry (SQLite persistence)
│   └── connectors/             # 10+ source-specific implementations
├── knowledge_graph/
│   ├── graph.py                # NetworkX DiGraph: 12 node types, 12 edges
│   └── provenance.py           # Provenance enrichment for graph
├── ml/
│   ├── leakage.py              # Name-based + correlation-based leak detection
│   └── evaluation.py           # Honest small-n CV with imputation inside folds
├── continuous_learning/
│   ├── incremental_engine.py   # Change-driven partial pipeline execution
│   ├── repository_registry.py  # Source version tracking
│   ├── version_history.py      # Model + dataset versioning
│   ├── change_detector.py      # Diff-based change detection
│   └── dependency_graph.py     # Step dependency resolution
├── orchestrator.py             # Pipeline controller: 23-step sequence
├── cli.py                      # `agriai` console entry point
└── utils/
    ├── io_utils.py             # File I/O helpers
    └── logging_utils.py        # Structured logging

src/                            # Supporting infrastructure
├── data_sources/               # External data integration utilities
│   ├── config_loader.py        # YAML config loading with env override
│   ├── integration.py          # DataSourceIntegration coordinator
│   ├── storage.py              # ImmutableStorage (content-addressed)
│   └── sync.py                 # SyncManager (checksum-based dedup)
├── ml/
│   ├── pre_training_checks.py  # PreTrainingChecks (7 gates)
│   └── experiment_tracker.py   # MLflow wrapper with local fallback
├── provenance/
│   └── lineage_tracker.py      # Per-cell data lineage tracking
├── validation/
│   ├── schema_validator.py     # UAMS schema compliance checks
│   ├── range_validator.py      # Biological range enforcement
│   ├── data_quality.py         # Quality metric computation
│   ├── provenance_validator.py # Provenance integrity checks
│   └── reports.py             # Validation report generation
└── utils/
    └── logging_config.py       # Logging configuration

config/                         # Runtime configuration
├── apis.yaml                   # External source definitions (10 sources)
├── benchmark.yaml              # Benchmark thresholds
└── explainability.yaml         # Explainability settings

tests/                          # 40+ test files covering all agents
evaluation/                     # 11-stage automated evaluation suite
benchmarks/                     # Gold standard benchmarking data
```

## Core Components

### Agents

Every agent extends `BaseAgent` (ABC) and implements:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `agent_name` | `→ str` | Unique identifier |
| `process` | `(df, **kwargs) → pd.DataFrame` | Core logic, transforms the cumulative DataFrame |
| `run` | `(df, contract, **kwargs) → AgentContract` | Retry loop wrapping `process`, saves artifacts |

The `BaseAgent` provides:
- Configurable retry (`AGENT_RETRY_MAX`, `AGENT_RETRY_DELAY_SEC`)
- `save_artifact(df, filename)` — saves CSV/XLSX/Parquet to output dir
- `save_text_artifact(text, filename)` — saves text/MD/HTML artifacts
- Structured logging via `get_logger()`

### Orchestrator

The `Orchestrator` (`agri_ai_agent/orchestrator.py`) manages:

- **Sequential execution** — iterates `PIPELINE_STEPS` in strict order (23 steps)
- **State tracking** — `OrchestratorState`: pipeline_id, status, completed/failed agents
- **Checkpoint/recovery** — per-agent Parquet checkpoints in `.checkpoints/`
- **Incremental mode** — skips agents with existing checkpoints
- **Failure handling** — critical steps (extraction, training) halt; non-critical continue with warnings
- **Provenance** — writes `run_manifest.json` and `Pipeline_Provenance.json` per run

### Knowledge Graph

`KnowledgeGraph` (`agri_ai_agent/knowledge_graph/graph.py`) — NetworkX `DiGraph`:

| Node Types (12) | Edge Types (12) |
|-----------------|-----------------|
| Paper, Treatment, Crop, Observation, Yield, Soil, Climate, Management, Repository, Dataset, Evidence, Document | DESCRIBES, TREATS, GROWS, OBSERVES, YIELDS, HAS_SOIL, HAS_CLIMATE, MANAGES, PROVIDES, CONTAINS, SUPPORTED_BY, FROM_SOURCE |

Supports: centrality (degree, betweenness, in/out-degree), shortest path, PageRank, subgraph extraction, JSON serialization, and automatic construction from DataFrame via `build_from_dataframe()`.

### External Data Layer

Plugin-based ingestion system (`agri_ai_agent/external_data/`). The `ExternalDataConnector` ABC defines:

```python
class ExternalDataConnector(ABC):
    def connect(self) -> bool: ...
    def discover(self, query: str | None) -> list[dict]: ...
    def download(self, resource_id: str, target_dir: Path) -> Path | None: ...
    def validate(self, package: DatasetPackage) -> bool: ...
    def register(self, package: DatasetPackage) -> str: ...
    def update(self) -> int: ...
    def close(self) -> None: ...
```

Connectors are auto-discovered via `pkgutil` — no registration needed. Supported sources: CGIAR, FAOSTAT, NASA POWER, SoilGrids, ISRIC, Zenodo, Mendeley Data, Kaggle, ICAR, State Agricultural Universities.

## Agent Communication (AgentContract)

Every agent accepts and returns an `AgentContract` dataclass:

```python
@dataclass
class AgentContract:
    agent_name: str
    status: str                # pending → running → success / failed
    input_data: dict           # Input configuration
    output_data: dict          # Output summary (rows, columns processed)
    artifacts: list[str]       # Generated file paths
    errors: list[str]          # Error traces per attempt
    warnings: list[str]        # Non-fatal warnings
    metadata: dict             # Agent-specific metrics
    started_at: datetime | None
    completed_at: datetime | None
    execution_time_sec: float
    retry_count: int
    dataset_id: str
    provider: str
    repository_version: str
    connector_name: str
    checksum: str
    processing_stage: str
    processing_mode: str       # "full" or "incremental"
```

Specialised subclasses: `TrainingResult`, `PredictionResult`, `ExportResult`, `IngestionResult`, `SchemaMappingResult`, `OntologyMappingResult`, `QualityAssuranceResult`, `FeatureEngineeringResult`, `LeakageDetectionResult`, `ModelReadinessResult`, `DocumentationResult`, etc.

## Pipeline Execution Flow

```
Orchestrator.run()
│
├─ 1. repository_sync      — Repository Sync & Change Detection
├─ 2. external_data        — External Data Source Layer (plugin-based)
├─ 3. dataset_normalization  — Normalise to DatasetPackage format
├─ 4. dataset_ingestion_bridge — Assign Dataset_ID / Document_ID / Paper_ID /
│                                   Experiment_ID / Treatment_ID / Observation_ID
├─ 5. extraction           — 6-reader hybrid PDF extraction
├─ 6. evidence_fusion      — Confidence-weighted merge & deduplication
├─ 7. ontology              — Column name → UAMS normalisation
├─ 8. table_intelligence   — Table type classification
├─ 9. schema_population    — Derived column computation
├─10. knowledge_integration — Domain-based missing value fill
├─11. knowledge             — Domain knowledge constraints
├─12. observation_generation — Build observation hierarchy
├─13. validation            — Biological range, unit harmonisation, quality
├─14. feature_store         — Validation gate + Parquet persistence
├─15. feature               — 146 engineered features
├─16. model_selection       — Adaptive pool + GridSearchCV
├─17. training              — Model training + evaluation
├─18. prediction            — Load best models, generate predictions
├─19. recommendation        — Top-3 treatment alternatives
├─20. fuzzy                 — Mamdani inference (221 rules)
├─21. benchmark             — Phase-by-phase metrics
├─22. explainability        — Feature-attribution explanations
├─23. ready_reckoner        — Excel/CSV/HTML/JSON exports
│
└─ (optional) run_continuous — Drift detection, incremental retraining
```

## Configuration Management

- **`AgriAISettings`** (`agri_ai_agent/config/settings.py`): All directory paths, retry parameters, LLM settings, export formats, checkpoint and incremental mode flags.
- **`config/apis.yaml`**: External data source definitions — URLs, strategies, auth, priorities, timeouts. Overridable via `AGRI_{SOURCE}_{KEY}` env vars.
- **`config/benchmark.yaml`**: Benchmark thresholds and performance targets.
- **`config/explainability.yaml`**: SHAP/permutation importance configuration.
- **`agri_ai_agent/config/schema.py`**: UAMS v2.0 — 154+ canonical columns, 26 groups (A-Z), 700+ entry `VARIANT_MAP` for column name normalisation, plus helper sets (`NON_FEATURE_COLS`, `POST_HARVEST_VARIABLES`, `PRE_HARVEST_MEASUREMENTS`, `NUMERIC_UAMS_COLUMNS`).

## Extension Points

### New Agents

1. Create a new class extending `BaseAgent` in `agri_ai_agent/agents/`
2. Implement `agent_name` property and `process(df, **kwargs) → pd.DataFrame`
3. Add the step tuple `(key, AgentClass, "Display Name")` to `PIPELINE_STEPS` in `orchestrator.py`
4. Import the class in `orchestrator.py`

### New External Data Connectors

1. Create a file in `agri_ai_agent/external_data/connectors/`
2. Subclass `ExternalDataConnector` and implement all 7 methods
3. Add source config to `config/apis.yaml`
4. The connector is auto-discovered — no registration code needed

### New Validators

Validation rules are defined as dictionaries in `ValidationAgent`:
- `RANGE_CONSTRAINTS`: `{column: (min, max)}` for hard bounds
- `BIOLOGICAL_RULES`: `{column: {min, max, unit, crop_ranges}}` for crop-aware checks
- `AGRONOMIC_RULES`: `[{name, condition, check, message, severity}]` for cross-field logic
