# Architecture — Agentic Agricultural Intelligence Framework (AAIF)

## Overview

The AAIF is a **17-agent sequential pipeline** that converts unstructured agricultural research PDFs into structured, ML-ready datasets and actionable fertilizer recommendations. Each agent extends `BaseAgent`, communicates via typed `AgentContract` messages, and is orchestrated by a central `Orchestrator` with checkpoint/recovery, retry logic, and provenance logging.

## Agent Pipeline

```
Master Datasets ─► 1. Extraction ─► 2. Evidence Fusion ─► 3. Ontology ─►
4. Table Intelligence ─► 5. Schema Population ─► 6. Knowledge Integration ─►
7. Knowledge ─► 8. Validation ─► 9. Feature Engineering ─►
10. Model Selection ─► 11. Training ─► 12. Prediction ─►
13. Recommendation ─► 14. Fuzzy Logic ─► 15. Benchmark ─►
16. Explainability ─► 17. Ready Reckoner
```

| # | Agent | Class | Purpose |
|---|-------|-------|---------|
| 1 | **ExtractionAgent** | `extraction_agent.py` | 6-reader hybrid PDF extraction (Pdfminer, Camelot, Pdfplumber, Poppler, OCR, Semantic) with regex pattern matching for agronomic variables |
| 2 | **EvidenceFusionAgent** | `evidence_fusion_agent.py` | Confidence-weighted multi-reader deduplication and cell-level conflict resolution |
| 3 | **OntologyAgent** | `ontology_agent.py` | Normalises heterogeneous column names to UAMS via a 120+ entry master column variant map |
| 4 | **TableIntelligenceAgent** | `table_intelligence_agent.py` | Table type classification, summary statistics computation |
| 5 | **SchemaPopulationAgent** | `schema_population_agent.py` | Derived column computation and missing value inference |
| 6 | **KnowledgeIntegrationAgent** | `knowledge_integration_agent.py` | Domain-knowledge-based missing value imputation |
| 7 | **KnowledgeAgent** | `knowledge_agent.py` | Agronomic domain constraint application |
| 8 | **ValidationAgent** | `validation_agent.py` | Biological range enforcement (pH∈[3,10], yield∈[0,50000]), IQR outlier detection (3×IQR), OCR artefact flagging |
| 9 | **FeatureAgent** | `feature_agent.py` | 16+ composite features: NPK Index, Soil Fertility Index, Climate Index, Growing Degree Days, NUE, WUE, polynomial terms (→146 total cols) |
| 10 | **ModelSelectionAgent** | `model_selection_agent.py` | Adaptive model pool selection by sample size, SelectKBest feature selection, GridSearchCV |
| 11 | **TrainingAgent** | `training_agent.py` | 8 regression families (Linear, Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, SVR) with 5-fold CV / LOO |
| 12 | **PredictionAgent** | `prediction_agent.py` | Loads best models, generates predictions for 5 targets |
| 13 | **RecommendationAgent** | `recommendation_agent.py` | Per-crop top-3 treatment alternatives with confidence scoring |
| 14 | **FuzzyAgent** | `fuzzy_logic_agent.py` | 221 Mamdani rules (v3.0), 10 trapezoidal/triangular input MFs, 3 output MFs, centroid defuzzification |
| 15 | **BenchmarkAgent** | `benchmark_agent.py` | Phase-by-phase timing, data quality and coverage metrics |
| 16 | **ExplainabilityAgent** | `explainability_agent.py` | Feature-attribution analysis per prediction |
| 17 | **ReadyReckonerAgent** | `ready_reckoner_agent.py` | Excel/CSV/HTML/JSON per-crop decision-support table export |

### Additional Agents (not in default pipeline)

| Agent | Class | Purpose |
|-------|-------|---------|
| **ProvenanceAgent** | `provenance_agent.py` | Per-cell data lineage tracking from source PDF to final schema |
| **LLMExtractionAgent** | `llm_extraction_agent.py` | Optional OpenAI-backed LLM extraction with source grounding and unit-aware validation |

## Orchestrator

**File:** `agri_ai_agent/orchestrator.py`

The `Orchestrator` class:
- Maintains an `OrchestratorState` with pipeline ID, status, completed/failed agent lists
- Iterates through `PIPELINE_STEPS` (ordered tuple of (key, class, name))
- Supports **incremental mode**: skips agents with existing Parquet checkpoints
- Supports **checkpoint recovery**: saves per-agent state to `.checkpoints/<step>.parquet`
- **Critical steps** (extraction, training): halt pipeline on failure
- **Non-critical steps**: continue with warnings on failure
- Writes `run_manifest.json` and `Pipeline_Provenance.json` per run

## Contract Protocol

**File:** `agri_ai_agent/contracts/messages.py`

Every agent accepts and returns an `AgentContract` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Agent identifier |
| `status` | `str` | `pending` → `running` → `success` / `failed` |
| `input_data` | `dict` | Input configuration |
| `output_data` | `dict` | Output summary (rows, columns) |
| `artifacts` | `list[str]` | Generated file paths |
| `errors` | `list[str]` | Per-attempt error traces |
| `warnings` | `list[str]` | Non-fatal warnings |
| `metadata` | `dict` | Agent-specific metadata |
| `started_at` | `Optional[datetime]` | Start timestamp |
| `completed_at` | `Optional[datetime]` | End timestamp |
| `execution_time_sec` | `float` | Wall-clock execution time |
| `retry_count` | `int` | Number of retry attempts |

Subclasses: `TrainingResult`, `PredictionResult`, `ExportResult`, `IngestionResult`, `SchemaMappingResult`, `FeatureEngineeringResult`, etc.

## Base Agent

**File:** `agri_ai_agent/agents/base_agent.py`

Abstract class `BaseAgent(ABC)`:
- Attributes: `settings`, `retry_max`, `retry_delay`, `logger`, `contract`, `dataframe`
- `run(df, contract, **kwargs)`: Retry loop with configurable max attempts and delay
- `process(df, **kwargs)` → `pd.DataFrame`: Abstract method each agent implements
- `save_artifact(df, filename, subdir)` → `Path`: Saves CSV/XLSX/JSON to output directory
- `save_text_artifact(text, filename, subdir)` → `Path`: Saves text artifact

## Knowledge Graph

**File:** `agri_ai_agent/knowledge_graph/graph.py`

NetworkX `DiGraph` with 8 node types and 8 edge types:

| Node Types | Edge Types |
|------------|------------|
| Paper, Treatment, Crop, Observation, Yield, Soil, Climate, Management | DESCRIBES, TREATS, GROWS, OBSERVES, YIELDS, HAS_SOIL, HAS_CLIMATE, MANAGES |

Supports: centrality analysis (degree, betweenness), shortest path, PageRank, subgraph extraction, JSON serialization, and automatic graph construction from DataFrame via `build_from_dataframe()`.

## Configuration

**File:** `agri_ai_agent/config/settings.py`

`AgriAISettings` defines all directory paths, retry/timeout parameters, LLM settings (optional), export formats, checkpoint mode, and incremental mode. All paths are relative to the package root.

## Schema

### Universal Agricultural Machine Learning Schema (UAMS v2.0)

**File:** `agri_ai_agent/config/schema.py`

- **26 groups** (A–Z), covering Paper Metadata, Crop Information, Experimental Design, Environment, Soil Properties, Fertilizer Information, Crop Growth Parameters, Yield Parameters, Grain Quality, ML Targets, Engineered Features, Leakage Labels, Encoded Variables, ML Predictions, Soil Biological Properties, Soil Enzyme Activity, Soil Microbial Community, Remote Sensing Indices, Soil Exchangeable Cations, Water Extractable Organic, Soil Physical Properties, Nematode Ecology, Economic Parameters, Nutrient Uptake, Irrigation & Management, Soil pH Variants.
- Full column variant map (`VARIANT_MAP`) with ~700+ entries for normalising heterogeneous source column names
- `NON_FEATURE_COLS`, `POST_HARVEST_VARIABLES`, `PRE_HARVEST_MEASUREMENTS`, `NUMERIC_UAMS_COLUMNS` sets for downstream processing

## Data Flow

```
PDFs / Excel ──► Extraction ──► EvidenceFusion ──► Ontology ──►
TableIntelligence ──► SchemaPopulation ──► KnowledgeIntegration ──►
Knowledge ──► Validation ──► Feature ──► ModelSelection ──►
Training ──► Prediction ──► Recommendation ──► Fuzzy ──►
Benchmark ──► Explainability ──► ReadyReckoner
```

Each agent receives the cumulative `pd.DataFrame` from its predecessor, processes/appends columns, and passes it forward. The `AgentContract` carries metadata alongside the data.

## Key Design Decisions

1. **Sequential agent pipeline** — agents run in strict order; each sees the cumulative output of all prior agents
2. **Typed message contracts** — every agent communicates via `AgentContract`, enabling introspection, logging, and provenance
3. **Checkpoint/recovery** — per-agent Parquet checkpoints allow resumption after failure
4. **Configurable retry** — each agent retries up to `AGENT_RETRY_MAX` times with `AGENT_RETRY_DELAY_SEC` between attempts
5. **Critical vs non-critical failures** — extraction and training failures halt the pipeline; other agent failures are warned and skipped
6. **Incremental mode** — skips agents whose checkpoints already exist, enabling fast re-runs on large PDF corpora
