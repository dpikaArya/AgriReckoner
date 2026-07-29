# Data Pipeline — Agricultural Intelligence Framework

## End-to-End Flow

```
External Data Sources
  (CGIAR, FAOSTAT, NASA POWER, SoilGrids, ISRIC,
   Zenodo, Mendeley, Kaggle, ICAR, SAUs)
        │
        ▼  ExternalDataConnector: connect() → discover() → download()
        │  → validate() → register()
        ▼
  DatasetPackage objects (standardised format)
        │
        ▼  DatasetNormalizationAgent
  Normalized DataFrames
        │
        ▼  DatasetIngestionBridgeAgent
  ID Hierarchy Assignment (Dataset_ID → ... → Observation_ID)
        │
        ▼  ExtractionAgent (6 readers)
  Raw extracted variables from PDFs
        │
        ▼  EvidenceFusionAgent
  Confidence-weighted deduplicated extractions
        │
        ▼  OntologyAgent
  UAMS-normalised column names (VARIANT_MAP, 700+ entries)
        │
        ▼  TableIntelligenceAgent / SchemaPopulationAgent
  Derived columns, computed values, table classification
        │
        ▼  KnowledgeIntegrationAgent / KnowledgeAgent
  Domain-knowledge imputation, constraint enforcement
        │
        ▼  ObservationGenerationAgent
  Observation hierarchy (ID generation, treatment dedup)
        │
        ▼  ValidationAgent
  Range checks · Unit harmonisation · Quality gates
        │
        ▼  FeatureStoreAgent (Validation Gate)
  Parquet persistence · Rejected observation filtering
        │
        ▼  FeatureAgent
  146 engineered features: NPK Index, GDD, NUE, WUE, etc.
        │
        ▼  ModelSelectionAgent / TrainingAgent
  ML-ready feature matrix (leakage-free)
```

## DatasetPackage Format

The `DatasetPackage` dataclass (`agri_ai_agent/external_data/dataset_package.py`) is the standardised container returned by every external data connector:

| Field | Type | Description |
|-------|------|-------------|
| `dataset_id` | `str` | SHA-256 hash of `provider/resource_id/version` (16 chars) |
| `source` | `str` | Connector source name (e.g., "CGIAR") |
| `provider` | `str` | Data provider (aliased from source) |
| `resource_id` | `str` | Unique ID within the source |
| `name` | `str` | Human-readable name |
| `version` | `str` | Version string (default "1.0.0") |
| `document_type` | `str` | `tabular`, `pdf`, `json`, etc. |
| `metadata` | `dict` | Source-specific metadata |
| `documents` | `list[dict]` | Extracted document records |
| `tables` | `list[pd.DataFrame]` | Tabular data content |
| `data` | `pd.DataFrame | None` | Primary DataFrame |
| `download_path` | `Path | None` | Local filesystem path |
| `checksum` | `str` | SHA-256 hash (16 chars) of metadata |
| `is_valid` | `bool` | Validation result |
| `row_count` | `int` | Number of rows |
| `column_count` | `int` | Number of columns |
| `validation_errors` | `list[str]` | Validation failure details |
| `supplementary_files` | `list[Path]` | Sidecar files |
| `license` | `str` | Dataset license |

## Unique ID Hierarchy

The framework assigns a 6-level hierarchical ID to every observation:

```
Dataset_ID ────► Document_ID ────► Paper_ID ────► Experiment_ID ────► Treatment_ID ────► Observation_ID
```

### ID Assignment Rules

| Level | Pattern | Assignment |
|-------|---------|------------|
| `Dataset_ID` | `{provider}/{resource_id}` | From DatasetPackage, or generated from Paper_ID |
| `Document_ID` | `{Dataset_ID}/doc/{n}` | Auto-generated per dataset package |
| `Paper_ID` | `{Dataset_ID}/paper` | Auto-generated; falls back to `PAPER/{uuid}` |
| `Experiment_ID` | `{Paper_ID}/exp/{Location_Site_Season_Year_Design_Crop}` | Grouped by geography, season, design, crop |
| `Treatment_ID` | `{Experiment_ID}/trt/{Treatment_Fertilizer_Dose_Variety}` | Unique treatment configuration per experiment |
| `Observation_ID` | `OBS/{dataset_id}/{uuid}` | Globally unique per row |

**Deduplication:** Rows sharing the same `(Paper_ID, Experiment_ID, Treatment_ID)` are fused: numeric columns are averaged, categorical columns take the first non-null value, and a `_replicates` counter tracks the original count.

### ID Generation Code Path

`DatasetIngestionBridgeAgent` assigns IDs for external packages; `ObservationGenerationAgent` normalises and deduplicates for the full pipeline. Both ensure every row gets a complete ID chain before the validation gates.

## Validation Gates

### Gate 1: Range Constraints

Hard bounds enforced on numeric columns (`ValidationAgent.RANGE_CONSTRAINTS`):

| Column | Min | Max | Unit |
|--------|-----|-----|------|
| Soil_pH | 0 | 14 | pH |
| EC | 0 | 100 | dS/m |
| Humidity | 0 | 100 | % |
| Temperature_Max | -20 | 60 | °C |
| Temperature_Min | -30 | 50 | °C |
| Rainfall | 0 | 10000 | mm |
| SPAD | 0 | 100 | SPAD units |
| Yield_per_Hectare | 0 | 100000 | kg/ha |
| Nitrogen | 0 | 1000 | kg/ha |
| Organic_Carbon | 0 | 100 | % |

### Gate 2: Biological Rules

Crop-aware range checks with domain-appropriate bounds:

| Variable | Global Range | Crop Example | Unit |
|----------|-------------|--------------|------|
| Plant_Height_cm | [1, 800] | Maize: [100, 400] | cm |
| Yield_per_Hectare | [100, 20000] | Wheat: [1000, 8000] | kg/ha |
| Soil_pH | [3.0, 10.0] | — | pH |
| Harvest_Index | [0.1, 0.7] | — | ratio |
| Leaf_Number | [2, 100] | — | count |

### Gate 3: Agronomic Rules

Cross-field consistency checks:

- **Yield/Biomass ratio** ∈ [0.1, 0.8]
- **Tmax ≥ Tmin** (error if violated)
- **Height/Leaf area** plausibility
- **OM/OC ratio** ∈ [1.0, 2.5]
- **N rate** ≤ 300 kg/ha
- **Harvest index** ∈ [0.15, 0.65]

### Gate 4: Statistical Outliers

- **IQR method**: 1.5× IQR beyond Q1/Q3
- **MAD method**: Modified Z-score > 3.5
- Both reported as warnings, not errors

### Gate 5: Unit Harmonisation

Automatic detection and conversion:

| Detection Trigger | Conversion |
|------------------|------------|
| Yield_per_Acre exists, Yield_per_Hectare missing | × 2.47105 |
| Temperature > 100 °F | (°F - 32) × 5/9 |
| Column name contains "inch" | × 2.54 |
| N/P/K values < 1 kg/ha | × 1000 (g/ha) |
| Plot yield + plot size | 10000 / plot_size_m2 |

### Feature Store Gate

The `FeatureStoreAgent` filters rows with `_validation_status == "ACCEPTED"` (or `_is_valid == True`) before passing data to feature engineering. Rejected observations are persisted to Parquet but excluded from training.

## Feature Engineering

The `FeatureAgent` produces 146 total columns from the validated schema:

| Feature Category | Examples |
|-----------------|----------|
| Composite indices | NPK Index, Soil Fertility Index, Climate Index |
| Efficiency metrics | Nitrogen Use Efficiency (NUE), Water Use Efficiency (WUE) |
| Thermal time | Growing Degree Days (GDD), Heat Units |
| Interaction terms | Temp × Rainfall, N × P |
| Polynomial expansions | Temp_squared, Temp_cubed |
| Moving averages | Rainfall_7d_MA, Temp_7d_MA |
| Yield-normalised | Yield_per_Plant, Harvest_Index_Calc |
| Risk indices | Stress_Index, Disease_Risk_Index, Rainfall_Anomaly |

## Data Quality Checks

The `ValidationAgent._quality_checks()` method runs:

| Check | Details |
|-------|---------|
| Duplicate rows | `df.duplicated().sum()` |
| Duplicate columns | `df.columns[df.columns.duplicated()]` |
| Missing identifiers | Paper_ID, DOI, Experiment_ID, Plot_ID, Sample_ID |
| Negative values | Count of negative entries in numeric columns |
| Range violations | Per-column hard bound checks |
| Biological violations | Crop-specific range checks |
| Agronomic violations | Cross-field consistency rules |
| IQR outliers | 1.5× IQR beyond quartiles |
| MAD outliers | Modified Z-score > 3.5 |
| Unit inconsistencies | Yield_per_Hectare/Yield_per_Acre ratio off |
| Ontology consistency | Tmax < Tmin, Avg temp ≠ (Tmax+Tmin)/2 |

## Provenance and Lineage Tracking

### Per-Cell Provenance

The `ValidationAgent._build_provenance()` creates provenance records for every extracted variable:

```json
{
  "paper": "paper_name.pdf",
  "doi": "10.1000/xyz123",
  "variable": "Soil_pH",
  "value": "6.5",
  "unit": "pH",
  "page": 5,
  "table": "Table 2",
  "row": "3",
  "column": "pH",
  "source_text": "soil pH was 6.5",
  "char_start": 1234,
  "char_end": 1248,
  "confidence": "high",
  "validation": "ACCEPTED"
}
```

### Lineage Tracker

`src/provenance/lineage_tracker.py` provides `LineageTracker` for end-to-end lineage from source PDF to final schema. It tracks which source files contributed to each output value, including transformation history and intermediate datasets.

### Provenance Agent

`ProvenanceAgent` (`agri_ai_agent/agents/provenance_agent.py`) is available but not wired into the default pipeline. It produces a complete per-cell provenance graph linking every UAMS column value back to its source PDF page, reader, and extraction method.

### Pipeline Provenance

The `Orchestrator._write_provenance()` method writes `Pipeline_Provenance.json` containing:
- Pipeline ID, version, status, timestamps
- Completed and failed agent lists
- Per-agent: status, execution_time_sec, retry_count, errors, artifacts

## Dataset Versioning

### Immutable Storage

`ImmutableStorage` (`src/data_sources/storage.py`) stores data in a content-addressed, versioned directory structure:

```
external_data/raw/
├── CGIAR/
│   └── resource_123/
│       ├── v1.0/
│       │   ├── a1b2c3d4e5f6g7h8.csv
│       │   └── a1b2c3d4e5f6g7h8.metadata.json
│       └── v1.1/
│           └── ...
├── FAOSTAT/
│   └── ...
```

Each file is stored by its SHA-256 checksum (16 chars), enabling:
- **Deduplication**: identical content stored once
- **Integrity verification**: checksum = filename
- **Versioning**: multiple versions coexist under `v{version}/`
- **Provenance**: metadata sidecar stores download timestamp, source URL, size

### Dataset Registry

`DatasetRegistry` (`agri_ai_agent/external_data/registry_db.py`) uses SQLite to track:
- Registered datasets and their versions
- Checksum deduplication
- Update checks for incremental sync
- Skip-if-unchanged logic

### Paper Registry

`paper_registry.sqlite` tracks processed PDFs using fuzzy string matching (`SequenceMatcher > 0.90`) to detect duplicate papers by title, DOI, and crop. Current skip rate: 84.6% in incremental mode.
