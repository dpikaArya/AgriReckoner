# Agricultural Intelligence Framework 

Scientific Objectives & Main Implementation Features
1. Scientific Objectives
The Ready Reckoner Table AI Framework (also termed Agentic Agricultural Intelligence Framework, AAIF) addresses a fundamental challenge in precision agriculture: converting unstructured, heterogeneous research literature into actionable, crop-specific decision-support tools.
The core scientific objectives are:
1.	Automated Knowledge Extraction from Literature. Agronomic research is dispersed across thousands of PDF publications with inconsistent formatting, terminology, and tabular structures. The framework automates the ingestion, parsing, and structured extraction of key agronomic variables (soil properties, fertilizer treatments, growth parameters, and yield outcomes) from these documents, eliminating manual data curation.
2.	Universal Schema Harmonisation. To enable cross-study and cross-crop comparisons, extracted data is mapped to the Universal Agricultural Machine Learning Schema (UAMS) v1.0 — a 138-column, 14-group ontology covering paper metadata, crop information, experimental design, environment, soil chemistry, fertilization, growth physiology, yield, and grain quality. This standardisation is a prerequisite for any downstream meta-analysis or machine learning.
3.	Yield Prediction via Multi-Model Machine Learning. Once harmonised, the dataset supports supervised learning for predicting key agronomic targets — yield per hectare, plant height, SPAD (chlorophyll index), and shoot biomass — using an ensemble of 8 regression model families (Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, SVR) with cross-validation and hyperparameter tuning. The goal is to identify the best predictive model per target under data-scarce conditions.
4.	Fuzzy-Logic Fertilizer Recommendation. A Mamdani fuzzy inference system with 221 rules operates on 10 agronomic input variables (N, P, K, Zn, soil pH, rainfall, temperature, organic carbon, growth stage, yield prediction) to produce crop-specific, linguistically interpretable fertilizer recommendations (N/P/K dosages) with confidence scores. This bridges the gap between data-driven prediction and expert-system reasoning.
5.	Ready Reckoner Table Generation. The terminal output is a per-crop Ready Reckoner Table — a compact decision-support artefact summarising optimal fertilizer regimes, expected yields, treatment alternatives, and confidence levels, exportable in Excel, CSV, and HTML formats for extension-agent and farmer use.
2. Main Implementation Features
The framework is implemented as a 10-phase agentic pipeline orchestrated by a central Orchestrator class. It sequences **17 agents wired into the default pipeline** (`orchestrator.PIPELINE_STEPS`), each governed by a typed AgentContract message protocol. One further agent — the **ProvenanceAgent** — is available in the codebase but is **not** part of the default pipeline. An **optional LLM extraction agent** (OpenAI-backed, see [Optional LLM extraction](#optional-llm-extraction)) can be enabled separately. Key implementation features include:
Feature	Description
Incremental PDF Ingestion	Scans a PDF directory; detects crop, DOI, title, and duplicates via fuzzy string matching (SequenceMatcher >0.90). Skips previously registered papers via a SQLite paper_registry (current skip rate: 84.6%).
6-Reader Hybrid Extraction	Dispatches each PDF through Pdfminer, Camelot, Pdfplumber, Poppler (pdftotext), OCR, and Semantic readers with configurable timeouts. Regex patterns extract soil pH, N/P/K, yield, temperature, rainfall, and 15+ agronomic variables.
Confidence-Weighted Evidence Fusion	The EvidenceFusionAgent deduplicates and merges multi-reader outputs using per-reader confidence weights, resolving conflicts at the cell level.
Ontology Mapping & Table Intelligence	The OntologyAgent normalises heterogeneous column names to UAMS via a 120-entry master column map. The TableIntelligenceAgent classifies table types and computes summary statistics.
Biological Range Validation	Enforces domain constraints (e.g., soil pH ∈ [3,10], yield ∈ [0, 50,000] kg/ha, Tmax ≥ Tmin). Flags IQR-based outliers (3× IQR), OCR artefacts, and unit inconsistencies.
Feature Engineering (146 features)	Derives 16+ composite features: NPK Index, Soil Fertility Index, Climate Index, Growing Degree Days, Nitrogen/Water Use Efficiency, Growth-Yield Index, polynomial temperature terms, and factor-encoded categorical variables.
Adaptive Model Selection	The ModelSelectionAgent selects the model pool based on sample size, applies SelectKBest feature selection when features exceed samples, and runs GridSearchCV with target-specific parameter grids.
Mamdani Fuzzy Expert System	221 rules in YAML (v3.0) with 10 trapezoidal/triangular input MFs and 3 output MFs (Low/Medium/High for N, P, K). Centroid defuzzification produces continuous recommendation values, with crop-specific adjustment factors for all 9 crops.
Explainability & Provenance	The ExplainabilityAgent generates per-prediction feature-attribution explanations. A ProvenanceAgent that tracks per-cell data lineage from source PDF to final schema exists in the codebase but is **not wired into the default pipeline** — it is available for callers that instantiate it directly.
Continuous Learning	Drift detection monitors model performance over time. New papers trigger incremental retraining; the paper registry and model versions are updated without full pipeline re-execution.
11-Stage Evaluation Suite	Automated scoring across ingestion completeness, extraction accuracy, ontology coverage, schema compliance, unit consistency, data quality, feature utility, data-leakage checks, statistical soundness, model readiness, and documentation.
Knowledge Graph	A NetworkX graph with 8 node types (Paper, Crop, Soil, Treatment, Yield, Feature, Model, Rule) encodes entity relationships for graph-based queries and downstream reasoning.

3. Current Scale & Results
Metric	Value
Package version	2.0.0
License	Apache-2.0
UAMS schema columns	138 (14 groups, A–N)
Crops covered	9 (Barley, Bell Pepper, Black Wheat, Cabbage, Carrot, Chickpea, Cotton, Maize, Spinach)
Fuzzy rules	221 (Mamdani v3.0)
Agents wired into default pipeline	17 (+ ProvenanceAgent available but not wired, + optional LLM extraction agent)
Pipeline phases	10
Model R²	Previously reported values (e.g. 0.995) were target-leakage artefacts, not validated skill — see [ML Models](#ml-models)


---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Phases](#pipeline-phases)
- [Schema & Data](#schema--data)
- [ML Models](#ml-models)
- [Fuzzy Logic](#fuzzy-logic)
- [Results Summary](#results-summary)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This framework automates the conversion of unstructured agricultural research PDFs into structured, ML-ready datasets. It supports **precision agriculture decision support** — specifically, generating Ready Reckoner Tables that summarize optimal fertilizer rates, expected yields, and crop-specific recommendations.

### Key Capabilities

| Capability | Details |
|---|---|
| **PDF Ingestion** | Auto-detects crop, DOI, year from 91+ research papers |
| **Incremental Processing** | Skips already-extracted papers (84.6% skip rate) |
| **Hybrid Extraction** | 6 readers: Pdfminer, Camelot, Pdfplumber, Poppler, OCR, Semantic |
| **Evidence Fusion** | Confidence-weighted multi-reader deduplication |
| **Schema Mapping** | Maps to UAMS v1.0 (138 agricultural variables) |
| **Data Validation** | Biological range checks, outlier detection, crop verification |
| **Feature Engineering** | 146 features: interactions, ratios, polynomials, derived indices |
| **ML Training** | 8 model types × 5 targets = 39 trained models |
| **Fuzzy Logic** | 221 Mamdani rules for N/P/K fertilizer recommendations |
| **Ready Reckoner** | Per-crop summary with yields, treatments, recommendations |
| **Evaluation** | 11-stage evaluation suite with automated scoring |

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    PHASE 0: LOAD MASTER DATASETS                │
  │              5 crop-specific Excel workbooks (45 rows)          │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                 PHASE 1: INCREMENTAL INGESTION                  │
  │           91 PDFs scanned → 105 papers registered               │
  │           SQLite registry tracks processed papers               │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │           PHASE 2-3: AI EXTRACTION + SCHEMA MAPPING            │
  │     6 hybrid readers → Evidence fusion → UAMS (138 columns)     │
  │     68 papers mapped to Universal Agricultural Schema           │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                   PHASE 4: DATA VALIDATION                     │
  │        Biological range checks, outlier detection,              │
  │        crop verification, DOI validation, temperature           │
  │        consistency checks                                       │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                 PHASE 5: FEATURE ENGINEERING                    │
  │     16 derived features → 146 total columns                     │
  │     NPK index, Soil fertility, Climate index,                   │
  │     Growth-Yield interactions                                   │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                   PHASE 6: ML TRAINING                         │
  │     5 targets × 8 models = 39 trained models                   │
  │     Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, SVR, KNN       │
  │     Cross-validation, feature selection, hyperparameter tuning  │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                  PHASE 7: FUZZY LOGIC                          │
  │     221 Mamdani rules (v3.0)                                    │
  │     10 input variables, centroid defuzzification                │
  │     Crop-specific N/P/K recommendations                        │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │               PHASE 8: RECOMMENDATION AGENT                    │
  │     Per-crop fertilizer recommendations                        │
  │     Confidence scoring, treatment alternatives                  │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │                PHASE 9: READY RECKONER                         │
  │     Excel, HTML export generation                               │
  │     Per-crop yield summaries                                    │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │              PHASE 10: CONTINUOUS LEARNING                      │
  │     Paper registry update, drift detection                      │
  │     Model versioning, incremental retraining                    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Phases

| Phase | Name | Description | Key Output |
|-------|------|-------------|------------|
| 0 | **Master Datasets** | Load crop-specific Excel workbooks | 45 rows × 81 cols |
| 1 | **Incremental Ingestion** | Scan PDFs, skip registered papers | 91 PDFs → 0 new (incremental) |
| 2-3 | **AI Extraction** | 6 hybrid readers + schema mapping | 68 rows × 138 cols |
| 4 | **Validation** | Biological ranges, outlier detection | Cleaned schema |
| 5 | **Feature Engineering** | 16 derived features, interaction terms | 68 rows × 146 cols |
| 6 | **ML Training** | 5 targets, 8 model types, CV | 39 trained models |
| 7 | **Fuzzy Logic** | 221 Mamdani rules, 10 input vars | Fertilizer rules |
| 8 | **Recommendations** | Per-crop N/P/K recommendations | 9 crop recs |
| 9 | **Ready Reckoner** | Excel + HTML generation | Decision support tables |
| 10 | **Continuous Learning** | Registry update, drift detection | Updated paper DB |

---

## Schema & Data

### Universal Agricultural Schema (UAMS v1.0)

- **138 columns** across 14 groups (A–N)
- **68 research papers** mapped to schema
- **9 crops**: Barley, Bell Pepper, Black Wheat, Cabbage, Carrot, Chickpea, Cotton, Maize, Spinach

| Schema Group | Columns | Description |
|-------------|---------|-------------|
| A. Paper Metadata | 6 | Paper_ID, DOI, Journal, Year, Authors, Country |
| B. Crop Information | 6 | Crop, Variety, Season, Growth Duration, Stage |
| C. Experimental Design | 9 | Design, Replications, Plot Size, Spacing |
| D. Environment | 8 | Lat, Lon, Alt, Temp, Rainfall, Humidity |
| E. Soil Properties | 16 | pH, EC, N, P, K, micronutrients |
| F. Fertilizer Information | 7 | Treatment, Fertilizer, Dose |
| G. Crop Growth Parameters | 22 | Height, Biomass, SPAD, Leaf Area |
| H. Yield Parameters | 13 | Yield per Plot/Hectare, Fruit, Seeds |
| I. Grain Quality | 14 | Protein, Ash, Gluten |
| J. ML Target Variables | 5 | Prediction targets |
| K. Engineered Features | 16 | Derived domain features |
| L. Leakage Labels | 1 | Prediction-time availability |
| M. Encoded Variables | 6 | Numeric categorical codes |
| N. ML Predictions | 9 | Model outputs and recommendations |
| **Total** | **138** | 14 groups (A–N) |

### Data Coverage

| Field | Rows with Data | Status |
|-------|---------------|--------|
| Yield_per_Hectare | 14 | Primary target |
| Soil_pH | 22 | Cleaned (outliers removed) |
| Rainfall | 40 | Well-populated |
| Plant_Height_cm | 2 | Sparse |
| SPAD | 1 | Sparse |
| Phosphorus | 5 | Partial |
| Potassium | 7 | Partial |
| Nitrogen | 0 | Missing (requires future extraction) |

---

## ML Models

### Training Setup

- **5 targets**: Yield_per_Plot, Yield_per_Hectare, Plant_Height_cm, SPAD, Shoot_Biomass_g
- **8 model types**: LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost, SVR
- **Cross-validation**: 5-fold where sample size permits

### On the previously reported R² scores (important)

Earlier versions of this README published a "Best Model Per Target" table with headline
scores such as **R² = 0.995** (Ridge, Plant Height) and **R² = 1.0** (Linear Regression).
**Those numbers were target-leakage artefacts, not evidence of predictive skill.** Post-harvest
outcomes, target columns, and model-prediction columns were reachable by the feature matrix,
so models were effectively reading the answer. They were also computed from a single hold-out on
only a handful of test points. They must **not** be cited as validated performance.

The refactor addresses this at two levels:

- **Leakage control** — `agri_ai_agent/ml/leakage.py` removes columns unavailable at prediction
  time (UAMS post-harvest, target group J, and prediction group N), including engineered features
  derived from those outcome families, before any model sees the data.
- **Honest small-n evaluation** — `agri_ai_agent/ml/evaluation.py` cross-validates each model
  (leave-one-out below n=30, repeated 5-fold above), fitting imputation inside each fold so no test
  information leaks into training. It **refuses to report any skill metric below n=8** ("insufficient
  data") and **flags every metric below n=30 as advisory** (cross-validated but non-robust).

Given the current data volume (single- and low-double-digit sample counts per target), any metric
this pipeline emits today is **advisory**. Reliable per-target performance requires substantially
more extracted training data. No validated R² table is published here until that data exists.

---

## Fuzzy Logic

### Rule System (v3.0)

- **221 rules** for fertilizer recommendations
- **10 input variables**: Nitrogen, Phosphorus, Potassium, Zinc, Soil_pH, Rainfall, Temperature_Max, Organic_Carbon, Growth_Stage, Yield_Prediction
- **Mamdani inference** with centroid defuzzification
- **Crop-specific adjustments** for all 9 crops

### Recommendation Output

| Crop | Recommendation | Confidence |
|------|---------------|------------|
| Barley | Balanced NPK (15-15-15) | High |
| Bell Pepper | Balanced NPK (15-15-15) | High |
| Black Wheat | Balanced NPK (15-15-15) | High |
| Cabbage | Dolomite + Bio NPK | High |
| Carrot | Balanced NPK (15-15-15) | High |
| Chickpea | Balanced NPK (15-15-15) | High |
| Cotton | Balanced NPK (15-15-15) | High |
| Maize | Balanced NPK (15-15-15) | High |
| Spinach | DAP (Phosphorus-rich) | Medium |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Research papers processed | 68 (from 91 PDFs) |
| Master dataset rows | 45 (5 crops) |
| Total training rows | 113 (merged) |
| Schema columns | 138 (14 groups, A–N) |
| Engineered features | 146 |
| Model R² | Not published as validated — earlier scores (e.g. 0.995) were target-leakage artefacts; metrics are advisory at current n (see [ML Models](#ml-models)) |
| Fuzzy rules | 221 (v3.0) |
| Crops covered | 9 |
| Agents wired into default pipeline | 17 (+ ProvenanceAgent available but not wired, + optional LLM extraction agent) |
| Pipeline phases | 10 |
| Evaluation stages | 11 |
| Pipeline run time | ~25 seconds (incremental) |
| PDF skip rate | 84.6% (incremental mode) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Poppler (for `pdftotext`) — [install guide](#poppler-installation)
- **macOS only:** `brew install libomp` (required by `xgboost`; import fails without it)
- Source research PDFs live in a **gitignored `Data ADES/`** directory at the repo root.
  This directory is not shipped with the repository; create it and add your own PDFs.

### Installation

```bash
git clone https://github.com/matrixflora/Ready-Reckoner-AI-Framework.git
cd Ready-Reckoner-AI-Framework
pip install -e .          # installs the agri-ai-agent package (v2.0.0)
```

### Poppler Installation (Windows)

Download Poppler from: https://github.com/osber/poppler-windows/releases

Extract to `C:\poppler\poppler-24.08.0\Library\bin\` and add to PATH.

### Run the pipeline (primary entry point)

The primary engine is the `agriai` CLI (installed by `pip install -e .`):

```bash
agriai run --file data.csv        # run on a prepared feature/data table
agriai run --papers "Data ADES/"  # ingest PDFs, then run the full pipeline
```

Equivalently, without the console script:

```bash
python -m agri_ai_agent run --file data.csv
```

This runs the 10-phase pipeline: ingestion → extraction → validation → feature
engineering → ML training → fuzzy logic → recommendations → ready reckoner →
continuous learning.

### Legacy runner (frozen)

`run_aaf_pipeline.py` is the **legacy, validated monolithic PDF runner**, frozen at git tag
**`legacy-monolith-v1`**. It remains the reference for PDF-based runs while the agent path is
being validated on real PDFs, and will be **retired once that validation is complete**. Prefer
the `agriai` entry point above for new work.

```bash
python run_aaf_pipeline.py   # legacy path (git tag legacy-monolith-v1)
```

### Run Evaluation

```bash
python run_eval.py
```

### Optional LLM extraction

An optional OpenAI-backed extraction agent (`LLMExtractionAgent`) can extract UAMS values
from free text with source-grounding and per-cell provenance. It is **not** part of the
default pipeline and requires the `llm` extra plus an API key:

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=...            # your OpenAI key
python scripts/smoke_llm_extract.py  # live smoke test (makes a real API call)
```

---

## Project Structure

```
Ready-Reckoner-AI-Framework/
├── agri_ai_agent/                  # Core agent framework
│   ├── agents/                     # Pipeline agents
│   │   ├── knowledge_agent.py      # PDF ingestion + registry
│   │   ├── extraction_agent.py     # 6 hybrid extraction readers
│   │   ├── evidence_fusion_agent.py # Confidence-weighted merge
│   │   ├── ontology_agent.py       # Column name resolution
│   │   ├── table_intelligence_agent.py  # Table type classification
│   │   ├── schema_population_agent.py   # Derived column computation
│   │   ├── validation_agent.py     # Biological + agronomic rules
│   │   ├── feature_agent.py        # Feature engineering
│   │   ├── model_selection_agent.py # Adaptive model pool
│   │   ├── training_agent.py       # Model training
│   │   ├── fuzzy_logic_agent.py    # 221 Mamdani rules
│   │   ├── prediction_agent.py     # Yield prediction
│   │   ├── recommendation_agent.py # Top-3 alternatives
│   │   ├── benchmark_agent.py      # Pipeline metrics
│   │   ├── explainability_agent.py # Per-prediction explanations
│   │   ├── ready_reckoner_agent.py # JSON/CSV/MD/HTML exports
│   │   ├── continuous_learning_agent.py  # Drift detection
│   │   └── provenance_agent.py     # Per-cell provenance
│   ├── config/
│   │   ├── schema.py               # UAMS_COLUMNS (138 vars)
│   │   └── settings.py             # AgriAISettings
│   ├── contracts/
│   │   └── messages.py             # AgentContract, OrchestratorState
│   ├── knowledge_graph/
│   │   └── graph.py                # NetworkX graph (8 node types)
│   ├── rules/
│   │   ├── fertilizer_rules.yaml   # 221 Mamdani rules (v3.0)
│   │   ├── membership_functions.py # Input membership functions
│   │   └── output_memberships.py   # Output membership functions
│   ├── orchestrator.py             # 10-phase pipeline orchestrator
│   └── utils/
│       ├── io_utils.py             # File I/O helpers
│       └── logging_utils.py        # Structured logging
│
├── tests/                          # Test suites
├── benchmarks/                     # Gold standard benchmarks
├── evaluation/                     # 11-stage evaluation suite
│   ├── stage01_ingestion.py
│   ├── stage02_extraction.py
│   ├── stage03_ontology.py
│   ├── stage04_schema.py
│   ├── stage05_unit.py
│   ├── stage06_quality.py
│   ├── stage07_feature.py
│   ├── stage08_leakage.py
│   ├── stage09_statistics.py
│   ├── stage10_model_readiness.py
│   ├── stage11_documentation.py
│   └── reports/                    # Generated evaluation reports
│
├── config/                         # Agent configuration
│   ├── benchmark.yaml
│   └── explainability.yaml
│
├── fuzzy_logic/                    # Fuzzy logic outputs
│   ├── fertilizer_rules.yaml
│   └── fuzzy_report.md
│
├── models/                         # Trained ML models (.pkl)
├── outputs/                        # Generated results
│   ├── Universal_Agricultural_Schema.csv   # Master schema (68×138)
│   ├── features_dataset.csv                # Engineered features (68×146)
│   ├── model_metrics.xlsx                  # ML performance metrics
│   ├── Ready_Reckoner.xlsx                 # Final decision support
│   ├── AAIF_Final_Report.html              # Summary report
│   ├── ingestion_report.csv                # PDF processing log
│   ├── recommendations/                    # Fertilizer recs
│   └── Validated_Extractions.json          # Structured extractions
│
├── Data ADES/                      # Source research PDFs (gitignored)
├── database/
│   └── paper_registry.sqlite       # Paper processing registry
│
├── run_aaf_pipeline.py             # Main pipeline entry point
├── run_eval.py                     # Evaluation runner
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
├── LICENSE                         # Apache-2.0 License
└── README.md                       # This file
```

---

## Output Files

### Primary Outputs

| File | Description |
|------|-------------|
| `outputs/Universal_Agricultural_Schema.csv` | Master schema (68 rows × 138 cols) |
| `outputs/features_dataset.csv` | Engineered features (68 rows × 146 cols) |
| `outputs/model_metrics.xlsx` | ML performance (39 models × 10 metrics) |
| `outputs/Ready_Reckoner.xlsx` | Final decision support table |
| `outputs/AAIF_Final_Report.html` | Pipeline summary report |
| `outputs/Validated_Extractions.json` | Structured extraction data |
| `outputs/recommendations/` | Per-crop fertilizer recommendations |

### Model Files

| File | Description |
|------|-------------|
| `models/best_model_Yield_per_Hectare.pkl` | Best yield prediction model |
| `models/best_model_Plant_Height_cm.pkl` | Best growth prediction model |
| `models/best_model_SPAD.pkl` | Best chlorophyll prediction model |
| `models/best_model_Shoot_Biomass_g.pkl` | Best biomass prediction model |
| `models/best_model_Yield_per_Plot.pkl` | Best per-plot yield model |

### Evaluation Reports

| Report | Description |
|--------|-------------|
| `evaluation/reports/Executive_Summary.md` | High-level findings |
| `evaluation/reports/Pipeline_Performance.md` | Phase-by-phase metrics |
| `evaluation/reports/Repository_Scorecard.md` | Code quality assessment |
| `evaluation/reports/Dataset_Quality_Score.md` | Data quality scoring |
| `evaluation/reports/Evaluation_Dashboard.html` | Interactive HTML dashboard |

---

## Dependencies

```
# Core
pandas>=2.0
numpy>=1.24
scipy>=1.10

# PDF Processing
pdfminer.six
pdfplumber
PyPDF2

# Machine Learning
scikit-learn>=1.3
xgboost>=2.0

# Configuration
PyYAML>=6.0
openpyxl>=3.1

# Data
pyarrow>=10.0
chardet>=5.0
duckdb>=0.8.0

# Testing
pytest>=7.0
pytest-cov>=4.0
```

See `requirements.txt` for exact versions.

---

## License

This project is licensed under the Apache 2 License — see [LICENSE](LICENSE) for details.
