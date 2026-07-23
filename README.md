# Agricultural Intelligence Framework 

An end-to-end **Agentic Agricultural Intelligence Framework (AAIF)** that ingests research PDFs, extracts agronomic data into a Universal Agricultural Schema, trains ML models, applies fuzzy logic for fertilizer recommendations, and produces a **Ready Reckoner Table** — a per-crop yield decision support tool.

> **Current state:** 68 research papers across 9 crops → 146 engineered features → 39 ML models → 221 fuzzy rules → Ready Reckoner Table

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
| G. Crop Growth | 22 | Height, Biomass, SPAD, Leaf Area |
| H. Yield Parameters | 13 | Yield per Plot/Hectare, Fruit, Seeds |
| I. Grain Quality | 14 | Protein, Ash, Gluten |
| J–N. ML / Derived / Encoded | 56 | Targets, features, predictions |

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

### Best Model Per Target

| Target | Best Model | R² | RMSE | Samples |
|--------|-----------|-----|------|---------|
| **Plant_Height_cm** | Ridge | **0.995** | 0.14 | 10 |
| **Yield_per_Hectare** | XGBoost | **0.823** | 46.30 | 14 |
| **SPAD** | LinearRegression | 0.333 | 2.62 | 11 |
| **Shoot_Biomass_g** | LinearRegression | 0.150 | 1.35 | 18 |
| **Yield_per_Plot** | RandomForest | -0.129 | 467.23 | 8 |

> Note: R² values reflect current data availability. Plant_Height and Yield_per_Hectare show strong signal. Other targets require more training data for reliable predictions.

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
| Schema columns | 138 |
| Engineered features | 146 |
| ML models trained | 39 |
| Best model R² | 0.995 (Ridge, Plant_Height) |
| Fuzzy rules | 221 (v3.0) |
| Crops covered | 9 |
| Pipeline phases | 10 |
| Evaluation stages | 11 |
| Pipeline run time | ~25 seconds (incremental) |
| PDF skip rate | 84.6% (incremental mode) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Poppler (for pdftotext) — [install guide](#poppler-installation)

### Installation

```bash
git clone https://github.com/matrixflora/Ready-Reckoner-AI-Framework.git
cd Ready-Reckoner-AI-Framework
pip install -r requirements.txt
```

### Poppler Installation (Windows)

Download Poppler from: https://github.com/osber/poppler-windows/releases

Extract to `C:\poppler\poppler-24.08.0\Library\bin\` and add to PATH.

### Run Full Pipeline

```bash
python run_aaf_pipeline.py
```

This runs all 10 phases: ingestion → extraction → validation → feature engineering → ML training → fuzzy logic → recommendations → ready reckoner → continuous learning.

### Run Evaluation

```bash
python run_eval.py
```

### Run Individual Components

```bash
# Post-pipeline analysis
python post_pipeline_analysis.py

# Export all results
python export_all_results.py

# Pdfplumber extraction
python pdfplumber_extraction.py
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
├── LICENSE                         # MIT License
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

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
