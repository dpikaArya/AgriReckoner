# Ready Reckoner Table — AI Framework

An end-to-end **Agentic Agricultural Intelligence Framework (AAIF)** that ingests research PDFs, extracts treatment-level agronomic data using 6 hybrid extraction engines, trains 12 machine learning models, applies fuzzy logic for fertilizer recommendations, and produces a **Ready Reckoner Table** — a concise per-crop yield decision support tool.

> **Live results:** 24 research papers → 187 treatment rows → 12 ML models → Ready Reckoner Table

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Agents (v2.0 — 17-Step Pipeline)](#pipeline-agents-v20--17-step-pipeline)
- [Extraction Engine](#extraction-engine)
- [ML Models](#ml-models)
- [Results Summary](#results-summary)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Test Suites](#test-suites)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This framework automates the conversion of unstructured agricultural research PDFs into structured, machine-learning-ready datasets. It was designed for **precision agriculture decision support** — specifically, generating Ready Reckoner Tables that summarize optimal fertilizer rates, expected yields, and crop-specific recommendations.

### Key Capabilities

| Capability | Details |
|---|---|
| **PDF Ingestion** | Auto-detects crop, DOI, year from 24+ research papers |
| **Hybrid Extraction** | 6 readers: Pdfminer, Camelot, Pdfplumber, Poppler, OCR, Semantic |
| **Evidence Fusion** | Confidence-weighted multi-reader deduplication |
| **Schema Mapping** | Maps to UAMS v1.0 (138 agricultural variables) |
| **Knowledge Graph** | NetworkX graph with 8 node types, 8 edge types, PageRank |
| **Provenance Tracking** | Per-cell DOI, extraction method, page, table provenance |
| **ML Training** | 12 models: Linear, Ridge, Lasso, ElasticNet, RF, XGB, etc. |
| **Fuzzy Logic** | 221 Mamdani rules for fertilizer recommendations |
| **Ready Reckoner** | Per-paper summary table with yields, treatments, recommendations |
| **Benchmarking** | Automated pipeline metrics with historical comparison |
| **Explainability** | Per-prediction JSON explanations with feature importance |
| **Continuous Learning** | Drift detection, retraining, model versioning |
| **Evaluation Dashboard** | 8-metric HTML dashboard with performance tracking |

---

## Architecture

```
                         ┌──────────────────────────┐
                         │    PDF Research Papers     │
                         │      (24 papers)           │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 1: Knowledge Agent           │
                    │  (DOI, Crop, Year detection)       │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 2: Knowledge Integration     │
                    │  (Crop/Soil/Climate reference DB)  │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 3: Extraction Agent          │
                    │  (6 hybrid readers)                │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 4: Evidence Fusion           │
                    │  (Confidence-weighted merging)     │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 5: Ontology Agent            │
                    │  (Column name resolution)          │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 6: Table Intelligence        │
                    │  (Table type classification)       │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 7: Schema Population         │
                    │  (Derived column computation)      │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 8: Validation Agent v2       │
                    │  (Biological + agronomic rules)    │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 9: Feature Agent v2          │
                    │  (150+ engineered features)        │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 10: Model Selection Agent    │
                    │  (Adaptive pool + GridSearchCV)    │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 11: Training Agent           │
                    │  (XGBoost, RF, Ridge, etc.)        │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 12: Fuzzy Logic Agent        │
                    │  (221 Mamdani rules)               │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 13: Prediction Agent         │
                    │  (Yield prediction + confidence)   │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 14: Recommendation Agent v2  │
                    │  (Top-3 alternatives + scores)     │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 15: Benchmark Agent [NEW]    │
                    │  (Pipeline metrics + comparison)   │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 16: Explainability Agent     │
                    │  (Per-prediction explanations)     │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  Step 17: Ready Reckoner Agent v2  │
                    │  (JSON/CSV/MD/HTML exports)        │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │   Outputs: Excel, HTML, Models,    │
                    │   Benchmark, Explainability, Reports│
                    └──────────────────────────────────┘
```

---

## Pipeline Agents (v2.0 — 17-Step Pipeline)

| Step | Agent | Description | Test Count |
|------|-------|-------------|------------|
| 1 | **Knowledge Agent** | PDF ingestion, DOI/crop/year detection, SQLite registry | — |
| 2 | **Knowledge Integration** | Crop yield ranges, soil/climate defaults from reference DB | 16 |
| 3 | **Extraction Agent** | 6 hybrid readers (pdfminer, camelot, pdfplumber, poppler, OCR, semantic) | — |
| 4 | **Evidence Fusion** | Confidence-weighted multi-reader deduplication, unit normalization | 11 |
| 5 | **Ontology Agent** | Column name resolution via synonyms, case-insensitive matching | 22 |
| 6 | **Table Intelligence** | Table type classification (yield/treatment/growth), statistics parsing | 20 |
| 7 | **Schema Population** | Derived columns (Yield_per_Hectare, GDD, NUE, N×P interaction) | 16 |
| 8 | **Validation Agent v2** | Biological range checks, agronomic consistency, MAD outlier detection | 17 |
| 9 | **Feature Agent v2** | 150+ engineered features (interactions, ratios, polynomials, bins) | 9 |
| 10 | **Model Selection** | Adaptive model pool by sample size, GridSearchCV hyperparameter optimization | 12 |
| 11 | **Training Agent** | XGBoost, RandomForest, Ridge, etc. with cross-validation | — |
| 12 | **Fuzzy Logic Agent** | 221 Mamdani rules for N/P/K recommendations | 19 |
| 13 | **Prediction Agent** | Yield prediction with confidence intervals | — |
| 14 | **Recommendation Agent v2** | Top-3 alternatives with economic/environmental/risk scoring | 18 |
| 15 | **Benchmark Agent** ★ | Pipeline metrics, historical comparison, trend analysis | 22 |
| 16 | **Explainability Agent** ★ | Per-prediction JSON: model, features, rules, evidence, uncertainty | 23 |
| 17 | **Ready Reckoner Agent v2** | JSON/CSV/MD/HTML/statistics exports | 14 |

★ = new in v2.0

### Supporting Modules

| Module | Description | Tests |
|--------|-------------|-------|
| **Knowledge Graph** | NetworkX directed graph, 8 node types, 8 edge types, PageRank, centrality | 33 |
| **Provenance Agent** | Per-cell DOI, extraction method, page/table, confidence tagging | 27 |
| **Evaluation Dashboard** | 8-metric HTML dashboard with JSON metrics export | 26 |
| **Performance Targets** | Target validation with direction-aware pass/fail checks | 15 |
| **Continuous Learning** | Drift detection, retrain triggers, model versioning | 31 |
| **Benchmark Agent** | Pipeline metrics, historical comparison, trend analysis | 22 |
| **Explainability Agent** | Per-prediction explanations with feature importance | 23 |

---

## Extraction Engine

The hybrid extraction engine orchestrates 6 specialized readers:

| Reader | Method | Strength |
|--------|--------|----------|
| **Pdfminer** | Text extraction + regex NLP | Layout-aware text parsing |
| **Camelot** | Lattice + stream detection | Tables with clear borders |
| **Pdfplumber** | Per-page table extraction | Auto header detection |
| **Poppler** | pdftotext -layout mode | Fast, layout-preserving |
| **OCR** | Tesseract + pdf2image | Scanned/image PDFs |
| **Semantic** | Sentence-transformers + NER | Context-aware extraction |

Results are merged via **confidence-weighted deduplication** (ValidationAgent).

---

## ML Models

12 models trained on extracted yield data:

| Model | Type | R² | CV R² (mean±std) |
|-------|------|-----|-------------------|
| Multiple Linear Regression | Linear | 0.9896 | 0.976±0.021 |
| Ridge Regression | Regularized | 0.9849 | 0.975±0.021 |
| Lasso Regression | Regularized | 0.9896 | 0.977±0.021 |
| Elastic Net | Regularized | 0.9874 | 0.978±0.015 |
| Decision Tree | Tree | 1.0000 | 0.996±0.005 |
| Random Forest | Ensemble | 0.9997 | 0.988±0.020 |
| Extra Trees | Ensemble | 1.0000 | 0.997±0.003 |
| Gradient Boosting | Boosting | 1.0000 | 0.999±0.002 |
| XGBoost | Boosting | 1.0000 | 0.656±0.685 |
| AdaBoost | Boosting | 1.0000 | 0.997±0.003 |
| SVR | Kernel | 0.3103 | -0.307±0.683 |
| KNN Regression | Instance | 0.9976 | 0.961±0.044 |

**Multiple Regression (OLS):**
```
Yield = 33.20 - 22.01 × Spike_Length + 6.73 × Table_Row
R² = 0.9896, Adj R² = 0.9872, F = 426.74 (p < 0.001)
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| PDFs processed | 24 (21 unique, 3 duplicates) |
| Table rows extracted | 187 |
| Papers with yield data | 12 |
| ML models trained | 12 |
| Best CV R² model | Gradient Boosting (0.999) |
| Most generalizable | Elastic Net (CV R² = 0.978) |
| Fuzzy rules | 221 Mamdani rules |
| Engineered features | 150+ |
| Knowledge graph edges | 8 edge types, 8 node types |
| Pipeline agents | 17 (including 2 new in v2.0) |
| Test suites | 18 custom suites |
| Tests passing | 351/351 (0 failures) |
| Framework score | 87% production-ready |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Poppler (for pdftotext) — [install guide](#poppler-installation)

### Installation

```bash
git clone https://github.com/matrixflora/Ready-Reckoner-Table-AI-Framework.git
cd Ready-Reckoner-Table-AI-Framework
pip install -r requirements.txt
```

### Poppler Installation (Windows)

Download Poppler from: https://github.com/osber/poppler-windows/releases

Extract to `C:\poppler\poppler-24.08.0\Library\bin\` and add to PATH, or place in a known location.

### Run Full Pipeline

```bash
python run_aaf_pipeline.py
```

### Run Post-Pipeline Analysis

```bash
python post_pipeline_analysis.py
```

### Export All Results

```bash
python export_all_results.py
```

### Run Individual Components

```bash
# Hybrid extraction only
python -m aaif.extraction.hybrid_extractor "Data ADES/"

# Pdfplumber extraction
python pdfplumber_extraction.py

# Schema generation
python universal_schema_generator.py
```

---

## Project Structure

```
Ready-Reckoner-Table-AI-Framework/
├── agri_ai_agent/                  # Core agent framework
│   ├── agents/                     # All pipeline agents
│   │   ├── base_agent.py           # BaseAgent ABC (run, process, contract)
│   │   ├── knowledge_agent.py      # Step 1: PDF ingestion + registry
│   │   ├── knowledge_integration_agent.py  # Step 2: Crop/Soil/Climate ref DB
│   │   ├── extraction_agent.py     # Step 3: 6 hybrid extraction readers
│   │   ├── evidence_fusion_agent.py # Step 4: Confidence-weighted merge
│   │   ├── ontology_agent.py       # Step 5: Column name resolution
│   │   ├── table_intelligence_agent.py  # Step 6: Table type classification
│   │   ├── schema_population_agent.py   # Step 7: Derived column computation
│   │   ├── validation_agent.py     # Step 8: Biological + agronomic rules
│   │   ├── feature_agent.py        # Step 9: 150+ engineered features
│   │   ├── model_selection_agent.py # Step 10: Adaptive pool + GridSearch
│   │   ├── training_agent.py       # Step 11: Model training
│   │   ├── fuzzy_logic_agent.py    # Step 12: 221 Mamdani rules
│   │   ├── prediction_agent.py     # Step 13: Yield prediction
│   │   ├── recommendation_agent.py # Step 14: Top-3 alternatives + scores
│   │   ├── benchmark_agent.py      # Step 15: Pipeline metrics (v2.0)
│   │   ├── explainability_agent.py # Step 16: Per-prediction explanations (v2.0)
│   │   ├── ready_reckoner_agent.py # Step 17: JSON/CSV/MD/HTML exports
│   │   ├── continuous_learning_agent.py  # Drift detection + retraining
│   │   └── provenance_agent.py     # Per-cell provenance tracking
│   ├── config/                     # Configuration
│   │   ├── schema.py               # UAMS_COLUMNS (138 vars)
│   │   └── settings.py             # AgriAISettings (all directories)
│   ├── contracts/                  # Inter-agent messages
│   │   └── messages.py             # AgentContract, OrchestratorState
│   ├── dashboard/                  # Evaluation dashboard
│   │   └── __init__.py             # EvaluationDashboard (8 metrics)
│   ├── evaluation/                 # Performance targets
│   │   └── __init__.py             # PerformanceValidator
│   ├── knowledge_graph/            # Knowledge graph module
│   │   └── graph.py                # KnowledgeGraph (networkx, 8 node types)
│   ├── rules/                      # Fuzzy logic
│   │   ├── fertilizer_rules.yaml   # 221 Mamdani rules
│   │   ├── membership_functions.py # Input membership functions
│   │   └── output_memberships.py   # Output membership functions
│   ├── orchestrator.py             # 17-step pipeline orchestrator
│   └── utils/                      # Logging, helpers
│
├── tests/                          # Test suites (351 tests)
│   ├── test_evidence_fusion_agent.py     # 11 tests
│   ├── test_ontology_agent.py            # 22 tests
│   ├── test_table_intelligence_agent.py  # 20 tests
│   ├── test_schema_population_agent.py   # 16 tests
│   ├── test_validation_agent_v2.py       # 17 tests
│   ├── test_feature_agent_v2.py          # 9 tests
│   ├── test_model_selection_agent.py     # 12 tests
│   ├── test_knowledge_integration_agent.py  # 16 tests
│   ├── test_recommendation_agent_v2.py   # 18 tests
│   ├── test_fuzzy_expanded.py            # 19 tests
│   ├── test_ready_reckoner_v2.py         # 14 tests
│   ├── test_continuous_learning_agent_v2.py  # 31 tests
│   ├── test_knowledge_graph.py           # 33 tests
│   ├── test_provenance_agent.py          # 27 tests
│   ├── test_evaluation_dashboard.py      # 26 tests
│   ├── test_performance_targets.py       # 15 tests
│   ├── test_benchmark_agent.py           # 22 tests (v2.0)
│   └── test_explainability_agent.py      # 23 tests (v2.0)
│
├── config/                         # Agent configuration
│   ├── benchmark.yaml              # BenchmarkAgent settings
│   └── explainability.yaml         # ExplainabilityAgent settings
│
├── Data ADES/                      # Source research PDFs (gitignored)
├── outputs/                        # Generated results
│   ├── benchmark/                  # Benchmark history + reports
│   ├── models/                     # Trained ML models
│   ├── predictions/                # Prediction outputs
│   ├── reckoners/                  # Ready Reckoner exports
│   ├── reports/explainability/     # Per-prediction explanations
│   └── contracts/                  # Agent execution contracts
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata + pytest config
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## Test Suites

```
tests/
├── test_benchmark_agent.py              22 tests
├── test_continuous_learning_agent_v2.py  31 tests
├── test_evaluation_dashboard.py          26 tests
├── test_evidence_fusion_agent.py         11 tests
├── test_explainability_agent.py          23 tests
├── test_feature_agent_v2.py               9 tests
├── test_fuzzy_expanded.py                19 tests
├── test_knowledge_graph.py               33 tests
├── test_knowledge_integration_agent.py   16 tests
├── test_model_selection_agent.py         12 tests
├── test_ontology_agent.py                22 tests
├── test_performance_targets.py           15 tests
├── test_provenance_agent.py              27 tests
├── test_ready_reckoner_v2.py             14 tests
├── test_recommendation_agent_v2.py       18 tests
├── test_schema_population_agent.py       16 tests
├── test_table_intelligence_agent.py      20 tests
└── test_validation_agent_v2.py           17 tests
─────────────────────────────────────────────
Total:                                351 tests ✅
```

---

## Output Files

### Master Results Workbook (`outputs/export/AAIF_All_Results.xlsx`)

23 sheets covering every aspect of the analysis:

| Sheet | Description |
|-------|-------------|
| Ready_Reckoner_24Papers | Per-paper yield summary (24 rows) |
| Model_Performance | 12 ML models × 16 metrics |
| Regression_Coefficients | OLS coefficients with p-values |
| Regression_Summary | R², F-stat, AIC, BIC, Durbin-Watson |
| Pipeline_ReadyReckoner | Pipeline-generated table (21 rows) |
| Ingestion_Report | All 24 PDFs metadata |
| Extraction_Report | Per-paper extraction success |
| Hybrid_Extraction | Raw extraction results (44 cols) |
| Treatment_Level_Data | 121 treatment rows (139 cols) |
| Features_Dataset | 146 engineered features |
| Feature_Importance | Feature importance rankings |
| Feature_Dictionary | 142 feature descriptions |
| Validation_Report | Per-column quality checks |
| Universal_Schema | Full 138-column UAMS |
| Fertilizer_Recommendations | Fuzzy logic recommendations |
| + 8 more sheets | Correlation, Encoding, VIF, etc. |

### Word Reports

| File | Description |
|------|-------------|
| `AAIF_Model_Report.docx` | Full 8-section narrative report |
| `AAIF_Regression_Report.docx` | Focused regression analysis |

---

## Dependencies

```
# PDF Processing
pdfminer.six
pdfplumber
PyPDF2
poppler-utils (system)

# Machine Learning
scikit-learn
xgboost
lightgbm
catboost

# Data Science
numpy
pandas
scipy
statsmodels

# NLP / Semantic
sentence-transformers
torch

# Fuzzy Logic
scikit-fuzzy

# Report Generation
python-docx
openpyxl
weasyprint (optional)

# Database
duckdb
chardet
```

See `requirements.txt` for exact versions.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{aaif2026,
  title={Ready Reckoner Table - AI Framework},
  author={Matrix Flora},
  year={2026},
  url={https://github.com/matrixflora/Ready-Reckoner-Table-AI-Framework}
}
```
