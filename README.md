# Ready Reckoner Table — AI Framework

An end-to-end **Agentic Agricultural Intelligence Framework (AAIF)** that ingests research PDFs, extracts treatment-level agronomic data using 6 hybrid extraction engines, trains 12 machine learning models, applies fuzzy logic for fertilizer recommendations, and produces a **Ready Reckoner Table** — a concise per-crop yield decision support tool.

> **Live results:** 24 research papers → 187 treatment rows → 12 ML models → Ready Reckoner Table

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Phases](#pipeline-phases)
- [Extraction Engine](#extraction-engine)
- [ML Models](#ml-models)
- [Results Summary](#results-summary)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
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
| **Schema Mapping** | Maps to UAMS v1.0 (138 agricultural variables) |
| **ML Training** | 12 models: Linear, Ridge, Lasso, ElasticNet, RF, XGB, etc. |
| **Fuzzy Logic** | 16 Mamdani rules for fertilizer recommendations |
| **Ready Reckoner** | Per-paper summary table with yields, treatments, recommendations |

---

## Architecture

```
                    ┌──────────────────────────┐
                    │    PDF Research Papers     │
                    │      (24 papers)           │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 1: Ingestion       │
                    │  (DOI, Crop, Year detect) │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼──────┐ ┌────────▼───────┐ ┌────────▼───────┐
    │  Phase 2-3:     │ │  Phase 2-3:    │ │  Phase 2-3:    │
    │  Regex Extract  │ │  Table Extract │ │  Poppler       │
    └─────────┬──────┘ └────────┬───────┘ └────────┬───────┘
              │                  │                   │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 4: Validation      │
                    │  Phase 5: Features        │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼──────┐ ┌────────▼───────┐ ┌────────▼───────┐
    │  Phase 6:       │ │  Phase 7:      │ │  Phase 8:      │
    │  ML Training    │ │  Fuzzy Logic   │ │  Recommendations│
    │  (12 models)    │ │  (16 rules)    │ │  (per crop)    │
    └─────────┬──────┘ └────────┬───────┘ └────────┬───────┘
              │                  │                   │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 9: Ready Reckoner  │
                    │  Phase 10: Learning       │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Outputs: Excel, HTML,   │
                    │   Models, Reports         │
                    └──────────────────────────┘
```

---

## Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Ingestion** | Read PDFs, detect crop/DOI/year, register in SQLite |
| 2-3 | **AI Extraction** | Regex NLP + 6 hybrid table readers → Universal Schema |
| 4 | **Validation** | Range checks, outlier detection, unit normalization |
| 5 | **Feature Engineering** | 16 derived features (GDD, NUE, WUE, interactions) |
| 6 | **Model Training** | 12 ML models with cross-validation |
| 7 | **Fuzzy Logic** | Mamdani inference for fertilizer dosing |
| 8 | **Recommendations** | Crop-specific fertilizer recommendations |
| 9 | **Ready Reckoner** | Per-paper summary table (XLSX + HTML) |
| 10 | **Continuous Learning** | Paper registry, duplicate detection |

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
| Pipeline runtime | ~325 seconds |

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
├── aaif/                           # Core AAIF package
│   └── extraction/                 # Hybrid extraction engine
│       ├── __init__.py             # UAMS_COLUMNS, parse_value, norm
│       ├── hybrid_extractor.py     # Orchestrator for all readers
│       ├── pdfminer_reader.py      # Pdfminer + regex NLP
│       ├── camelot_reader.py       # Camelot table detection
│       ├── pdfplumber_reader.py    # Pdfplumber extraction
│       ├── poppler_reader.py       # Poppler pdftotext
│       ├── ocr_reader.py           # Tesseract OCR
│       ├── semantic_extractor.py   # Sentence-transformers NER
│       └── validation_agent.py     # Confidence-weighted merge
│
├── agri_ai_agent/                  # Agent framework
│   ├── config/                     # Schema + settings
│   │   ├── schema.py               # UAMS_COLUMNS (138 vars)
│   │   └── settings.py             # AgriAISettings
│   ├── contracts/                  # Inter-agent messages
│   └── rules/                      # Fuzzy logic
│       ├── fertilizer_rules.yaml   # 16 Mamdani rules
│       └── membership_functions.py # Fuzzy membership fns
│
├── Data ADES/                      # Source research PDFs (gitignored)
│
├── outputs/                        # Generated results
│   ├── tables/                     # Model performance, regression
│   │   ├── Model_Performance_Assessment.xlsx
│   │   └── Multi_Regression_Results.xlsx
│   ├── recommendations/            # Fertilizer recommendations
│   ├── export/                     # Final exported reports
│   │   ├── AAIF_All_Results.xlsx   # Master workbook (23 sheets)
│   │   ├── AAIF_Model_Report.docx  # Full narrative report
│   │   └── AAIF_Regression_Report.docx
│   ├── Ready_Reckoner.xlsx         # Pipeline Ready Reckoner
│   ├── Ready_Reckoner_24Papers.xlsx
│   ├── AAIF_Final_Report.html
│   └── ... (40+ output files)
│
├── models/                         # Trained ML models
│   ├── xgboost_model.pkl
│   └── regression_model.pkl
│
├── database/                       # Paper registry
│   └── paper_registry.sqlite
│
├── fuzzy_logic/                    # Fuzzy system config
│
├── run_aaf_pipeline.py             # Main 10-phase pipeline
├── post_pipeline_analysis.py       # Post-pipeline ML analysis
├── export_all_results.py           # Export to Excel/Word
├── pdfplumber_extraction.py        # Standalone pdfplumber
├── enhanced_extraction.py          # Enhanced treatment extraction
├── universal_schema_generator.py   # UAMS schema generator
│
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata
├── LICENSE                         # MIT License
└── README.md                       # This file
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
