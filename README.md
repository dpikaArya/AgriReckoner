# Agricultural Intelligence Framework (AAIF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-569--passing-green.svg)]()
[![Schema](https://img.shields.io/badge/UAMS-v2.0--296--columns-success.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

An **agentic pipeline** that converts unstructured agricultural research PDFs into ML-ready datasets, yield-prediction models, and fuzzy-logic fertilizer recommendations — with full per-cell provenance tracking.

**Key capabilities:** 6-reader hybrid PDF extraction, 23-agent pipeline, UAMS v2.0 schema (26 groups, 296 columns), 553-entry variant map, 8 regression model families, 221 Mamdani fuzzy rules, per-cell provenance tracking, external data enrichment (NASA POWER, SoilGrids, ISRIC, CGIAR, FAOSTAT, HuggingFace…), and continuous learning with drift detection.

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

AAIF is an end-to-end scientific-AI platform that takes agronomic literature and tabular field data and produces:

1. **UAMS datasets** — observations harmonised into the Universal Agricultural Metadata Schema v2.0 (296 columns across 26 groups), with provenance from source paper to schema cell.
2. **Yield-prediction models** — cross-validated regressors (Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, SVR, Linear Regression) with honest, sample-size-aware evaluation.
3. **Fuzzy fertilizer recommendations** — a 221-rule Mamdani fuzzy engine that turns soil/climate conditions into crop- and dose-specific recommendations delivered via an interactive Ready Reckoner.

The system is built around **23 specialised agents** organised into **5 pipeline phases** (0–4), fronted by a **PHASE −1 external data source layer**, and supported by a **continuous learning** loop with drift detection and versioned model/dataset history.

| Property | Value |
|----------|-------|
| Agents | 23 sequential + external-data connectors |
| Pipeline phases | 5 (0–4) + PHASE −1 (External Data) |
| UAMS schema | v2.0 — 296 columns, 26 groups (A–Z) |
| Variant map | 553 unique source→UAMS mappings |
| Model families | 8 regression + adaptive GridSearchCV tuning |
| Fuzzy rules | 221 Mamdani (10 input MFs / 3 output MFs) |
| Python | 3.10+ (CI matrix: 3.10 / 3.11 / 3.12) |
| License | Apache 2.0 |

---

## Architecture

```
Research Papers / Master Datasets / External APIs
        │
        ▼
 PHASE −1  External Data Layer   ──  Connector registry (CGIAR, FAOSTAT, NASA POWER,
        │                            SoilGrids, ISRIC, Zenodo, Mendeley, Kaggle, ICAR, SAU)
        ▼
 PHASE 0  Ingestion & Normalization ── ① Repository Sync → ② External Data Source
        │                               ③ Dataset Normalization → ④ Ingestion Bridge
        ▼
 PHASE 1  Extraction & Knowledge  ── ⑤ 6-Reader Hybrid Extraction → ⑥ Evidence Fusion
        │                             ⑦ Ontology → ⑧ Table Intelligence → ⑨ Schema Population
        │                             ⑩ Knowledge Integration → ⑪ Knowledge → ⑫ Observation Gen.
        ▼
 PHASE 2  Validation & Features   ── ⑬ Validation → ⑭ Feature Store → ⑮ Feature Engineering
        │                             (140+ composite features: NPK Index / GDD / NUE / WUE)
        ▼
 PHASE 3  ML Training & Prediction ── ⑯ Model Selection → ⑰ Training → ⑱ Prediction
        │
        ▼
 PHASE 4  Recommendation & Export ── ⑲ Recommendation → ⑳ Fuzzy Logic → ㉑ Benchmark
        │                             ㉒ Explainability → ㉓ Ready Reckoner
        ▼
   UAMS v2.0  •  Models  •  Ready Reckoner  •  Reports
```

The full Mermaid flowchart of the 23-agent pipeline lives in [docs/flowchart.md](docs/flowchart.md); the detailed component reference is in [ARCHITECTURE.md](ARCHITECTURE.md).

### v3 Enrichment Pipeline

A standalone enrichment pipeline (`run_enrichment.py`) that bypasses the LLM orchestrator for external data joins, delivering a **95% faster** enrichment pass with no LLM cost:

| Metric | v2 (23-agent) | v3 (standalone) | Improvement |
|--------|---------------|-----------------|-------------|
| Pipeline runtime | ~45 min (LLM-bound) | **~2 min** | **95% faster** |
| External sources | 3 (LLM-limited) | **5+ concurrent** | **+67%** |
| UAMS columns matched | 34 / 296 | **67 / 296** | **+97%** |
| External columns added | 0 | **9** (PARAMETER, Year, Month, Value, DOY, Property, Depth, Statistic, Unit) | new capability |
| Spatial join | none | **1°-bin matching** on lat/lon | new capability |
| Column mapping | manual per agent | **VARIANT_MAP-driven** (auto-mapped) | automated |

---

## Pipeline Phases

### PHASE −1 — External Data Source Layer
Plugin connectors auto-discovered via `pkgutil`: CGIAR, FAOSTAT, NASA POWER, SoilGrids, ISRIC, Zenodo, Mendeley, Kaggle, ICAR, and SAU portals. Every source emits a standardised `DatasetPackage` (metadata + optional bytes), with health checks, per-source retry, quick-sync filtering, and a SQLite-backed `DatasetRegistry`.

### PHASE 0 — Ingestion & Normalization
- **① Repository Sync** — change detection over source repositories
- **② External Data Source** — plugin ingestion of public ag datasets
- **③ Dataset Normalization** — heterogeneous inputs → `DatasetPackage`
- **④ Dataset Ingestion Bridge** — Paper_ID / Dataset_ID hierarchy assignment

### PHASE 1 — Extraction & Knowledge
- **⑤ Extraction** — 6-reader hybrid: Pdfminer / Camelot / Pdfplumber / Poppler / OCR / Semantic
- **⑥ Evidence Fusion** — confidence-weighted multi-reader de-duplication
- **⑦ Ontology** — column → UAMS mapping via the 553-entry variant map
- **⑧ Table Intelligence** — table classification and statistics
- **⑨ Schema Population** — derived columns and missing-value inference
- **⑩ Knowledge Integration** — domain-aware imputation
- **⑪ Knowledge** — domain constraint enforcement
- **⑫ Observation Generation** — hierarchy builder (Paper_ID / Dataset_ID)

### PHASE 2 — Validation & Features
- **⑬ Validation** — biological range checks, IQR outliers, unit harmonisation
- **⑭ Feature Store** — validation gate + Parquet persistence
- **⑮ Feature Engineering** — 140+ composite features (NPK index, GDD, NUE, WUE, etc.)

### PHASE 3 — ML Training & Prediction
- **⑯ Model Selection** — GridSearchCV, SelectKBest, adaptive model pool, nested CV, leakage checks
- **⑰ Training** — 8 regression families with 5-fold CV / leave-one-out, feature importance, model persistence
- **⑱ Prediction** — best-model inference for 5 target variables

### PHASE 4 — Recommendation & Export
- **⑲ Recommendation** — top-3 treatment alternatives
- **⑳ Fuzzy Logic** — 221 Mamdani rules, centroid defuzzification
- **㉑ Benchmark** — phase timing, quality and coverage metrics
- **㉒ Explainability** — SHAP / permutation feature attribution
- **㉓ Ready Reckoner** — per-crop decision tables (Excel / CSV / HTML / JSON / Markdown)

### Continuous Learning
`ChangeDetector` (drift detection) → `DependencyGraph` (minimal re-execution) → `IncrementalEngine` (change-driven partial retrains) → `VersionHistory` (model/dataset version ledger).

**Production run:** `PROD_20260729_134301` — 12 stages in **134.2 s**, 22/22 agents completed, 0 failures, 66 observations.

---

## Schema & Data

### Universal Agricultural Metadata Schema (UAMS v2.0)

The schema is the single source of truth for harmonised observations. It evolved from **UAMS v1.0 (138 columns, 14 groups)** to **UAMS v2.0 (296 columns, 26 groups)**.

| UAMS Group | Contents |
|------------|----------|
| A. Paper Metadata | Paper_ID, DOI, Journal, Year, Authors, Country |
| B. Crop Information | Crop, Scientific_Name, Variety, Season |
| C. Experimental Design | Design, Replications, Plot_Size, Spacing, Location |
| D. Environment | Latitude, Longitude, Altitude, Temperature, Rainfall, Humidity |
| E. Soil Properties | pH, EC, Organic_Carbon, N, P, K, micronutrients |
| F. Fertilizer Information | doses, sources, application details |
| G–I. Growth / Yield / Grain Quality | plant parameters, yield components, grain quality |
| J. ML Target Variables | yield & quality targets for modelling |
| K–N. ML Pipeline Columns | engineered features, leakage labels, encodings, predictions |
| O–U. Soil Biology & Physics | biological properties, enzyme activity, microbial community, exchangeable cations, remote-sensing indices, physical properties |
| V–Z. Advanced Domains | nematode ecology, economics, nutrient uptake, irrigation & management, pH variants |

- **296 canonical columns** across **26 groups (A–Z)** — see `outputs/Universal_Agricultural_Schema.csv`
- **553-entry VARIANT_MAP** resolves paper-specific column names to canonical UAMS columns
- **Provenance everywhere** — every schema cell traces back to source paper, page, and table

### Data Sources

- Master datasets: `BellPepper_Master_Dataset.xlsx`, `BlackWheat_Master_Dataset.xlsx`, `Carrot_Master_Dataset.xlsx`, `Gram_Master_Dataset.xlsx`, `Spinach_Master dataset.xlsx`
- Research PDF corpus (`DataADES/`)
- External providers: NASA POWER (climate), SoilGrids (soil properties), ISRIC, CGIAR, FAOSTAT, Zenodo, HuggingFace

---

## ML Models

### Model Pool

| Model | Min Samples | Notes |
|-------|-------------|-------|
| Ridge | 1 | baseline regularised linear model |
| Linear Regression | 10 | baseline, no hyperparameters |
| Random Forest | 30 | `n_estimators=100`, `n_jobs=-1` |
| XGBoost | 50 | requires OpenMP; skipped gracefully if unavailable |
| Gradient Boosting | 50 | adaptive learning-rate grid |
| Lasso / ElasticNet / SVR | — | penalised linear + kernel regression families |

### Evaluation Protocol

- **Sample-size-aware:** n ≥ 30 → Repeated 5×3-fold CV (robust); 8–29 → Leave-One-Out (advisory); < 8 → refuses to report metrics.
- **Leakage-safe:** name-based + correlation-based (|ρ| ≥ 0.999) leakage detection; `SimpleImputer` fitted **inside each fold** via a `Pipeline`.
- **Metrics:** R² (mean ± std across folds), RMSE, MAE, MAPE, n, CV scheme, robustness flag.

### Trained Artifacts

9 serialised models + a best-overall `yield_model.pkl`, with per-model `metrics.csv`, `feature_importance.csv`, and an HTML leaderboard.

---

## Fuzzy Logic

- **221 Mamdani rules** (`fuzzy_logic/fertilizer_rules.yaml`)
- **10 input membership functions** (soil N/P/K, pH, rainfall, temperature, humidity, …) and **3 output membership functions** (dose action, interval, risk)
- **Centroid defuzzification** for crisp recommendation doses
- Fuzzy outputs are fused with ML yield predictions to produce the **Ready Reckoner** — per-crop, per-scenario fertilizer decision tables with confidence, economic, environmental, and risk scores.

---

## Results Summary

### UAMS Updation

| Metric | Before | Now | Change |
|--------|--------|-----|--------|
| Schema version | UAMS v1.0 | **UAMS v2.0** | major upgrade |
| Schema columns | 138 | **296** | **+114%** |
| Schema groups | 14 | **26 (A–Z)** | **+86%** |
| Variant map | 190 (duplicated) | **553 deduplicated entries** | +2× usable coverage |
| UAMS columns matched by enrichment | 34 / 296 | **67 / 296** | **+97%** |
| External columns added | 0 | **9** (PARAMETER, Year, Month, Value, DOY, Property, Depth, Statistic, Unit) | new capability |
| Enrichment runtime | ~45 min (LLM-bound) | **~2 min** | **95% faster** |

### Best Models — Yield Prediction (CPY, 5,334 samples, 73 features, 5-fold CV)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **Extra Trees** | **0.9890** | **898.6** | 291.7 |
| **XGBoost** | **0.9853** | **1028.0** | 451.1 |
| **Random Forest** | **0.9845** | **1067.7** | 359.4 |
| Decision Tree | 0.9758 | 1300.6 | 420.9 |
| Gradient Boosting | 0.9731 | 1440.8 | 874.0 |
| Ridge | 0.3068 | 6034.1 | 2066.6 |

The **Extra Trees** ensemble achieves the best generalization on the large dataset (R² = **0.989**, RMSE = **898.6 kg/ha**), closely followed by **XGBoost** and **Random Forest** — all tree-based ensembles clearly outperform the linear baselines.

### Best Models — Spike-Length Regression (12 samples, 2 features)

| Model | R² | RMSE | CV R² |
|-------|-----|------|-------|
| **Gradient Boosting** | **1.0000** | **0.0007** | 0.9988 |
| **Extra Trees** | **1.0000** | 0.0000 | 0.9969 |
| **Random Forest** | **0.9997** | **0.4824** | 0.9881 |
| Multiple Linear Regression | 0.9896 | 2.8737 | 0.9764 |
| Elastic Net | 0.9874 | 3.1578 | 0.9777 |

> Small-sample results (n = 12) are advisory; CV R² (Repeated 5-fold) is the trusted ranking signal. **Gradient Boosting** leads with CV R² = **0.9988**.

### Ready Reckoner / Recommendations

| Metric | Value |
|--------|-------|
| Recommendation scenarios | 66 |
| Confidence score (mean) | 0.404 |
| Economic score (mean) | 0.97 |
| Environmental score (mean) | 0.70 |
| Risk distribution | 63 low / 3 moderate / 0 high |
| Mean recommended dose | 6.17 kg/ha (Jaivik Poshak, bio-NPK) |

---

## Quick Start

```bash
git clone https://github.com/Adpika/Agricultural-Intelligence-Framework.git
cd Agricultural-Intelligence-Framework
pip install -e .
agriai run --file data.csv         # run on prepared data
agriai run --papers "DataADES/"    # ingest PDFs then run pipeline
```

Requires Python 3.10+ and [Poppler](https://github.com/osber/poppler-windows/releases) (Windows) or `brew install libomp` (macOS for XGBoost).

### Basic Usage

```python
from agri_ai_agent.orchestrator import Orchestrator
from agri_ai_agent.config.settings import AgriAISettings

settings = AgriAISettings()
orch = Orchestrator(settings)
df = orch.run(filepath="data.csv")
```

### Verify

```bash
python -m pytest -m "not live"        # 569 passed, 2 skipped
ruff check .                          # clean
ruff format --check .                 # clean
mypy src/ --ignore-missing-imports    # clean
```

---

## Project Structure

```
Agricultural-Intelligence-Framework/
├── agri_ai_agent/            # Core package (23 agents, orchestrator, config, extractors,
│                             #   external_data connectors, continuous_learning, rules)
├── src/                      # Supporting pipeline modules
│   ├── data_sources/         #   HTTP client, cache, storage, sync, config loader
│   ├── ml/                   #   training, benchmark, feature analysis, ensembles, validation
│   ├── validation/           #   schema, range, data-quality, provenance, reports
│   ├── provenance/           #   lineage tracking
│   └── utils/                #   logging, io helpers
├── DataADES/                 # Research PDF corpus
├── data/master_datasets/     # Master Excel datasets
├── fuzzy_logic/              # 221-rule Mamdani fuzzy engine
├── outputs/                  # All generated artifacts (schema, models, reports)
├── tests/                    # 569-test suite (incl. golden-baseline, e2e, agents)
├── benchmarks/               # Gold-standard benchmarking harness
├── evaluation/               # 11-stage evaluation scripts
├── docs/                     # Architecture, flowchart, info-flow docs
├── scripts/                  # Ontology doc generation, etc.
├── config/                   # API keys/config (apis.yaml)
├── production_run.py         # 12-stage end-to-end orchestrator
├── run_enrichment.py         # v3 standalone external-data enrichment
├── run_aaf_pipeline.py       # Legacy monolithic runner
├── pyproject.toml            # Package config (ruff, mypy, pytest)
└── .github/workflows/        # CI + Quality-Check workflows
```

---

## Output Files

All artifacts land in `outputs/`:

| File | Description |
|------|-------------|
| `Universal_Agricultural_Schema.csv` / `.xlsx` | 296-column UAMS v2.0 harmonised dataset |
| `Universal_Agricultural_ML_Master.csv` | ML-ready feature matrix |
| `features_dataset.csv` | Engineered feature set |
| `Model_Results_All.csv` | Per-model R² / RMSE / MAE leaderboard |
| `models/` | 9 serialised `.joblib` models + `yield_model.pkl` |
| `Ready_Reckoner.xlsx` / `.csv` / `.html` / `.json` | Per-crop fertilizer decision tables (66 scenarios) |
| `recommendations/fertilizer_recommendations.xlsx` | Fertilizer dose recommendations |
| `Validation_Report.csv` / `.xlsx` | Validation results |
| `feature_importance.csv` | Per-model feature importance |
| `Correlation_Matrix.csv` | Feature correlation matrix |
| `benchmark_history.json` | Run history and timings |
| `Pipeline_Manifest.json` / `Pipeline_Provenance.json` | Run provenance and manifests |
| `AAIF_Final_Report.html` | Incremental-mode execution report |
| `evaluation/` | Extraction-method comparison, schema validation, feature reports |
| `export/AAIF_All_Results.xlsx` | Consolidated Excel export |

---

## Dependencies

Core (`pyproject.toml`):

- **Data:** `pandas`, `numpy`, `pyarrow`, `openpyxl`, `duckdb`
- **Extraction:** `PyPDF2`, `pdfplumber`, `pdfminer.six`, `chardet`
- **ML:** `scikit-learn`, `xgboost`, `scipy`, `joblib`
- **External data:** `datasets`, `huggingface-hub`, `requests`
- **Config / misc:** `PyYAML`, `networkx`

Optional extras:

- `llm` — `openai`
- `ml` — `optuna`, `lightgbm`, `shap`
- `ml-full` — adds `catboost`, `python-docx`
- `dev` — `pytest`, `pytest-cov`, `ruff`, `mypy`, `pip-audit`

---

## License

Apache 2.0. See [LICENSE](LICENSE).
