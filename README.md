# Agricultural Data Engineering System (ADES)

An Agentic AI pipeline that transforms heterogeneous agricultural datasets into the **Universal Agricultural Machine Learning Schema (UAMS) v1.0** - production-ready for regression, ensemble, deep learning, time series, explainable AI, causal inference, and future agentic systems.

## Architecture

ADES decomposes the original monolithic merge agent into **12 specialized autonomous agents** coordinated by an **Orchestrator Agent**:

| # | Agent | Responsibility |
|---|-------|---------------|
| 1 | **Dataset Ingestion Agent** | Read Excel, CSV, TSV, Parquet, JSON, SQLite, DuckDB; auto-detect encoding, delimiter, worksheet |
| 2 | **Schema Mapping Agent** | Map columns to UAMS v1.0; identify duplicates, synonyms, aliases |
| 3 | **Ontology Mapping Agent** | Map variables to AGROVOC, Crop Ontology, Plant Ontology, ENVO, FAO |
| 4 | **Unit Harmonization Agent** | Convert acre/ha, kg/acre→kg/ha, inch→cm, ppm, etc. |
| 5 | **Quality Assurance Agent** | Detect duplicates, impossible values, outliers, mixed types |
| 6 | **Feature Engineering Agent** | GDD, Heat Units, Harvest Index, NUE, WUE, interaction features |
| 7 | **Leakage Detection Agent** | Flag post-harvest variables; set `Feature_Available_Before_Prediction` |
| 8 | **Encoding Agent** | Crop_Code, Season_Code, Variety_Code, Fertilizer_Code, etc. |
| 9 | **Statistical Diagnostics Agent** | Mean, median, skewness, kurtosis, correlation, VIF, normality tests |
| 10 | **Model Readiness Agent** | Evaluate compatibility with 20+ model families |
| 11 | **Documentation Agent** | Generate README, Dataset Card, Feature Dictionary, reports |
| 12 | **Export Agent** | CSV, Parquet, SQLite, DuckDB, 15-sheet Excel workbook |

## Quick Start

```bash
pip install -r requirements.txt

# Run via CLI
python -m ades run input_dataset.xlsx

# Or using the ades command (after setup)
ades run input_dataset.xlsx

# Legacy entry point (backward compatible)
python universal_schema_generator.py
```

## Outputs

All outputs written to `outputs/`:

| File | Description |
|------|-------------|
| `Universal_Agricultural_ML_Master.csv` | Full ML-ready dataset |
| `Universal_Agricultural_ML_Master.parquet` | Columnar storage format |
| `Universal_Agricultural_ML_Master.sqlite` | SQLite database |
| `Universal_Agricultural_ML_Master.duckdb` | DuckDB database (if duckdb installed) |
| `Universal_Agricultural_Machine_Learning_Schema_v1.xlsx` | 15-sheet Excel workbook |
| `Variable_Mapping.csv` | Column mapping history |
| `Ontology_Mapping.csv` | Ontology term mappings |
| `Quality_Report.md` | Data quality findings |
| `Model_Readiness_Report.md` | Model compatibility report |
| Various reports | Statistics, VIF, missing data, validation |

## Architecture Details

- **Plugin-based**: each agent is independent with its own prompts, config, tests, logging, and retry logic
- **JSON contracts**: agents communicate only through structured `AgentContract` objects via `ades/contracts/messages.py`
- **Checkpoint recovery**: orchestrator saves parquet checkpoints per step; supports `--incremental` mode
- **Provenance**: full pipeline execution log written to `Pipeline_Provenance.json`

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=ades --cov-report=term
```

## Project Structure

```
├── ades/                      # Core ADES package
│   ├── agents/                # 12 specialized agents
│   ├── config/                # UAMS schema + settings
│   ├── contracts/             # JSON inter-agent contracts
│   ├── utils/                 # Logging + IO utilities
│   ├── orchestrator.py        # Orchestrator Agent
│   └── cli.py                # CLI entry point
├── tests/                     # Automated tests
├── data/master_datasets/      # Input Excel files (gitignored)
├── outputs/                   # Generated outputs (gitignored)
└── universal_schema_generator.py  # Backward-compatible wrapper
```

## License

See repository license.
