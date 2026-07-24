# Universal Agricultural Machine Learning Schema (UAMS) Specification Repository

**Version:** 1.0  
**Status:** Stable  
**Last Updated:** 2026-07-04

This repository defines the **Universal Agricultural Machine Learning Schema (UAMS) v1.0** — a standardized schema for agricultural datasets designed to be immediately usable for machine learning, deep learning, time series forecasting, explainable AI, causal inference, digital twins, and future agentic AI systems.

## Purpose

Agricultural research generates heterogeneous datasets with inconsistent column naming, units, ontologies, and formats. UAMS provides a **single unified schema** that:

- Standardizes column names across crops, regions, and experiments
- Maps variables to international ontologies (AGROVOC, Crop Ontology, ENVO, FAO)
- Harmonizes units to SI standards
- Encodes categorical variables numerically
- Flags data leakage sources
- Engineers domain-relevant features
- Validates against physical constraints
- Documents dataset readiness for 20+ model families

## Schema Overview

| Group | Columns | Description |
|-------|---------|-------------|
| A. Paper Metadata | 6 | Bibliographic identifiers |
| B. Crop Information | 6 | Crop identity and growth |
| C. Experimental Design | 9 | Trial design parameters |
| D. Environment | 8 | Geospatial and weather |
| E. Soil Properties | 16 | Physical and chemical soil |
| F. Fertilizer Information | 7 | Treatment and application |
| G. Crop Growth Parameters | 22 | Biophysical measurements |
| H. Yield Parameters | 13 | Yield components |
| I. Grain Quality | 14 | Nutritional composition |
| J. ML Target Variables | 5 | Prediction targets |
| K. Engineered Features | 16 | Derived domain features |
| L. Leakage Labels | 1 | Prediction-time availability |
| M. Encoded Variables | 6 | Numeric categorical codes |
| N. ML Predictions | 9 | Model outputs and recommendations |
| **Total** | **138** | |

## Specification Documents

| Document | Description |
|----------|-------------|
| [UAMS_Specification.md](UAMS_Specification.md) | Complete schema specification with column definitions, data types, units, and constraints |
| [Ontology_Registry.md](Ontology_Registry.md) | Variable-to-ontology mappings (AGROVOC, Crop Ontology, Plant Ontology, ENVO, FAO) |
| [Encoding_Scheme.md](Encoding_Scheme.md) | Categorical encoding definitions and code tables |
| [Validation_Rules.md](Validation_Rules.md) | Data quality rules, physical constraints, and validation procedures |
| [Derived_Features.md](Derived_Features.md) | Engineered feature formulas, derivations, and scientific rationale |
| [Data_Contracts.md](Data_Contracts.md) | JSON schemas for agent communication contracts |
| [UAMS_ChangeLog.md](UAMS_ChangeLog.md) | Version history and schema evolution |
| [UAMS_QuickReference.md](UAMS_QuickReference.md) | Condensed one-page reference card |

## Model Compatibility

Datasets conforming to UAMS v1.0 are immediately compatible with:

| Category | Models |
|----------|--------|
| Regression | Multiple Linear, Polynomial, Ridge, Lasso, Elastic Net |
| Tree-based | Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost |
| Distance-based | Support Vector Regression, KNN |
| Deep Learning | Neural Networks, Deep Learning, LSTM, Transformer Models |
| Time Series | ARIMA, Prophet, LSTM-based forecasting |
| Explainable AI | SHAP, LIME, Partial Dependence Plots, Permutation Importance |
| Causal Inference | DoWhy, CausalForest, DoubleML |
| Digital Twins | Physics-informed ML, surrogate modeling |
| Agentic AI | Multi-agent systems, autonomous decision pipelines |

## Usage

### Validating a dataset against UAMS

```python
from agri_ai_agent.config.schema import UAMS_COLUMNS, SCHEMA_GROUPS

def validate_uams_compliance(df):
    missing = [c for c in UAMS_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in UAMS_COLUMNS]
    return {
        "schema_version": "1.0",
        "total_uams_columns": len(UAMS_COLUMNS),
        "present": len(UAMS_COLUMNS) - len(missing),
        "missing": missing,
        "extra_columns": extra,
    }
```

### Programmatic access

```python
from agri_ai_agent.config.schema import UAMS_COLUMNS, VARIANT_MAP, SCHEMA_GROUPS
```

## License

This specification is part of the Agricultural Data Engineering System (ADES).
