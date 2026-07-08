# UAMS Change Log

**All notable changes to the Universal Agricultural Machine Learning Schema**

---

## v1.0.0 (2026-07-04)

**Initial Release** — Stable schema definition

### Schema Components (128 columns)

- **A. Paper Metadata** (6 cols): Bibliographic identifiers
- **B. Crop Information** (5 cols): Crop identity, variety, growth duration
- **C. Experimental Design** (9 cols): Trial parameters and spatial layout
- **D. Environment** (8 cols): Geospatial and weather variables
- **E. Soil Properties** (16 cols): Complete soil physico-chemical analysis
- **F. Fertilizer Information** (7 cols): Treatment and application details
- **G. Crop Growth Parameters** (20 cols): Biophysical measurements
- **H. Yield Parameters** (13 cols): Yield and yield-attributing traits
- **I. Grain Quality** (14 cols): Nutritional composition
- **J. ML Target Variables** (5 cols): Prediction targets
- **K. Engineered Features** (16 cols): Domain-derived features
- **L. Leakage Labels** (1 col): Prediction-time availability indicator
- **M. Encoded Variables** (6 cols): Numeric categorical encodings

### Initial Deliverables

- Full schema specification (`UAMS_Specification.md`)
- Ontology registry with AGROVOC, Crop Ontology, Plant Ontology, ENVO, FAO mappings
- Encoding scheme for categorical variables
- Validation rules with 30+ constraints
- 16 derived feature formulas with scientific rationale
- JSON data contracts for agent communication
- Condensed quick reference card

### Pipeline Implementation

- 12 specialized autonomous agents
- 1 Orchestrator with checkpoint recovery and incremental execution
- Plugin-based architecture with JSON contracts
- Backward-compatible wrapper for legacy callers
- 23 automated tests (20 unit + 3 integration)

---

## Upcoming (Planned for v1.1.0)

- Support for `NDVI`, `EVI`, `LAI` remote sensing indices
- Additional soil micronutrients: Chlorine, Nickel, Cobalt, Selenium
- Time-series specific columns: `Date`, `Day_of_Year`, `Growth_Stage`
- Multilingual crop names (Hindi, Spanish, French, Arabic)
- Automated feature selection recommendations per model type
- Privacy-preserving aggregation flags
- Schema extension mechanism for custom experiment-specific variables

---

## Versioning Convention

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

| Increment | Meaning | Example |
|-----------|---------|---------|
| MAJOR | Breaking changes (column removal, type changes) | 1.0.0 → 2.0.0 |
| MINOR | Backward-compatible additions | 1.0.0 → 1.1.0 |
| PATCH | Clarifications, corrections, documentation | 1.0.0 → 1.0.1 |

### Deprecation Policy

- Deprecated columns will remain in the schema for at least one MINOR version
- Removal announced in the change log one version in advance
- Migration scripts provided for breaking changes

---

## How to Propose Changes

1. Open an issue describing the proposed change
2. Include rationale, impacted columns, and migration path
3. For new columns: specify name, type, units, and ontology mapping
4. Tag with `schema-change` label
