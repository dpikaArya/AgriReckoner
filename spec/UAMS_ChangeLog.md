# UAMS Change Log

**All notable changes to the Universal Agricultural Machine Learning Schema**

---

## Documentation reconciliation (2026-07-24)

The specification documents had drifted from the schema defined in code
(`agri_ai_agent/config/schema.py`). This entry records the reconciliation to the
measured source of truth:

- **Total columns: 138** (was documented as 128). Group **B** is 6 columns (adds
  `Growth_Stage`) and group **G** is 22 columns; both were previously mis-counted.
- **14 groups (A–N)**, not 13. Group **N. ML Predictions** (9 cols) was entirely
  missing from the narrative spec and has been added.
- `Ontology_Registry.md` is now generated from `spec/uams_ontology.yaml` via
  `scripts/gen_ontology_docs.py` (138 rows) and must not be hand-edited.
- Group sizes (A–N): 6, 6, 9, 8, 16, 7, 22, 13, 14, 5, 16, 1, 6, 9 = **138**.

No columns were removed or renamed; this is a documentation correction only.

---

## v1.0.0 (2026-07-04)

**Initial Release** — Stable schema definition

### Schema Components (138 columns)

- **A. Paper Metadata** (6 cols): Bibliographic identifiers
- **B. Crop Information** (6 cols): Crop identity, variety, growth duration, stage
- **C. Experimental Design** (9 cols): Trial parameters and spatial layout
- **D. Environment** (8 cols): Geospatial and weather variables
- **E. Soil Properties** (16 cols): Complete soil physico-chemical analysis
- **F. Fertilizer Information** (7 cols): Treatment and application details
- **G. Crop Growth Parameters** (22 cols): Biophysical measurements
- **H. Yield Parameters** (13 cols): Yield and yield-attributing traits
- **I. Grain Quality** (14 cols): Nutritional composition
- **J. ML Target Variables** (5 cols): Prediction targets
- **K. Engineered Features** (16 cols): Domain-derived features
- **L. Leakage Labels** (1 col): Prediction-time availability indicator
- **M. Encoded Variables** (6 cols): Numeric categorical encodings
- **N. ML Predictions** (9 cols): Model outputs and recommendation fields

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
---

## v2.1 external enrichment (Phase 18, 2026-08-11)

- **Appended** 12361 externally validated observations from production acquisition (FAOSTAT bulk, NASA POWER, SoilGrids, CHIRPS, MapSPAM) into `outputs/UAMS_v2.1.parquet`.
- UAMS_v2.parquet is unchanged (checksum `ec79c66eb7aab2057db29050edaa51126241e38d06117570ed177e06d7355bd5`); backup at `UAMS_v2.bak_2026-08-11T12-11-48.parquet`.
- Each promoted row carries provenance_id, source, license, checksum, spatial/temporal match grade, mapping method, and unit status in SoftNotes/Caption.
---

## v2.1 external enrichment (Phase 18, 2026-08-11)

- **Appended** 0 externally validated observations from production acquisition (FAOSTAT bulk, NASA POWER, SoilGrids, CHIRPS, MapSPAM) into `outputs/UAMS_v2.1.parquet`.
- UAMS_v2.parquet is unchanged (checksum `ec79c66eb7aab2057db29050edaa51126241e38d06117570ed177e06d7355bd5`); backup at `UAMS_v2.bak_2026-08-11T12-57-35.parquet`.
- Each promoted row carries provenance_id, source, license, checksum, spatial/temporal match grade, mapping method, and unit status in SoftNotes/Caption.
---

## v2.1 external enrichment (Phase 18, 2026-08-11)

- **Appended** 12361 externally validated observations from production acquisition (FAOSTAT bulk, NASA POWER, SoilGrids, CHIRPS, MapSPAM) into `outputs/UAMS_v2.1.parquet`.
- UAMS_v2.parquet is unchanged (checksum `ec79c66eb7aab2057db29050edaa51126241e38d06117570ed177e06d7355bd5`); backup at `UAMS_v2.bak_ec79c66eb7aab2057db29050edaa51126241e38d06117570ed177e06d7355bd5.parquet`.
- Each promoted row carries provenance_id, source, license, checksum, spatial/temporal match grade, mapping method, and unit status in SoftNotes/Caption.
