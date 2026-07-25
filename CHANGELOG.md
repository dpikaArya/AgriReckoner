# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Machine-readable ontology registry (`spec/uams_ontology.yaml`) with a loader/validator
  (`agri_ai_agent/ontology/`), wired into the ontology agent so mappings carry real term IRIs
  (AGROVOC/ENVO/CO/PO) and canonical units. `spec/Ontology_Registry.md` is generated from it.
- Leakage guard (`agri_ai_agent/ml/leakage.py`, name/acronym + correlation safety-net) and
  honest cross-validated small-n evaluation (`agri_ai_agent/ml/evaluation.py`): leave-one-out /
  repeated k-fold with imputation inside folds, no metric below n=8, advisory below n=30.
- OpenAI LLM extraction (`agri_ai_agent/extractors/`, `agents/llm_extraction_agent.py`) with a
  grounding pass (source-span echo + unit conversion + registry range check) and per-value
  provenance. Usable via `agriai extract --papers <dir>`. Validated live on open-access PDFs.
- Unit-aware grounding (`agri_ai_agent/extractors/units.py`): converts compatible units and
  rejects incompatible ones (e.g. a soil `g/kg` concentration for a `kg/ha` rate column).
- Reproducibility: frozen baseline checksums (git tag `legacy-monolith-v1`) with a guard test,
  and a per-run manifest (version, git SHA, config).
- CI (`.github/workflows/ci.yml`), `Makefile`, `.pre-commit-config.yaml`, `tests/conftest.py`,
  `tests/test_docs_claims.py` (asserts README counts match code), and this changelog.

### Fixed
- The agent orchestrator now runs end-to-end (previously it dropped the input dataframe and the
  first agents crashed); reordered pipeline steps; training resolves a real measured target.
- `base_agent._resolve_duplicate_columns` no longer multiplies duplicate columns.
- Target leakage: outcome-derived engineered features and `Predicted_Yield` no longer enter the
  feature matrix; `model_selection` reports nested-CV generalization scores.

### Changed
- Documentation reconciled to the measured source of truth: 138 columns / 14 groups (A–N);
  17 agents wired (ProvenanceAgent available but not wired; optional LLM extraction agent);
  221 fuzzy rules; Apache-2.0. The previously published model R² table is documented as
  target-leakage-inflated, not validated skill. `run_aaf_pipeline.py` marked frozen-legacy.

## [2.0.0]

- Baseline release of the Agricultural Intelligence Framework (AAIF).
