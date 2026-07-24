# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Continuous integration workflow (`.github/workflows/ci.yml`): lint, a
  Python 3.10/3.12 test matrix, and a wheel-smoke build.
- Developer tooling: `Makefile`, `.pre-commit-config.yaml`, and shared pytest
  fixtures in `tests/conftest.py`.
- This changelog.

### Changed
- Quality-uplift refactor in progress: aligning the codebase with the project
  engineering standards (naming, function size, docstrings, tests, and
  linting). No behavioural changes intended during this phase.
- Documentation reconciled to the measured source of truth: UAMS is 138 columns
  across 14 groups (A–N); 17 agents are wired into the default pipeline, with the
  ProvenanceAgent available but not wired and an optional LLM extraction agent;
  fuzzy rule set is 221 rules; license is Apache-2.0. The previously published
  model R² table is documented as target-leakage-inflated, not validated skill.
- `spec/Ontology_Registry.md` is now generated from `spec/uams_ontology.yaml`.

### Added
- Leakage guard (`agri_ai_agent/ml/leakage.py`) and honest cross-validated
  small-n evaluation (`agri_ai_agent/ml/evaluation.py`): leave-one-out / repeated
  k-fold, no metric reported below n=8, advisory caveat below n=30.
- Optional OpenAI-based extraction agent (`pip install -e ".[llm]"`).
- `tests/test_docs_claims.py`: asserts README headline counts match the code.

## [2.0.0]

- Baseline release of the Agricultural Intelligence Framework (AAIF).
