# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — numbers shown to users
- Phosphorus and potassium now mean the element, and an oxide figure is converted instead of
  being stored as though it were the element. The specification declared the oxide basis
  (`Phosphorus … (P₂O₅)`) while the ontology asserted the element, and no conversion existed
  anywhere, so a package-of-practices rate written N:P₂O₅:K₂O entered the phosphorus column
  2.29x too high. `agri_ai_agent/extractors/nutrients.py` derives each factor from atomic
  masses (P₂O₅ ×0.4364, K₂O ×0.8302, plus CaO, MgO, SO₃, Na₂O) and every ingest path applies
  it, whether the oxide is written as a formula (`P2O5`, `P₂O₅`, `P²O⁵`) or in words
  (`potassium oxide`, `oxide basis`), in the column header or in the unit beside the number.
  A term naming a form whose basis is genuinely not fixed — `phosphate`, `potash`, `DAP` — is
  refused rather than assumed, because the basis cannot be recovered once the value is stored;
  a label that simply says nothing is read on the schema's declared basis. Resolves #20.
- Recommendations no longer invent an expected yield gain. `run_aaf_pipeline.py` set the
  expected yield to the observed yield times 1.15 and the gain to 15% of that, and reported it
  as high confidence; every row of the shipped `fertilizer_recommendations.csv` came from that
  arithmetic. An increase is now reported only against a real baseline.
- Confidence now falls as evidence thins. The previous ladder started at 0.75 and only ever
  lowered it, so a row with no soil data read as "High" while a row with partial data read as
  "Medium".
- The economic score no longer saturates. Cost was computed as price-per-kg times dose divided
  by 1000, understating it a thousandfold while revenue was converted correctly, which pinned
  the score at 1.0 for every recommendation and deflated `Risk_Score` by construction.
- Predictions are checked against the registry's declared range before they are written. The
  ranges existed but had no production caller, and a yield near 975,000 kg/ha reached the
  shipped Ready Reckoner.
- Ontology terms are resolved against their authority instead of asserted from generated
  identifiers. Of 106 mappings, 18 were correct: `Nitrogen` pointed at "nitric acid",
  `Yield_per_Hectare` at "Yunnan", `Ash` at "donkeys". Terms are now re-derived by label search
  and kept only on exact agreement; unresolvable ones are recorded as `skos:closeMatch` with
  `verified: false` rather than claiming exactness.
- LLM extraction no longer cuts the paper before its results. The prompt was truncated at
  12,000 characters while the first table in real open-access papers begins at 14,700–82,000,
  so the model had never been shown a results table.
- Dataset normalisation no longer duplicates every row. It merged the input frame with a
  package built from that same frame, so each observation appeared twice on every run.
- Evaluation holds out whole papers. Rows from one trial share a site, season and soil;
  splitting them at random scores a model on a trial it has already seen.
- Nutrient uptake and use-efficiency columns are excluded from features: uptake is
  concentration multiplied by biomass, so it is a function of the yield being predicted.
- `kg/hm2` is read as `kg/ha` (a square hectometre is a hectare), and a yield parsed from prose
  keeps its unit, so `2.5 t ha-1` is no longer stored as `2.5 kg/ha`.
- `requests` is declared, so a clean install can import the external data layer.

### Added
- Open-access corpus extraction (`scripts/extract_oa_corpus.py`): the model chooses which
  table is the experiment and which columns hold treatment, yield and dose; code reads every
  number, and each row keeps its source cell for checking. `scripts/pilot_oa_corpus.py`
  measures how much of the literature is usable.
- Ontology tooling: `scripts/validate_ontology.py` resolves every CURIE against its authority
  and exits non-zero on a mismatch; `scripts/repair_ontology.py` re-derives mappings by label.
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
