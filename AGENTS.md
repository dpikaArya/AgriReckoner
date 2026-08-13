# AGENTS.md — Agricultural Intelligence Framework (AAIF)

This file tells any agent working in this repository which rules govern its behavior.

## 1. Governing policy

`SCIENTIFIC_AGENT_POLICY.md` is the authoritative universal scientific operating policy (UAASOP, version 0.1.0). It applies to all scientific, engineering, and data science work in this repository. Follow it for every task. This file extends that policy with project specific instructions; it does not replace it and must not contradict its scientific safety, provenance, verification, reproducibility, security, or accountability requirements.

`AGENT_CONFIG.yaml` provides the machine readable policy settings.

## 2. What this project is

AAIF converts unstructured agricultural research PDFs and tabular field data into ML-ready datasets (UAMS v2.0, 296 columns), yield-prediction models, and fuzzy-logic fertilizer recommendations, with per-cell provenance tracking. It is a 23-agent pipeline in 5 phases plus an external-data layer, with a continuous learning loop.

## 3. Project specific instructions

These extend the canonical policy for this domain.

### 3.1 Evidence and provenance conventions

- UAMS records carry per-cell provenance from source paper to schema cell. Preserve this chain; never drop provenance fields during transformation.
- Every retrieved statement must keep its evidence chain intact before any reasoning module consumes it. `config/verification.yaml` defines required chain links and thresholds — do not weaken them.
- Source PDFs, datasets, and external joins are the evidence layer. Distinguish `observed`, `calculated`, `estimated`, `inferred`, and `predicted` values in UAMS output, consistent with policy section 4.
- The generic UAASOP contracts in `contracts/` (provenance, evidence, claim, validation) are the machine readable contract layer; project specific records must remain conformant with the intent of those contracts.
- Do not fabricate papers, citations, DOIs, measurements, or source rows. Every UAMS cell must trace to a real source, a documented transformation, or an explicitly labeled estimate.

### 3.2 Verification

- Run the project's own checks before reporting results: `make lint` (ruff) and `make test` (pytest, `-m 'not live'`).
- Apply the layered verification required by policy section 1.7. This project's verification layer lives in `src/validation/` and is configured in `config/verification.yaml`; use it, do not bypass it.
- Statistical sanity rules (value ranges, SD ordering, CI order, N-observations) in `config/verification.yaml` are the project's statistical checks. Surface, do not hide, violations.
- Consequential model or dataset results require recorded validation before being reported as conclusions.

### 3.3 Schema and data contracts

- The authoritative domain schema is the UAMS specification in `spec/` (`UAMS_Specification.md`, `uams_ontology.yaml`, `ontology_labels.json`). Preserve it.
- Inter-agent communication uses the `AgentContract` structure documented in `spec/Data_Contracts.md`.
- When extending the schema, update the spec and the change log (`spec/UAMS_ChangeLog.md`) with the same rigor the policy requires for provenance.

### 3.4 Reproducibility

- Record code version, configuration, commands, input references, parameters, and random seeds for every run that produces a consequential result. The pipeline records run manifests and model/dataset versions; keep that behavior intact.
- Large data (PDFs, datasets, models, outputs) stays out of git via `.gitignore`. Record references, checksums, and provenance instead.

### 3.5 Security and compute

- Follow policy sections 1.12 and 1.13. Credentials live only in `config/api_keys.env` and environment variables; never hard-code, log, or commit secret values. `.env.*` and secret file patterns are gitignored.
- External data connectors (NASA POWER, SoilGrids, FAOSTAT, CGIAR, etc.) respect per-source limits in `config/connector_limits.yaml`. Do not exceed rate limits, do not spam health checks, and stop runaway loops. HPC and queue rules from policy section 1.15 apply whenever batch or remote compute is used.
- The 23-agent pipeline consumes significant compute; estimate resources before expensive runs (policy section 1.14).

### 3.6 Failure preservation

- Failed extractions, rejected hypotheses, validation failures, retries, and corrections affecting scientific interpretation must be preserved (policy section 1.6). This project's logs, reports, and test results are the record; do not delete or silently rewrite them.

### 3.7 Human review

- Consequential scientific decisions — schema changes, model acceptance, recommendation generation, publication-grade claims, conflicting evidence resolution, and irreversible data changes — require recorded human review (policy section 1.20 and `AGENT_CONFIG.yaml`).

## 4. Default workflow

Follow the default agent workflow from the canonical policy:

```
UNDERSTAND
→ PLAN
→ INSPECT
→ EXECUTE
→ VALIDATE
→ TRACE
→ REVIEW
→ REPORT
```

For this pipeline: UNDERSTAND the research question and UAMS target columns; PLAN the phase(s) involved; INSPECT the relevant spec, config, and existing tests; EXECUTE with least privilege; VALIDATE with the project verification layer; TRACE each output back through its evidence chain; REVIEW consequential results with a human; REPORT with explicit confidence and uncertainty labels.

## 5. Reference files

| File / path | Role |
| --- | --- |
| `SCIENTIFIC_AGENT_POLICY.md` | Authoritative universal policy (UAASOP 0.1.0) |
| `AGENT_CONFIG.yaml` | Machine readable policy settings |
| `contracts/` | Generic UAASOP machine readable contracts (provenance, evidence, claim, validation) |
| `spec/` | UAMS domain schema and data contract documentation |
| `config/verification.yaml` | Evidence chain verification configuration |
| `src/provenance/` | Provenance and lineage tracking implementation |
| `src/validation/` | Verification and validation implementation |
