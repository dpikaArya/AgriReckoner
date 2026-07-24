# Info-flow: LLM extraction

How every piece of information reaches the model and how its output is trusted.

## Into the prompt

| Variable | Source | Where |
|---|---|---|
| System instructions | `agri_ai_agent/prompts/extraction_system.txt` (versioned) | `LLMExtractor._system` |
| Target column list | `EXTRACTION_COLUMNS`, derived from `spec/uams_ontology.yaml` (columns with a validated range) | `extractors/fields.py` → `_build_user_prompt` |
| Paper text | caller-supplied `papers[paper_id]`, truncated to 12 000 chars | `_build_user_prompt` |
| Output schema | generated from `EXTRACTION_COLUMNS` (enum) | `build_extraction_schema` |
| model / temperature | `AgriAISettings.LLM_MODEL` / `LLM_TEMPERATURE` (env `AGRI_LLM_*`), temperature 0 | `make_openai_completer` |

Nothing the model needs is dropped: the column list, the units it must report as-is, and
the anti-fabrication rules are all in the prompt. The registry is the single source of the
column set, so prompt and validation cannot drift.

## Out of the model → trust

The model returns values **unverified**. Trust is established by code, not by the model:

1. **Structured output** — `response_format=json_schema, strict=True` forces the shape.
2. **Grounding** (`extractors/grounding.py`): a value is accepted only if
   - its number appears literally in the cited `source_quote` (numeric-echo check), and
   - it falls within the registry's validated range for that column.
3. Rejected values are **flagged with a reason**, kept in `LLM_Extraction_Provenance.csv`
   for human review — never silently dropped. Only `status == "reported"` values enter the
   UAMS row.

## Explicit non-responsibilities of the model

- The model must NOT compute or convert units (kept as reported; conversion is deterministic
  downstream). 
- The model must NOT emit ontology IRIs/CURIEs (those come only from the registry).
- If a value is not explicitly stated, the model returns `null` (missing is correct;
  fabricated is a serious error).

## Offline / no-key behaviour

`LLMExtractionAgent` is non-load-bearing: with no `OPENAI_API_KEY` (and no injected
extractor) it logs a warning and returns the input unchanged, so the pipeline and CI run
fully offline. Tests inject a fake completer; the live path is exercised by
`scripts/smoke_llm_extract.py`.
