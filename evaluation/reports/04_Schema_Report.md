# Stage 04: Schema Mapping Report
Generated: 2026-07-04 22:30:09
UAMS schema version: 1.0
UAMS columns: 128

## Summary
- Variable mapping accuracy: 97.7%
- Schema coverage: 100.0%
- Schema completeness: 97.7%
- Duplicate removal accuracy: 100.0%
- Total columns in output: 131
- UAMS columns present: 128
- Unmapped columns: 1
- Duplicate columns: 0

## Unmapped Columns (1)
- custom_col

## Duplicate Columns (0)

## Bottlenecks & Recommendations
1. **Unmapped columns** indicate missing entries in VARIANT_MAP or the UAMS schema
2. **Duplicate columns** suggest the ingestion step needs better column deduplication
3. Expand VARIANT_MAP to cover remaining unmapped column name variants
4. Consider automated synonym discovery using WordNet or domain thesauri
