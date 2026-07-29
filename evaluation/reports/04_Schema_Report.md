# Stage 04: Schema Mapping Report
Generated: 2026-07-28 15:48:24
UAMS schema version: 2.0
UAMS columns: 296 (Core A-N: 193, Extended O-Z: 103)

## Summary
- Variable mapping accuracy: 66.7%
- Overall schema coverage: 46.3%
- Core schema coverage (A-N): 71.0% (137/193)
- Extended schema coverage (O-Z): 0.0% (0/103)
- Schema completeness: 99.3%
- Duplicate removal accuracy: 100.0%
- Total columns in output: 138
- UAMS columns present: 137
- Unmapped columns: 1
- Duplicate columns: 0

## Core Schema Groups (A-N)
- A. Paper Metadata: 6/8 (75%)
- B. Crop Information: 6/9 (67%)
- C. Experimental Design: 9/14 (64%)
- D. Environment: 8/12 (67%)
- E. Soil Properties: 16/26 (62%)
- F. Fertilizer Information: 7/12 (58%)
- G. Crop Growth Parameters: 22/27 (81%)
- H. Yield Parameters: 13/27 (48%)
- I. Grain Quality: 14/21 (67%)
- J. ML Target Variables: 5/5 (100%)
- K. Engineered Features: 15/16 (94%)
- L. Leakage Labels: 1/1 (100%)
- M. Encoded Variables: 6/6 (100%)
- N. ML Predictions: 9/9 (100%)

## Unmapped Columns (1)
- custom_col

## Duplicate Columns (0)

## Bottlenecks & Recommendations
1. **Extended schema (O-Z)**: Data ADES extensions require specialized source data (soil enzymes, microbial communities, remote sensing, nematodes)
2. **Core schema**: Focus on populating remaining core columns from available paper data
3. Expand VARIANT_MAP to cover remaining unmapped column name variants
4. Consider automated synonym discovery using WordNet or domain thesauri
