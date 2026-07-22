# Dataset Quality Score
Generated: 2026-07-20 17:17:17
Dataset: 43 rows × 138 columns

## Quality Scorecard
| Component | Score | Max | % |
|-----------|-------|-----|---|
| Schema Completeness | 20.0 | 20 | 100.0% |
| Ontology Completeness | 0.0 | 10 | 0.0% |
| Feature Completeness | 15.0 | 15 | 100.0% |
| Missing Value Score | 0.9 | 15 | 6.1% |
| Documentation Score | 10.0 | 10 | 100.0% |
| ML Readiness Score | 9.0 | 15 | 60.0% |
| Scientific Reproducibility | 15.0 | 15 | 100.0% |
| **Overall Quality Score** | **69.9** | **100** | **69.9%** |

## Detailed Breakdown

### Schema Completeness (20.0/20)
- UAMS columns in dataset: 138/138
- Missing schema groups: TODO

### Ontology Completeness (0.0/10)
- Ontology sources mapped: 0/5

### Feature Completeness (15.0/15)
- Engineered features present: 9/9
- Features missing: []

### Missing Values (0.9/15)
- Missing rate: 93.9%
- Total missing: 5572/5934

### Documentation (10.0/10)
- Docs present: 5/5

### ML Readiness (9.0/15)
- Issues: missing=True, categorical=True, leakage_flag=False

### Scientific Reproducibility (15.0/15)
- Paper metadata: True
- Crop info: True
- Location: True
- Soil data: True
- Weather data: True
- Yield data: True

## Recommendations
1. Increase schema coverage by mapping more columns to UAMS
2. Expand ontology coverage with additional sources
3. Implement missing engineered features
4. Reduce missing values through imputation
5. Generate comprehensive documentation
6. Address ML readiness issues for production deployment
