# Reproducibility Report
Generated: 2026-07-07T22:01:32.427492

## Reproducibility Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Source papers identified | ✅ | DOI, Paper_ID columns |
| Extraction method documented | ✅ | Extraction_Report.md |
| Evidence validated | ✅ | Evidence_Validation_Report.md |
| Provenance tracked | ✅ | Evidence_Traceability.csv |
| Ontology mappings recorded | ✅ | Ontology_Mapping.csv |
| Unit conversions recorded | ✅ | Unit_Conversion_Report.md |
| Pipeline versioned | ✅ | Pipeline_Provenance.json |
| Data in open format | ✅ | CSV, Parquet, XLSX |
| Feature engineering documented | ✅ | Feature_Engineering_Report.md |
| Quality checks applied | ✅ | Quality_Report.md |

## Reproduction Steps
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run pipeline: `python -m ades run input.xlsx`
4. Verify outputs match documented provenance