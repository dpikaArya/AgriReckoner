# Enhanced Extraction Report

## Summary
| Metric | Value |
|--------|-------|
| Papers processed | 16 |
| Treatment rows extracted | 121 |
| Active features | 18 / 139 |
| Non-null yield observations | 17 |
| ML training threshold | 100 required → NOT MET |

## Per-Paper Extraction
| Paper | Rows | Yield |
|-------|------|-------|
| Black wheat.pdf | 30 | 10 (Yield_per_Plot) |
| Bell pepper.pdf | 24 | 7 (Yield_per_Plot) |
| Spinach.pdf | 16 | 0 |
| Cowpea paper publish.pdf | 12 | 1 (Yield_per_Plot) |
| fpls-16-1521113.pdf | 12 | 0 |
| fpls-12-748523.pdf | 10 | 0 |
| Carrot.pdf | 8 | 0 |
| Other (9 PDFs, metadata only) | 9 | 0 |

## Yield Observations
| Variable | Non-null | % |
|----------|----------|---|
| Yield_per_Plot | 17 | 14.0% |
| Yield_per_Hectare | 0 | 0.0% |
| Yield_per_Acre | 0 | 0.0% |
| Biomass_Yield | 0 | 0.0% |
| Harvest_Index | 0 | 0.0% |

## Missing Value % (Target Variables)
| Variable | Missing % |
|----------|-----------|
| Yield_per_Hectare | 100.0% |
| Yield_per_Plot | 86.0% |
| Yield_per_Acre | 100.0% |
| Biomass_Yield | 100.0% |
| Harvest_Index | 100.0% |
| Plant_Height_cm | 100.0% |
| SPAD | 93.4% |
| Soil_pH | 100.0% |
| Nitrogen | 100.0% |
| Phosphorus | 100.0% |
| Potassium | 100.0% |
| Organic_Carbon | 100.0% |
| Protein | 100.0% |

## Decision
**ML models NOT retrained.** Only 17 yield observations found (threshold: 100).
Additional extraction tools (Camelot, Tabula) or manual table annotation needed.
