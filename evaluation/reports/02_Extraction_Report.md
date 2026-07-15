# Stage 02: Scientific Information Extraction Report
Generated: 2026-07-04 22:30:09

## Summary
- Average Precision: 33.0%
- Average Recall: 33.0%
- Average F1 Score: 33.0%
- Average Hallucination Rate: 0.0%
- Average Extraction Completeness: 33.0%

## Per-Paper Results

### Bell pepper.pdf (Crop: Bell Pepper)
| Metric | Value |
|--------|-------|
| Precision | 21.6% |
| Recall | 21.6% |
| F1 Score | 21.6% |
| Hallucination Rate | 0.0% |
| Extraction Completeness | 21.6% |
| Fields Found | 19/88 |

**Category Breakdown:**
- Study Metadata: 2/6 (33%)
- Crop Extraction: 1/5 (20%)
- Location Extraction: 1/6 (17%)
- Soil Variables: 7/16 (44%)
- Weather Variables: 0/5 (0%)
- Management Practices: 2/7 (29%)
- Growth Parameters: 2/16 (12%)
- Yield Variables: 2/13 (15%)
- Laboratory Measurements: 2/14 (14%)

### Black wheat.pdf (Crop: Black Wheat)
| Metric | Value |
|--------|-------|
| Precision | 47.7% |
| Recall | 47.7% |
| F1 Score | 47.7% |
| Hallucination Rate | 0.0% |
| Extraction Completeness | 47.7% |
| Fields Found | 42/88 |

**Category Breakdown:**
- Study Metadata: 3/6 (50%)
- Crop Extraction: 2/5 (40%)
- Location Extraction: 6/6 (100%)
- Soil Variables: 12/16 (75%)
- Weather Variables: 1/5 (20%)
- Management Practices: 4/7 (57%)
- Growth Parameters: 4/16 (25%)
- Yield Variables: 5/13 (38%)
- Laboratory Measurements: 5/14 (36%)

### Carrot.pdf (Crop: Carrot)
| Metric | Value |
|--------|-------|
| Precision | 23.9% |
| Recall | 23.9% |
| F1 Score | 23.9% |
| Hallucination Rate | 0.0% |
| Extraction Completeness | 23.9% |
| Fields Found | 21/88 |

**Category Breakdown:**
- Study Metadata: 2/6 (33%)
- Crop Extraction: 2/5 (40%)
- Location Extraction: 1/6 (17%)
- Soil Variables: 7/16 (44%)
- Weather Variables: 0/5 (0%)
- Management Practices: 3/7 (43%)
- Growth Parameters: 0/16 (0%)
- Yield Variables: 2/13 (15%)
- Laboratory Measurements: 4/14 (29%)

### Cowpea paper publish.pdf (Crop: Gram)
| Metric | Value |
|--------|-------|
| Precision | 37.5% |
| Recall | 37.5% |
| F1 Score | 37.5% |
| Hallucination Rate | 0.0% |
| Extraction Completeness | 37.5% |
| Fields Found | 33/88 |

**Category Breakdown:**
- Study Metadata: 4/6 (67%)
- Crop Extraction: 3/5 (60%)
- Location Extraction: 1/6 (17%)
- Soil Variables: 9/16 (56%)
- Weather Variables: 0/5 (0%)
- Management Practices: 4/7 (57%)
- Growth Parameters: 3/16 (19%)
- Yield Variables: 3/13 (23%)
- Laboratory Measurements: 6/14 (43%)

### Spinach.pdf (Crop: Spinach)
| Metric | Value |
|--------|-------|
| Precision | 34.1% |
| Recall | 34.1% |
| F1 Score | 34.1% |
| Hallucination Rate | 0.0% |
| Extraction Completeness | 34.1% |
| Fields Found | 30/88 |

**Category Breakdown:**
- Study Metadata: 3/6 (50%)
- Crop Extraction: 1/5 (20%)
- Location Extraction: 2/6 (33%)
- Soil Variables: 12/16 (75%)
- Weather Variables: 0/5 (0%)
- Management Practices: 4/7 (57%)
- Growth Parameters: 3/16 (19%)
- Yield Variables: 0/13 (0%)
- Laboratory Measurements: 5/14 (36%)

## Bottlenecks & Recommendations
1. **Low recall** suggests many paper fields not captured in the UAMS schema
2. **Precision** can be improved with domain-specific NLP extraction
3. **Hallucination** indicates need for stricter field validation
4. Consider using LLM-based extraction for structured paper parsing
5. Add crop-specific extraction templates for each of the 5 crops
