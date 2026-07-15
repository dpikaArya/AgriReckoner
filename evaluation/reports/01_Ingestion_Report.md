# Stage 01: Research Paper Ingestion Report
Generated: 2026-07-04 22:30:04
Total papers: 5

## Summary
- PDF parsing success rate: 100.0%
- Average text extraction completeness: 100.0%
- Average table extraction accuracy: 0.0%
- Average OCR quality: 100.0%
- Total processing time: 4.52s
- Memory usage: 59.1 MB → 69.7 MB

## Per-Paper Results

### Bell pepper.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 36706 |
| Processing time | 0.905s |
| Tables found | 0 |
| Figures detected | 8 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### Black wheat.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 43804 |
| Processing time | 0.616s |
| Tables found | 0 |
| Figures detected | 15 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### Carrot.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 35039 |
| Processing time | 0.905s |
| Tables found | 0 |
| Figures detected | 6 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### Cowpea paper publish.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 39300 |
| Processing time | 1.060s |
| Tables found | 0 |
| Figures detected | 14 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### Spinach.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 32581 |
| Processing time | 0.989s |
| Tables found | 0 |
| Figures detected | 2 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

## Bottlenecks & Recommendations
1. **PDF parsing**: Ensure all PDFs are text-based (not scanned images)
2. **Table extraction**: Consider Camelot or Tabula for complex tables
3. **Figure detection**: Implement figure caption detection for completeness
4. **Reference parsing**: Add regex-based reference extraction
5. **OCR fallback**: Integrate Tesseract for scanned PDFs
