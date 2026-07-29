# Stage 01: Research Paper Ingestion Report
Generated: 2026-07-28 15:47:53
Total papers: 10

## Summary
- PDF parsing success rate: 100.0%
- Average text extraction completeness: 100.0%
- Average table extraction accuracy: 56.7%
- Average OCR quality: 100.0%
- Total processing time: 58.93s
- Memory usage: 88.5 MB → 479.6 MB

## Per-Paper Results

### 28july.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 69140 |
| Processing time | 3.009s |
| Tables found | 14 |
| Figures detected | 26 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 100.0% |
| OCR quality | 100.0% |

### 28july1.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 69317 |
| Processing time | 3.341s |
| Tables found | 0 |
| Figures detected | 23 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### 4449-7.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 45355 |
| Processing time | 2.411s |
| Tables found | 11 |
| Figures detected | 11 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 100.0% |
| OCR quality | 100.0% |

### Review_Paper_on_Effect_of_Micronutrients.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 48557 |
| Processing time | 2.647s |
| Tables found | 11 |
| Figures detected | 3 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 100.0% |
| OCR quality | 100.0% |

### The Effect of Micronutrients in Ensuring Efficient Use of Macronu.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 22956 |
| Processing time | 1.165s |
| Tables found | 9 |
| Figures detected | 2 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 100.0% |
| OCR quality | 100.0% |

### bouis-et-al-2000-the-consultative-group-on-international-agricultural-research-(cgiar)-micronutrients-project.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 38578 |
| Processing time | 1.542s |
| Tables found | 1 |
| Figures detected | 0 |
| Reference sections | 1 |
| Crop name in text | False |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 33.3% |
| OCR quality | 100.0% |

### erh064.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 60765 |
| Processing time | 2.179s |
| Tables found | 0 |
| Figures detected | 10 |
| Reference sections | 3 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### s10705-018-09968-7.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 46223 |
| Processing time | 2.471s |
| Tables found | 0 |
| Figures detected | 12 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 0.0% |
| OCR quality | 100.0% |

### s13593-017-0431-0.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 54848 |
| Processing time | 5.319s |
| Tables found | 1 |
| Figures detected | 23 |
| Reference sections | 3 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 33.3% |
| OCR quality | 100.0% |

### shukla-ijfapril2018.pdf
| Metric | Value |
|--------|-------|
| File exists | True |
| PDF parsing success | True |
| Text length (chars) | 99088 |
| Processing time | 6.706s |
| Tables found | 10 |
| Figures detected | 27 |
| Reference sections | 1 |
| Crop name in text | True |
| Text extraction completeness | 100.0% |
| Table extraction accuracy | 100.0% |
| OCR quality | 100.0% |

## Bottlenecks & Recommendations
1. **PDF parsing**: Ensure all PDFs are text-based (not scanned images)
2. **Table extraction**: Consider Camelot or Tabula for complex tables
3. **Figure detection**: Implement figure caption detection for completeness
4. **Reference parsing**: Add regex-based reference extraction
5. **OCR fallback**: Integrate Tesseract for scanned PDFs
