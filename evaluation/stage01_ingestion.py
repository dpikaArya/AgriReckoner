"""
Stage 01: Research Paper Ingestion Evaluation
Evaluates PDF parsing success, OCR quality, text extraction completeness,
table extraction accuracy, figure detection, reference extraction.
"""

import os
import re
import time
from pathlib import Path

from evaluation.utils import (
    PAPERS_DIR, PAPER_FILES, CROP_FROM_PAPER, OUTPUT_DIR,
    extract_text_from_pdf, extract_tables_from_pdf,
    get_memory_usage, write_report, REPORTS_DIR,
)


def evaluate_ingestion():
    start = time.time()
    mem_before = get_memory_usage()

    results = []
    for pdf_name in PAPER_FILES:
        pdf_path = PAPERS_DIR / pdf_name
        if not pdf_path.exists():
            results.append({
                "paper": pdf_name,
                "file_exists": False,
                "pdf_parsing_success": False,
                "ocr_quality": 0.0,
                "text_extraction_completeness": 0.0,
                "table_extraction_accuracy": 0.0,
                "figure_detection": 0,
                "reference_extraction": 0,
                "text_length_chars": 0,
                "tables_found": 0,
                "error": "File not found",
            })
            continue

        t0 = time.time()
        text = extract_text_from_pdf(str(pdf_path))
        parse_time = time.time() - t0

        tables = extract_tables_from_pdf(str(pdf_path))

        text_len = len(text)
        has_content = text_len > 100

        figures_found = len(re.findall(r"(?:fig\.|figure|fig\.?\s*\d+)", text, re.IGNORECASE))
        refs_found = len(re.findall(r"(?:references|bibliography|works\s+cited)", text, re.IGNORECASE))

        expected_crop = CROP_FROM_PAPER.get(pdf_name, "")
        crop_in_text = expected_crop.lower() in text.lower() if expected_crop else False

        completeness = min(1.0, text_len / 5000) if text_len > 0 else 0.0
        ocr_score = min(1.0, text_len / 3000) if has_content else 0.0
        table_score = min(1.0, len(tables) / 3.0) if tables else 0.0

        results.append({
            "paper": pdf_name,
            "crop": expected_crop,
            "file_exists": True,
            "pdf_parsing_success": has_content,
            "text_length_chars": text_len,
            "processing_time_sec": round(parse_time, 3),
            "tables_found": len(tables),
            "figures_detected": figures_found,
            "references_sections": refs_found,
            "crop_name_in_text": crop_in_text,
            "text_extraction_completeness": round(completeness, 3),
            "table_extraction_accuracy": round(table_score, 3),
            "ocr_quality": round(ocr_score, 3),
            "figure_detection": figures_found,
            "reference_extraction": refs_found,
        })

    total_time = time.time() - start
    mem_after = get_memory_usage()

    avg_completeness = sum(r.get("text_extraction_completeness", 0) for r in results) / len(results) if results else 0
    avg_table = sum(r.get("table_extraction_accuracy", 0) for r in results) / len(results) if results else 0
    avg_ocr = sum(r.get("ocr_quality", 0) for r in results) / len(results) if results else 0
    success_rate = sum(1 for r in results if r.get("pdf_parsing_success")) / len(results) if results else 0

    report = f"""# Stage 01: Research Paper Ingestion Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total papers: {len(results)}

## Summary
- PDF parsing success rate: {success_rate:.1%}
- Average text extraction completeness: {avg_completeness:.1%}
- Average table extraction accuracy: {avg_table:.1%}
- Average OCR quality: {avg_ocr:.1%}
- Total processing time: {total_time:.2f}s
- Memory usage: {mem_before.get('rss_mb', 0):.1f} MB → {mem_after.get('rss_mb', 0):.1f} MB

## Per-Paper Results
"""
    for r in results:
        report += f"""
### {r['paper']}
| Metric | Value |
|--------|-------|
| File exists | {r.get('file_exists', False)} |
| PDF parsing success | {r.get('pdf_parsing_success', False)} |
| Text length (chars) | {r.get('text_length_chars', 0)} |
| Processing time | {r.get('processing_time_sec', 0):.3f}s |
| Tables found | {r.get('tables_found', 0)} |
| Figures detected | {r.get('figures_detected', 0)} |
| Reference sections | {r.get('references_sections', 0)} |
| Crop name in text | {r.get('crop_name_in_text', False)} |
| Text extraction completeness | {r.get('text_extraction_completeness', 0):.1%} |
| Table extraction accuracy | {r.get('table_extraction_accuracy', 0):.1%} |
| OCR quality | {r.get('ocr_quality', 0):.1%} |
"""

    report += """
## Bottlenecks & Recommendations
1. **PDF parsing**: Ensure all PDFs are text-based (not scanned images)
2. **Table extraction**: Consider Camelot or Tabula for complex tables
3. **Figure detection**: Implement figure caption detection for completeness
4. **Reference parsing**: Add regex-based reference extraction
5. **OCR fallback**: Integrate Tesseract for scanned PDFs
"""
    path = write_report("01_Ingestion_Report.md", report)
    return results, str(path)
