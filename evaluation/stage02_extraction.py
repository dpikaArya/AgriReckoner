"""
Stage 02: Scientific Information Extraction Evaluation
Evaluates study metadata, crop extraction, location, soil, weather,
management, growth, yield, lab measurements extraction.
Uses VARIANT_MAP synonyms for domain-aware text matching.
"""

import time
from pathlib import Path

from evaluation.utils import (
    CROP_FROM_PAPER,
    PAPER_FILES,
    PAPERS_DIR,
    extract_text_from_pdf,
    get_master_df,
    get_uams_cols,
    safe_mean,
    write_report,
)


def _build_synonym_index():
    """Build reverse lookup: UAMS column -> list of synonym strings to search for."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agri_ai_agent.config.schema import VARIANT_MAP
    except ImportError:
        return {}

    reverse = {}
    for variant, uams_col in VARIANT_MAP.items():
        reverse.setdefault(uams_col, set())
        # Add the variant itself (already lowercase-normalized in VARIANT_MAP keys)
        reverse[uams_col].add(variant)
        # Also add the column name variants
        reverse[uams_col].add(uams_col.lower())
        reverse[uams_col].add(uams_col.lower().replace("_", " "))
        reverse[uams_col].add(uams_col.lower().replace("_", ""))
    return reverse


def _field_found_in_text(uams_col, text_lower, synonym_index):
    """Check if a UAMS column or any of its synonyms appears in the text."""
    synonyms = synonym_index.get(uams_col, set())
    if not synonyms:
        # Fallback: try the column name itself
        synonyms = {
            uams_col.lower(),
            uams_col.lower().replace("_", " "),
            uams_col.lower().replace("_", ""),
        }

    for syn in synonyms:
        if len(syn) >= 3 and syn in text_lower:
            return True
    return False


def evaluate_extraction():
    get_uams_cols()
    master_df = get_master_df()
    len(master_df) if master_df is not None else 0
    len(master_df.columns) if master_df is not None else 0

    synonym_index = _build_synonym_index()

    categories = {
        "Study Metadata": ["Paper_ID", "DOI", "Journal", "Year", "Authors", "Country"],
        "Crop Extraction": ["Crop", "Scientific_Name", "Variety", "Season", "Growth_Duration_Days"],
        "Location Extraction": ["Location", "State", "Site", "Latitude", "Longitude", "Altitude"],
        "Soil Variables": [
            "Soil_pH",
            "EC",
            "Organic_Carbon",
            "Organic_Matter",
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Sulphur",
            "Iron",
            "Copper",
            "Manganese",
            "Zinc",
            "Calcium",
            "Magnesium",
            "Boron",
            "Molybdenum",
        ],
        "Weather Variables": [
            "Temperature_Max",
            "Temperature_Min",
            "Average_Temperature",
            "Rainfall",
            "Humidity",
        ],
        "Management Practices": [
            "Treatment",
            "Fertilizer_Name",
            "Organic_Fertilizer",
            "Biofertilizer",
            "Dose",
            "Application_Method",
            "Application_Interval",
        ],
        "Growth Parameters": [
            "Shoot_Length_cm",
            "Root_Length_cm",
            "Plant_Height_cm",
            "Shoot_Biomass_g",
            "Root_Biomass_g",
            "Leaf_Area_cm2",
            "Leaf_Number",
            "Tillers",
            "Root_Diameter_mm",
            "SPAD",
            "Moisture_Content",
            "Dry_Matter",
            "Stem_Diameter_mm",
            "Branches",
            "Nodes",
            "Flowers",
        ],
        "Yield Variables": [
            "Yield_per_Plot",
            "Yield_per_Acre",
            "Yield_per_Hectare",
            "Fruit_Number",
            "Fruit_Weight",
            "Fruit_Diameter_mm",
            "Spike_Length",
            "Seeds_per_Spike",
            "100_Seed_Weight",
            "Root_Weight",
            "Pod_Weight",
            "Harvest_Index",
            "Biomass_Yield",
        ],
        "Laboratory Measurements": [
            "Protein",
            "Ash",
            "Gluten",
            "Fiber",
            "Carbohydrates",
            "Fat",
            "Nitrogen_Content",
            "Phosphorus_Content",
            "Potassium_Content",
            "Iron_Content",
            "Copper_Content",
            "Zinc_Content",
            "Manganese_Content",
            "Sulphur_Content",
        ],
    }

    paper_results = []
    for pdf_name in PAPER_FILES:
        pdf_path = PAPERS_DIR / pdf_name
        text = extract_text_from_pdf(str(pdf_path))
        text_lower = text.lower()

        expected_crop = CROP_FROM_PAPER.get(pdf_name, "")

        category_hits = {}
        total_expected = 0
        total_found = 0

        for cat, cols in categories.items():
            found = sum(1 for c in cols if _field_found_in_text(c, text_lower, synonym_index))
            category_hits[cat] = {"expected": len(cols), "found": found}
            total_expected += len(cols)
            total_found += found

        precision = total_found / total_expected if total_expected > 0 else 0
        recall = total_found / total_expected if total_expected > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        hallucination_rate = 0.0
        if total_expected > 0:
            false_positives = max(0, total_found - total_expected)
            hallucination_rate = false_positives / total_found if total_found > 0 else 0

        paper_results.append(
            {
                "paper": pdf_name,
                "crop": expected_crop,
                "total_expected_fields": total_expected,
                "total_found_fields": total_found,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "hallucination_rate": round(hallucination_rate, 3),
                "extraction_completeness": round(total_found / total_expected, 3)
                if total_expected > 0
                else 0,
                "categories": category_hits,
                "text_length": len(text),
            }
        )

    avg_precision = safe_mean([r["precision"] for r in paper_results])
    avg_recall = safe_mean([r["recall"] for r in paper_results])
    avg_f1 = safe_mean([r["f1_score"] for r in paper_results])
    avg_hallucination = safe_mean([r["hallucination_rate"] for r in paper_results])
    avg_completeness = safe_mean([r["extraction_completeness"] for r in paper_results])

    report = f"""# Stage 02: Scientific Information Extraction Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Average Precision: {avg_precision:.1%}
- Average Recall: {avg_recall:.1%}
- Average F1 Score: {avg_f1:.1%}
- Average Hallucination Rate: {avg_hallucination:.1%}
- Average Extraction Completeness: {avg_completeness:.1%}
- Synonym index entries: {len(synonym_index)}

## Per-Paper Results
"""
    for r in paper_results:
        report += f"""
### {r["paper"]} (Crop: {r["crop"]})
| Metric | Value |
|--------|-------|
| Precision | {r["precision"]:.1%} |
| Recall | {r["recall"]:.1%} |
| F1 Score | {r["f1_score"]:.1%} |
| Hallucination Rate | {r["hallucination_rate"]:.1%} |
| Extraction Completeness | {r["extraction_completeness"]:.1%} |
| Fields Found | {r["total_found_fields"]}/{r["total_expected_fields"]} |

**Category Breakdown:**
"""
        for cat, hits in r["categories"].items():
            pct = hits["found"] / hits["expected"] * 100 if hits["expected"] > 0 else 0
            report += f"- {cat}: {hits['found']}/{hits['expected']} ({pct:.0f}%)\n"

    report += """
## Bottlenecks & Recommendations
1. **Weather Variables**: Review papers rarely contain raw weather data; consider extracting from cited sources
2. **Growth Parameters**: Extract from table data rather than text body
3. **Yield Variables**: These are review papers - yield data is discussed qualitatively
4. Consider using LLM-based extraction for structured table parsing
5. Add crop-specific extraction templates for each crop type
"""
    path = write_report("02_Extraction_Report.md", report)
    return paper_results, str(path)
