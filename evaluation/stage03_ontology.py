"""
Stage 03: Ontology Mapping Evaluation
Evaluates AGROVOC, Crop Ontology, Plant Ontology, Environment Ontology, Unit Ontology mapping.
"""

import time

from evaluation.utils import OUTPUT_DIR, load_dataframe, write_report


def evaluate_ontology():
    onto_path = OUTPUT_DIR / "Ontology_Mapping.csv"
    if not onto_path.exists():
        return _empty_report("Ontology_Mapping.csv not found")

    df = load_dataframe(str(onto_path))
    if df is None or len(df) == 0:
        return _empty_report("Empty ontology mapping")

    total_entries = len(df)

    required_cols = [
        "AGROVOC",
        "Crop Ontology",
        "Plant Ontology",
        "Environment Ontology",
        "Unit Ontology",
    ]
    coverage_by_source = {}
    for col in required_cols:
        if col in df.columns:
            present = df[col].notna() & (df[col].astype(str).str.strip() != "")
            coverage_by_source[col] = {
                "mapped": int(present.sum()),
                "unmapped": int((~present).sum()),
                "coverage": float(present.mean()),
            }
        else:
            coverage_by_source[col] = {
                "mapped": 0,
                "unmapped": total_entries,
                "coverage": 0.0,
            }

    avg_coverage = sum(c["coverage"] for c in coverage_by_source.values()) / len(coverage_by_source)

    missing_terms = []
    for col in required_cols:
        if col in df.columns:
            unmapped = df[~df[col].notna() | (df[col].astype(str).str.strip() == "")]
            for _, row in unmapped.iterrows():
                var = (row.get("variable", row.get("Column", row.get("Field", f"row_{_}"))),)
                missing_terms.append(f"{var} -> {col}")

    report = f"""# Stage 03: Ontology Mapping Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total mapped entries: {total_entries}

## Summary
- Average ontology coverage: {avg_coverage:.1%}
- Missing ontology terms: {len(missing_terms)}

## Coverage by Ontology Source
"""
    for source, cov in coverage_by_source.items():
        report += f"- {source}: {cov['coverage']:.1%} ({cov['mapped']}/{cov['mapped'] + cov['unmapped']} mapped)\n"

    report += f"""
## Missing Ontology Terms: {len(missing_terms)}
"""
    for term in missing_terms[:50]:
        report += f"- {term}\n"
    if len(missing_terms) > 50:
        report += f"- ... and {len(missing_terms) - 50} more\n"

    report += """
## Bottlenecks & Recommendations
1. **Low coverage** may indicate missing ontology mappings for domain-specific terms
2. Consider expanding the AGROVOC vocabulary for local crop varieties
3. Add automatic ontology lookup via REST APIs (AGROVOC, FAO)
4. Implement fuzzy matching for near-miss ontology terms
5. Create crop-specific ontology subsets for faster mapping
"""
    path = write_report("03_Ontology_Report.md", report)
    return coverage_by_source, str(path)


def _empty_report(reason: str):
    report = f"""# Stage 03: Ontology Mapping Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
**Error:** {reason}

No ontology mapping data available to evaluate.
"""
    path = write_report("03_Ontology_Report.md", report)
    return {}, str(path)
