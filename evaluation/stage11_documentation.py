"""
Stage 11: Documentation Evaluation
Evaluates README completeness, Dataset Card, Feature Dictionary,
Variable Mapping, Ontology Mapping, Quality Reports, Pipeline Logs.
"""

import time

from evaluation.utils import BASE_DIR, write_report

DOCUMENTATION_FILES = {
    "README.md": "Project README",
    "outputs/Feature_Dictionary.csv": "Feature Dictionary",
    "outputs/Variable_Mapping.csv": "Variable Mapping",
    "outputs/Ontology_Mapping.csv": "Ontology Mapping",
    "outputs/Quality_Report.md": "Quality Report",
    "outputs/Data_Validation_Report.md": "Data Validation Report",
    "outputs/Missing_Data_Report.md": "Missing Data Report",
    "outputs/Model_Readiness_Report.md": "Model Readiness Report",
    "outputs/Unit_Conversion_Report.md": "Unit Conversion Report",
    "outputs/Feature_Engineering_Report.md": "Feature Engineering Report",
    "outputs/Leakage_Report.csv": "Leakage Report",
    "outputs/Pipeline_Log.md": "Pipeline Log",
    "outputs/Pipeline_Provenance.json": "Pipeline Provenance",
    "outputs/Schema_Metadata.json": "Schema Metadata",
    "spec/UAMS_Specification.md": "UAMS Specification",
    "spec/Data_Contracts.md": "Data Contracts",
    "spec/Derived_Features.md": "Derived Features",
    "spec/Encoding_Scheme.md": "Encoding Scheme",
    "spec/Ontology_Registry.md": "Ontology Registry",
    "spec/Validation_Rules.md": "Validation Rules",
    "spec/UAMS_QuickReference.md": "UAMS Quick Reference",
    "spec/UAMS_ChangeLog.md": "UAMS Change Log",
    "spec/README.md": "Spec README",
}


def evaluate_documentation():
    doc_status = {}
    present_count = 0
    absent_count = 0

    for rel_path, label in DOCUMENTATION_FILES.items():
        full_path = BASE_DIR / rel_path
        exists = full_path.exists()
        size_bytes = full_path.stat().st_size if exists else 0
        size_kb = size_bytes / 1024

        doc_status[rel_path] = {
            "label": label,
            "exists": exists,
            "size_kb": round(size_kb, 1),
        }

        if exists:
            present_count += 1
        else:
            absent_count += 1

    total_docs = len(DOCUMENTATION_FILES)
    coverage = present_count / total_docs if total_docs > 0 else 0

    broken_links = []
    if doc_status.get("README.md", {}).get("exists"):
        readme_path = BASE_DIR / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        import re

        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        for text, link in links:
            if link.startswith(("http", "https", "ftp")):
                continue
            link_path = BASE_DIR / link
            if not link_path.exists():
                broken_links.append((text, link))

    report = f"""# Stage 11: Documentation Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Documentation coverage: {coverage:.1%} ({present_count}/{total_docs} files present)
- Missing documentation: {absent_count}/{total_docs}
- Broken links: {len(broken_links)}

## Documentation Files
"""
    for rel_path, status in doc_status.items():
        icon = "✅" if status["exists"] else "❌"
        report += f"| {icon} | {status['label']} | {rel_path} | {'Yes' if status['exists'] else 'No'} | {status['size_kb']} KB |\n"

    report += f"""
## Missing Documentation ({absent_count})
"""
    for rel_path, status in doc_status.items():
        if not status["exists"]:
            report += f"- {status['label']} ({rel_path})\n"

    report += f"""
## Broken Links ({len(broken_links)})
"""
    for text, link in broken_links[:20]:
        report += f"- [{text}]({link})\n"

    report += """
## Bottlenecks & Recommendations
1. **Missing docs**: Generate all required documentation files
2. **README**: Ensure complete setup, usage, and architecture documentation
3. **Dataset Card**: Add comprehensive dataset card with variable descriptions
4. **Feature Dictionary**: Ensure all generated features are documented
5. **Broken links**: Fix or remove broken internal links in documentation
"""
    path = write_report("11_Documentation_Report.md", report)
    return {
        "coverage": coverage,
        "present": present_count,
        "missing": absent_count,
        "broken_links": len(broken_links),
    }, str(path)
