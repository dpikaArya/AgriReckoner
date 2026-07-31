"""
Dataset Quality Score Computation.
Computes schema completeness, ontology completeness, feature completeness,
missing value score, documentation score, ML readiness score,
scientific reproducibility score, overall quality score (max 100).
"""

import time
from pathlib import Path

from evaluation.utils import (
    OUTPUT_DIR,
    get_master_df,
    get_uams_cols,
    load_dataframe,
    write_report,
)


def compute_quality_score():
    master_df = get_master_df()
    uams_cols = get_uams_cols()
    uams_set = set(uams_cols)
    total_uams = len(uams_cols)

    if master_df is None:
        report = "# Dataset Quality Score\nNo data available.\n"
        path = write_report("Dataset_Quality_Score.md", report)
        return {"overall": 0}, str(path)

    n_rows, n_cols = master_df.shape

    # 1. Schema completeness (0-20) - use core UAMS columns (A-N)
    cols_in_schema = []
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agri_ai_agent.config.schema import SCHEMA_GROUPS

        core_cols_list = []
        for g_name, g_cols in SCHEMA_GROUPS.items():
            if any(
                g_name.startswith(p)
                for p in [
                    "A.",
                    "B.",
                    "C.",
                    "D.",
                    "E.",
                    "F.",
                    "G.",
                    "H.",
                    "I.",
                    "J.",
                    "K.",
                    "L.",
                    "M.",
                    "N.",
                ]
            ):
                core_cols_list.extend(g_cols)
        core_set = set(core_cols_list)
        cols_in_core = [c for c in master_df.columns if c in core_set]
        cols_in_schema = cols_in_core
        schema_completeness = len(cols_in_core) / len(core_cols_list) if core_cols_list else 0
    except (ImportError, Exception):
        cols_in_schema = [c for c in master_df.columns if c in uams_set]
        schema_completeness = len(cols_in_schema) / total_uams if total_uams > 0 else 0
    schema_score = schema_completeness * 20

    # 2. Ontology completeness (0-10)
    onto_path = OUTPUT_DIR / "Ontology_Mapping.csv"
    onto_score = 0
    if onto_path.exists():
        onto_df = load_dataframe(str(onto_path))
        if onto_df is not None and len(onto_df) > 0:
            onto_cols = [
                "AGROVOC",
                "Crop Ontology",
                "Plant Ontology",
                "Environment Ontology",
                "Unit Ontology",
            ]
            onto_present = sum(
                1 for c in onto_cols if c in onto_df.columns and onto_df[c].notna().any()
            )
            onto_score = (onto_present / len(onto_cols)) * 10

    # 3. Feature completeness (0-15)
    engineered = [
        "Growing_Degree_Days",
        "Heat_Units",
        "Harvest_Index_Calc",
        "Nitrogen_Use_Efficiency",
        "Water_Use_Efficiency",
        "Yield_per_Plant",
        "Yield_per_Hectare_Calc",
        "Temp_x_Rainfall",
        "N_x_P",
    ]
    existing_eng = [f for f in engineered if f in master_df.columns]
    feature_completeness = len(existing_eng) / len(engineered) if engineered else 0
    feature_score = feature_completeness * 15

    # 4. Missing value score (0-15)
    total_cells = n_rows * n_cols
    total_missing = int(master_df.isna().sum().sum())
    missing_pct = total_missing / total_cells if total_cells > 0 else 1
    missing_score = max(0, (1 - missing_pct) * 15)

    # 5. Documentation score (0-10)
    doc_files = [
        OUTPUT_DIR / "Feature_Dictionary.csv",
        OUTPUT_DIR / "Variable_Mapping.csv",
        OUTPUT_DIR / "Quality_Report.md",
        OUTPUT_DIR / "Model_Readiness_Report.md",
        OUTPUT_DIR / "Pipeline_Provenance.json",
    ]
    doc_present = sum(1 for f in doc_files if f.exists())
    doc_score = (doc_present / len(doc_files)) * 10

    # 6. ML readiness score (0-15)
    ml_issues = 0
    if missing_pct > 0.1:
        ml_issues += 1
    cat_cols = len(master_df.select_dtypes(include=["object", "category"]).columns)
    if cat_cols > 0:
        ml_issues += 1
    dur_indicator = "Feature_Available_Before_Prediction" not in master_df.columns
    if dur_indicator:
        ml_issues += 1
    ml_score = max(0, (1 - ml_issues / 5) * 15)

    # 7. Scientific reproducibility score (0-15)
    has_paper_meta = all(c in master_df.columns for c in ["Paper_ID", "DOI", "Year", "Authors"])
    has_crop_info = "Crop" in master_df.columns
    has_location = "Location" in master_df.columns
    has_soil = "Soil_pH" in master_df.columns
    has_weather = any(c in master_df.columns for c in ["Temperature_Max", "Rainfall"])
    has_yield_data = any(c in master_df.columns for c in ["Yield_per_Plot", "Target_Yield"])

    repro_factors = [
        has_paper_meta,
        has_crop_info,
        has_location,
        has_soil,
        has_weather,
        has_yield_data,
    ]
    repro_score = (sum(repro_factors) / len(repro_factors)) * 15

    overall = min(
        100,
        schema_score
        + onto_score
        + feature_score
        + missing_score
        + doc_score
        + ml_score
        + repro_score,
    )

    report = f"""# Dataset Quality Score
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {n_rows} rows × {n_cols} columns

## Quality Scorecard
| Component | Score | Max | % |
|-----------|-------|-----|---|
| Schema Completeness | {schema_score:.1f} | 20 | {schema_completeness:.1%} |
| Ontology Completeness | {onto_score:.1f} | 10 | {onto_score / 10:.1%} |
| Feature Completeness | {feature_score:.1f} | 15 | {feature_completeness:.1%} |
| Missing Value Score | {missing_score:.1f} | 15 | {missing_score / 15:.1%} |
| Documentation Score | {doc_score:.1f} | 10 | {doc_score / 10:.1%} |
| ML Readiness Score | {ml_score:.1f} | 15 | {ml_score / 15:.1%} |
| Scientific Reproducibility | {repro_score:.1f} | 15 | {repro_score / 15:.1%} |
| **Overall Quality Score** | **{overall:.1f}** | **100** | **{overall:.1f}%** |

## Detailed Breakdown

### Schema Completeness ({schema_score:.1f}/20)
- UAMS columns in dataset: {len(cols_in_schema)}/{total_uams}
- Missing schema groups: TODO

### Ontology Completeness ({onto_score:.1f}/10)
- Ontology sources mapped: {onto_present if onto_path.exists() else 0}/5

### Feature Completeness ({feature_score:.1f}/15)
- Engineered features present: {len(existing_eng)}/{len(engineered)}
- Features missing: {[f for f in engineered if f not in master_df.columns]}

### Missing Values ({missing_score:.1f}/15)
- Missing rate: {missing_pct:.1%}
- Total missing: {total_missing}/{total_cells}

### Documentation ({doc_score:.1f}/10)
- Docs present: {doc_present}/{len(doc_files)}

### ML Readiness ({ml_score:.1f}/15)
- Issues: missing={missing_pct > 0.1}, categorical={cat_cols > 0}, leakage_flag={dur_indicator}

### Scientific Reproducibility ({repro_score:.1f}/15)
- Paper metadata: {has_paper_meta}
- Crop info: {has_crop_info}
- Location: {has_location}
- Soil data: {has_soil}
- Weather data: {has_weather}
- Yield data: {has_yield_data}

## Recommendations
1. Increase schema coverage by mapping more columns to UAMS
2. Expand ontology coverage with additional sources
3. Implement missing engineered features
4. Reduce missing values through imputation
5. Generate comprehensive documentation
6. Address ML readiness issues for production deployment
"""
    path = write_report("Dataset_Quality_Score.md", report)
    return {
        "overall": round(overall, 1),
        "schema": round(schema_score, 1),
        "ontology": round(onto_score, 1),
        "features": round(feature_score, 1),
        "missing_values": round(missing_score, 1),
        "documentation": round(doc_score, 1),
        "ml_readiness": round(ml_score, 1),
        "reproducibility": round(repro_score, 1),
    }, str(path)
