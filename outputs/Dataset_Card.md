# Dataset Card
Generated: 2026-07-07T22:01:32.425141

## Dataset Description
- **Name**: Universal Agricultural Machine Learning Dataset (UAMS v1.0)
- **Format**: CSV, Parquet, SQLite, XLSX (15-sheet workbook)
- **Rows**: {len(df)}
- **Columns**: {len(df.columns)}

## Variable Summary
- Numeric features: 7
- Categorical features: 135
- Target variables: 5
- Engineered features: 14

## Data Sources
- 5 agricultural research papers (Bell pepper, Black wheat, Carrot, Cowpea, Spinach)
- 5 master datasets (Excel format)

## Intended Use
- ML model training for crop yield prediction
- Agricultural data science research
- Reproducible agricultural analytics

## Limitations
- Dataset size may be limited for deep learning approaches
- Geographic coverage limited to specific study locations
- Temporal coverage depends on original study durations