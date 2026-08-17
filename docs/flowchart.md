# AAIF Pipeline Flowchart

```mermaid
flowchart TB
    classDef phase0 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef phase1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef phase2 fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c
    classDef phase3 fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#b71c1c
    classDef phase4 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#4a148c
    classDef external fill:#efebe9,stroke:#4e342e,stroke-width:1px,color:#3e2723

    subgraph EXTERNAL["PHASE -1  External Data Source Layer"]
        direction TB
        REG["Connector Registry\npkgutil auto-discovery"]
        C1["CGIAR\nhuggingface.co"]
        C2["FAOSTAT\nfao.org"]
        C3["NASA POWER\npower.larc.nasa.gov"]
        C4["SoilGrids\nsoilgrids.org"]
        C5["ISRIC\nisric.org"]
        C6["Zenodo\nzenodo.org"]
        C7["Mendeley\ndata.mendeley.com"]
        C8["Kaggle\nkaggle.com"]
        C9["ICAR\nkrishikosh.egranth.ac.in"]
        C10["SAU\nState Ag. Univ. portals"]
        DP["DatasetPackage\nstandardized format"]

        REG --> C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10
        C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 & C9 & C10 --> DP
    end

    subgraph P0["PHASE 0  Ingestion & Normalization"]
        direction TB
        S1["① Repository Sync\nChange Detection"]
        S2["② External Data Source\nPlugin Ingestion"]
        S3["③ Dataset Normalization\n→ DatasetPackage"]
        S4["④ Dataset Ingestion Bridge\nID Hierarchy Assignment"]
    end

    subgraph P1["PHASE 1  Extraction & Knowledge"]
        direction TB
        S5["⑤ Extraction\n6-Reader Hybrid PDF\n(Pdfminer/Camelot/Pdfplumber\nPoppler/OCR/Semantic)"]
        S6["⑥ Evidence Fusion\nConfidence-Weighted\nMulti-Reader Dedup"]
        S7["⑦ Ontology\nColumn → UAMS\n553 Variant Map"]
        S8["⑧ Table Intelligence\nClassification & Stats"]
        S9["⑨ Schema Population\nDerived Columns\nMissing Value Inference"]
        S10["⑩ Knowledge Integration\nDomain Imputation"]
        S11["⑪ Knowledge\nDomain Constraints"]
        S12["⑫ Observation Generation\nHierarchy Builder\nPaper_ID / Dataset_ID"]
    end

    subgraph P2["PHASE 2  Validation & Features"]
        direction TB
        S13["⑬ Validation\nBiological Range\nIQR Outliers\nUnit Harmonization"]
        S14["⑭ Feature Store\nValidation Gate\nParquet Persistence"]
        S15["⑮ Feature Engineering\n140+ Composite Features\nNPK Index / GDD / NUE / WUE"]
    end

    subgraph P3["PHASE 3  ML Training & Prediction"]
        direction TB
        S16["⑯ Model Selection\nGridSearchCV\nSelectKBest\nAdaptive Pool"]
        S17["⑰ Training\n8 Regression Families\nLinear / Ridge / Lasso / ElasticNet\nRF / GBM / XGBoost / SVR\n5-Fold CV / LOO"]
        S18["⑱ Prediction\nLoad Best Models\n5 Target Predictions"]
    end

    subgraph P4["PHASE 4  Recommendation & Export"]
        direction TB
        S19["⑲ Recommendation\nTop-3 Treatment\nAlternatives"]
        S20["⑳ Fuzzy Logic\n221 Mamdani Rules\n10 Input MFs / 3 Output MFs\nCentroid Defuzzification"]
        S21["㉑ Benchmark\nPhase Timing\nQuality & Coverage Metrics"]
        S22["㉒ Explainability\nFeature Attribution\nSHAP / Permutation"]
        S23["㉓ Ready Reckoner\nExcel / CSV / HTML / JSON\nPer-Crop Decision Tables"]
    end

    subgraph CL["Continuous Learning"]
        CL1["Drift Detection\nChangeDetector"]
        CL2["Incremental Retrain\nIncrementalEngine"]
        CL3["Version History\nVersionHistory"]
    end

    subgraph OUT["Outputs"]
        M1["Models\n9 .joblib + yield_model.pkl"]
        M2["Ready Reckoner\nHTML / CSV / JSON / MD"]
        M3["Reports\nBenchmark / Validation\nProvenance / Assessment"]
        M4["Feature Dictionary\nCorrelation Matrix\nSummary Statistics"]
        M5["Observations\n66 Rows / 296 Columns\n26 Groups (A-Z)"]
    end

    DP --> P0
    P0 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4

    P0 --> S1 --> S2 --> S3 --> S4
    S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11 --> S12
    S12 --> S13 --> S14 --> S15
    S15 --> S16 --> S17 --> S18
    S18 --> S19 --> S20 --> S21 --> S22 --> S23

    S23 --> OUT

    S4 -.-> CL
    CL -.-> S16
```

## Pipeline Summary

| Property | Value |
|----------|-------|
| **Total Agents** | 23 sequential |
| **Pipeline Phases** | 5 (0–4) + PHASE -1 (External Data) |
| **UAMS Schema** | 296 columns, 26 groups (A–Z) |
| **Variant Map** | 553 unique entries |
| **Production Run** | PROD_20260729_134301 — 134.2s, 22/22 agents |
| **Output** | 66 observations, 9+1 trained models |
| **Fuzzy Rules** | 221 Mamdani (v3.0) |
| **Model Families** | 8 regression (Linear, Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, SVR) |

## Data Flow Legend

```
Solid arrow  ──►  Required sequential flow
Dotted arrow  ~~►  Optional / conditional flow
```
