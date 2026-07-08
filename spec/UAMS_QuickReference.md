# UAMS v1.0 — Quick Reference Card

**128 columns · 13 groups · SI units · ML-ready**

---

## Group Summary

| Group | Cols | Key Variables |
|-------|------|---------------|
| A. Paper Metadata | 6 | Paper_ID, DOI, Journal, Year, Authors, Country |
| B. Crop Information | 5 | Crop, Scientific_Name, Variety, Season, Growth_Duration_Days |
| C. Experimental Design | 9 | Design, Replications, Plot_Size, Spacing_Row, Spacing_Plant, Sample_Size, Location, State, Site |
| D. Environment | 8 | Lat, Lon, Altitude, Tmax, Tmin, Tmean, Rainfall, Humidity |
| E. Soil Properties | 16 | pH, EC, OC, OM, N, P, K, S, Fe, Cu, Mn, Zn, Ca, Mg, B, Mo |
| F. Fertilizer | 7 | Treatment, Fertilizer_Name, Organic_Fertilizer, Biofertilizer, Dose, Method, Interval |
| G. Growth Parameters | 20 | Shoot/Root Length, Plant Height, Biomass, Leaf Area, Leaves, Tillers, SPAD, etc. |
| H. Yield Parameters | 13 | Yield/Plot/Acre/Ha, Fruit #/Wt/Diam, Spike Length, Seeds/Spike, 100-SW, HI, Biomass |
| I. Grain Quality | 14 | Protein, Ash, Gluten, Fiber, Carbs, Fat, N/P/K/Fe/Cu/Zn/Mn/S Content |
| J. ML Targets | 5 | Target_Yield, Target_Fertilizer, Target_N/P/K |
| K. Engineered | 16 | GDD, HU, HI_Calc, NUE, WUE, Rain_Anom, Stress, Disease_Risk, Interactions, Rolling |
| L. Leakage | 1 | Feature_Available_Before_Prediction (0/1) |
| M. Encoded | 6 | Crop_Code, Season_Code, Variety_Code, Fertilizer_Code, Soil_Texture_Code, Country_Code |

---

## Key Validation Rules

| Rule | Constraint |
|------|-----------|
| pH | 0.0 – 14.0 |
| Temperature | -20°C – 60°C |
| Humidity | 0% – 100% |
| Rainfall | ≥ 0 mm |
| EC | ≥ 0 dS/m |
| Harvest Index | 0.0 – 1.0 |
| Tmax ≥ Tmin | Strict |
| Yield ≥ 0 | Strict |

---

## Key Derived Features

| Feature | Formula |
|---------|---------|
| GDD | max(0, (Tmax+Tmin)/2 - 10°C) |
| NUE | Yield / N_applied |
| WUE | Yield / Rainfall |
| Stress_Index | \|Tmean - 25\| / 25 |
| Disease_Risk | 1.0: H>80% & T>25°C; 0.5: H>60% & T>20°C |
| Yield_ha | Plot_yield / Plot_size × 10000 |

---

## Key Leakage Flags

- **POST_HARVEST** (leaked): Quality, yield, targets → exclude for prediction
- **PRE_HARVEST_MEASUREMENT** (safe): Leaf #, tillers, height at 30/60/90 DAS, SPAD
- **AVAILABLE_BEFORE_PREDICTION** (safe): All others

---

## ML Model Compatibility

| Model Type | Needs Scaling | Needs Encoding | No Missing |
|-----------|:---:|:---:|:---:|
| Linear Regression | ✗ | ✓ | ✗ |
| Ridge/Lasso/ElasticNet | ✓ | ✓ | ✗ |
| Random Forest / Extra Trees | ✗ | ✓ | ✗ |
| XGBoost / LightGBM | ✗ | ✓ | ✓ |
| CatBoost | ✗ | ✗ | ✓ |
| SVR / KNN | ✓ | ✓ | ✗ |
| Neural Networks / LSTM | ✓ | ✓ | ✓ |
| Time Series | ✗ | ✓ | ✓ |

---

## Export Formats

| Format | File |
|--------|------|
| CSV | Universal_Agricultural_ML_Master.csv |
| Parquet | Universal_Agricultural_ML_Master.parquet |
| SQLite | Universal_Agricultural_ML_Master.sqlite |
| DuckDB | Universal_Agricultural_ML_Master.duckdb |
| Excel (15 sheets) | Universal_Agricultural_Machine_Learning_Schema_v1.xlsx |

---

## Excel Workbook Sheets

| Sheet | Content |
|-------|---------|
| 01_Metadata | Paper_ID, DOI, Journal, Year, Authors, Country |
| 02_Field_Profile | Location, Design, Plot dimensions |
| 03_Crop_Profile | Crop, Variety, Season, Growth duration |
| 04_Soil_Profile | pH, EC, OC, N, P, K, micronutrients |
| 05_Weather_TimeSeries | Tmax, Tmin, Rainfall, Humidity, GDD |
| 06_Management_Events | Fertilizer, Dose, Method |
| 07_Plant_Observations | Heights, Biomass, Leaf Area, SPAD |
| 08_Remote_Sensing | SPAD, NDVI |
| 09_Sensor_Data | Tmax, Tmin, Rainfall, Humidity, pH, EC |
| 10_Laboratory_Analysis | Protein, Ash, Gluten, Fiber, Carbs, Fat, minerals |
| 11_Derived_Features | GDD, NUE, WUE, Stress, interactions |
| 12_Targets | Yield, Fertilizer, N/P/K targets |
| 13_Data_Quality | Feature_Available_Before_Prediction |
| 14_Feature_Dictionary | Column metadata |
| 15_Evidence_Metadata | Source provenance |

---

## File Naming Convention

```
Universal_Agricultural_Machine_Learning_Schema_v{major}.{minor}.{patch}.xlsx
Universal_Agricultural_ML_Master.{csv|parquet|sqlite|duckdb}
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run pipeline
ades run input_dataset.xlsx

# Or with legacy script
python universal_schema_generator.py

# Validate
python -m pytest tests/ -v
```

---

**UAMS v1.0** — Standardizing agricultural data for ML since 2026
