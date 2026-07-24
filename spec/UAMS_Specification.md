# Universal Agricultural Machine Learning Schema (UAMS) v1.0 — Full Specification

**Version:** 1.0  
**Status:** Stable  
**Total Columns:** 138  
**Column Groups:** 14 (A through N)

---

## A. Paper Metadata (6 columns)

Bibliographic identifiers linking each observation to its source publication.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 1 | `Paper_ID` | string | — | No | Unique identifier for the source paper |
| 2 | `DOI` | string | — | No | Digital Object Identifier |
| 3 | `Journal` | string | — | No | Journal or proceedings name |
| 4 | `Year` | integer | — | No | Publication year |
| 5 | `Authors` | string | — | No | Author list |
| 6 | `Country` | string | — | No | Country of study |

**Validation:** `Year` must be between 1950 and current year. `DOI` should match DOI pattern `^10\.\d{4,}/.*$`.

---

## B. Crop Information (6 columns)

Identity and phenological characteristics of the crop under study.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 7 | `Crop` | string | — | **Yes** | Common crop name (e.g., Wheat, Rice, Maize) |
| 8 | `Scientific_Name` | string | — | No | Binomial nomenclature (e.g., *Triticum aestivum*) |
| 9 | `Variety` | string | — | No | Cultivar or variety name |
| 10 | `Season` | string | — | No | Growing season (e.g., Rabi, Kharif, Spring) |
| 11 | `Growth_Duration_Days` | float | days | No | Days from sowing to harvest |
| 12 | `Growth_Stage` | string | — | No | Phenological growth stage at observation |

**Validation:** `Growth_Duration_Days` must be > 0 and typically < 365. `Crop` is required for schema compliance.

---

## C. Experimental Design (9 columns)

Trial design parameters and spatial layout information.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 13 | `Design` | string | — | No | Experimental design (e.g., RBD, CRD, Split-plot) |
| 14 | `Replications` | integer | — | No | Number of replications/blocks |
| 15 | `Plot_Size` | float | m² | No | Net plot area |
| 16 | `Spacing_Row` | float | cm | No | Row spacing |
| 17 | `Spacing_Plant` | float | cm | No | Plant-to-plant spacing |
| 18 | `Sample_Size` | integer | — | No | Number of plants sampled |
| 19 | `Location` | string | — | No | Experiment location name |
| 20 | `State` | string | — | No | State or administrative region |
| 21 | `Site` | string | — | No | Specific site/farm name |

**Validation:** `Replications` must be ≥ 2. `Plot_Size` must be > 0. `Spacing_Row` and `Spacing_Plant` must be ≥ 0.

---

## D. Environment (8 columns)

Geospatial coordinates and weather variables.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 22 | `Latitude` | float | decimal degrees | No | Latitude (-90 to 90) |
| 23 | `Longitude` | float | decimal degrees | No | Longitude (-180 to 180) |
| 24 | `Altitude` | float | m | No | Elevation above sea level |
| 25 | `Temperature_Max` | float | °C | No | Maximum temperature |
| 26 | `Temperature_Min` | float | °C | No | Minimum temperature |
| 27 | `Average_Temperature` | float | °C | No | Mean temperature |
| 28 | `Rainfall` | float | mm | No | Total precipitation |
| 29 | `Humidity` | float | % | No | Relative humidity |

**Validation:** `Temperature_Max` ≥ `Temperature_Min`. All temperatures between -20°C and 60°C. `Humidity` 0–100%. `Rainfall` ≥ 0. `Latitude` -90 to 90. `Longitude` -180 to 180.

---

## E. Soil Properties (16 columns)

Pre-sowing or pre-planting soil physico-chemical analysis.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 30 | `Soil_pH` | float | — | No | Soil pH (1:2.5 soil:water suspension) |
| 31 | `EC` | float | dS/m | No | Electrical conductivity |
| 32 | `Organic_Carbon` | float | % | No | Soil organic carbon |
| 33 | `Organic_Matter` | float | % | No | Soil organic matter |
| 34 | `Nitrogen` | float | kg/ha | No | Available nitrogen |
| 35 | `Phosphorus` | float | kg/ha | No | Available phosphorus (P₂O₅) |
| 36 | `Potassium` | float | kg/ha | No | Available potassium (K₂O) |
| 37 | `Sulphur` | float | ppm | No | Available sulphur |
| 38 | `Iron` | float | ppm | No | DTPA-extractable iron |
| 39 | `Copper` | float | ppm | No | DTPA-extractable copper |
| 40 | `Manganese` | float | ppm | No | DTPA-extractable manganese |
| 41 | `Zinc` | float | ppm | No | DTPA-extractable zinc |
| 42 | `Calcium` | float | ppm | No | Exchangeable calcium |
| 43 | `Magnesium` | float | ppm | No | Exchangeable magnesium |
| 44 | `Boron` | float | ppm | No | Hot-water soluble boron |
| 45 | `Molybdenum` | float | ppm | No | Available molybdenum |

**Validation:** `Soil_pH` 0–14. `EC` ≥ 0. `Organic_Carbon` and `Organic_Matter` typically 0–10%. All concentrations ≥ 0.

---

## F. Fertilizer Information (7 columns)

Fertilizer treatment and application details.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 46 | `Treatment` | string | — | No | Treatment identifier/name |
| 47 | `Fertilizer_Name` | string | — | No | Fertilizer product or type |
| 48 | `Organic_Fertilizer` | string | — | No | Organic amendment type |
| 49 | `Biofertilizer` | string | — | No | Biofertilizer type (e.g., Rhizobium, Azotobacter) |
| 50 | `Dose` | float | kg/ha | No | Application rate |
| 51 | `Application_Method` | string | — | No | Method (e.g., Broadcasting, Band placement, Foliar) |
| 52 | `Application_Interval` | string | — | No | Timing/frequency description |

**Validation:** `Dose` must be ≥ 0.

---

## G. Crop Growth Parameters (22 columns)

Biophysical measurements recorded during the growing season.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 53 | `Shoot_Length_cm` | float | cm | No | Shoot length |
| 54 | `Root_Length_cm` | float | cm | No | Root length |
| 55 | `Plant_Height_cm` | float | cm | No | Final plant height (120 DAS) |
| 56 | `Shoot_Biomass_g` | float | g | No | Shoot dry biomass |
| 57 | `Root_Biomass_g` | float | g | No | Root dry biomass |
| 58 | `Leaf_Area_cm2` | float | cm² | No | Leaf area (final, 120 DAS) |
| 59 | `Leaf_Number` | float | — | No | Number of leaves per plant |
| 60 | `Tillers` | float | — | No | Number of tillers per plant |
| 61 | `Root_Diameter_mm` | float | mm | No | Root diameter |
| 62 | `SPAD` | float | SPAD | No | Chlorophyll meter reading |
| 63 | `Moisture_Content` | float | % | No | Moisture content |
| 64 | `Dry_Matter` | float | % | No | Dry matter percentage |
| 65 | `Stem_Diameter_mm` | float | mm | No | Stem diameter |
| 66 | `Branches` | float | — | No | Number of branches per plant |
| 67 | `Nodes` | float | — | No | Number of nodes |
| 68 | `Flowers` | float | — | No | Number of flowers per plant |
| 69 | `Plant_Height_30_cm` | float | cm | No | Plant height at 30 DAS |
| 70 | `Plant_Height_60_cm` | float | cm | No | Plant height at 60 DAS |
| 71 | `Plant_Height_90_cm` | float | cm | No | Plant height at 90 DAS |
| 72 | `Leaf_Area_30_cm2` | float | cm² | No | Leaf area at 30 DAS |
| 73 | `Leaf_Area_60_cm2` | float | cm² | No | Leaf area at 60 DAS |
| 74 | `Leaf_Area_90_cm2` | float | cm² | No | Leaf area at 90 DAS |

**Validation:** All lengths ≥ 0. All biomass ≥ 0. `SPAD` typically 0–80. `Moisture_Content` 0–100%. Timepoint columns: `Plant_Height_30_cm ≤ Plant_Height_60_cm ≤ Plant_Height_90_cm ≤ Plant_Height_cm`.

---

## H. Yield Parameters (13 columns)

Yield and yield-attributing characters at harvest.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 75 | `Yield_per_Plot` | float | g/plot | No | Plot-level yield |
| 76 | `Yield_per_Acre` | float | kg/acre | No | Per-acre yield |
| 77 | `Yield_per_Hectare` | float | kg/ha | No | Per-hectare yield |
| 78 | `Fruit_Number` | float | — | No | Fruits per plant |
| 79 | `Fruit_Weight` | float | g | No | Individual fruit weight |
| 80 | `Fruit_Diameter_mm` | float | mm | No | Fruit diameter |
| 81 | `Spike_Length` | float | cm | No | Spike/panicle length |
| 82 | `Seeds_per_Spike` | float | — | No | Seeds per spike |
| 83 | `100_Seed_Weight` | float | g | No | 100-seed / test weight |
| 84 | `Root_Weight` | float | g | No | Root weight |
| 85 | `Pod_Weight` | float | g | No | Pod weight |
| 86 | `Harvest_Index` | float | ratio | No | Harvest index (yield / biomass) |
| 87 | `Biomass_Yield` | float | kg/ha | No | Total biomass yield |

**Validation:** All yield values ≥ 0. `Harvest_Index` 0–1. `100_Seed_Weight` > 0.

---

## I. Grain Quality (14 columns)

Post-harvest nutritional and quality analysis.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 88 | `Protein` | float | % | No | Crude protein content |
| 89 | `Ash` | float | % | No | Ash content |
| 90 | `Gluten` | float | % | No | Gluten content |
| 91 | `Fiber` | float | % | No | Crude fiber content |
| 92 | `Carbohydrates` | float | % | No | Carbohydrate content |
| 93 | `Fat` | float | % | No | Fat/oil content |
| 94 | `Nitrogen_Content` | float | % | No | Grain nitrogen content |
| 95 | `Phosphorus_Content` | float | % | No | Grain phosphorus content |
| 96 | `Potassium_Content` | float | % | No | Grain potassium content |
| 97 | `Iron_Content` | float | ppm | No | Grain iron content |
| 98 | `Copper_Content` | float | ppm | No | Grain copper content |
| 99 | `Zinc_Content` | float | ppm | No | Grain zinc content |
| 100 | `Manganese_Content` | float | ppm | No | Grain manganese content |
| 101 | `Sulphur_Content` | float | % | No | Grain sulphur content |

**Validation:** All percentages 0–100%. All concentrations ≥ 0. `Protein + Ash + Gluten + Fiber + Carbohydrates + Fat ≤ 100%` (approximate).

---

## J. ML Target Variables (5 columns)

Prediction target variables for supervised learning.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 102 | `Target_Yield` | float | kg/ha | No | Primary yield prediction target |
| 103 | `Target_Fertilizer` | float | kg/ha | No | Fertilizer dose prediction target |
| 104 | `Target_Nitrogen` | float | kg/ha | No | Nitrogen requirement target |
| 105 | `Target_Phosphorus` | float | kg/ha | No | Phosphorus requirement target |
| 106 | `Target_Potassium` | float | kg/ha | No | Potassium requirement target |

**Validation:** All targets ≥ 0.

---

## K. Engineered Features (16 columns)

Domain-specific derived features computed from raw measurements.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 107 | `Growing_Degree_Days` | float | °C·day | No | Accumulated thermal time above base (10°C) |
| 108 | `Heat_Units` | float | °C·day | No | Heat accumulation using Tmax |
| 109 | `Harvest_Index_Calc` | float | ratio | No | Computed harvest index |
| 110 | `Nitrogen_Use_Efficiency` | float | kg/kg | No | Yield per unit N applied |
| 111 | `Water_Use_Efficiency` | float | kg/ha·mm | No | Yield per unit rainfall |
| 112 | `Rainfall_Anomaly` | float | mm | No | Deviation from mean rainfall |
| 113 | `Stress_Index` | float | ratio | No | Thermal stress relative to optimum (25°C) |
| 114 | `Disease_Risk_Index` | float | 0–1 | No | Categorical disease risk score |
| 115 | `Yield_per_Plant` | float | g/plant | No | Per-plant yield estimate |
| 116 | `Yield_per_Plot_Calc` | float | g/plot | No | Direct plot yield |
| 117 | `Yield_per_Hectare_Calc` | float | kg/ha | No | Extrapolated hectare yield |
| 118 | `Temp_x_Rainfall` | float | °C·mm | No | Temperature–rainfall interaction |
| 119 | `N_x_P` | float | (kg/ha)² | No | Nitrogen–phosphorus interaction |
| 120 | `Temp_ squared` | float | °C² | No | Quadratic temperature term |
| 121 | `Rainfall_7d_MA` | float | mm | No | 7-day rainfall moving average |
| 122 | `Temp_7d_MA` | float | °C | No | 7-day temperature moving average |

**Formulas:**

```
Growing_Degree_Days = max(0, ((Tmax + Tmin) / 2) - Tbase)
    where Tbase = 10°C

Heat_Units = max(0, Tmax - Tbase)

Harvest_Index_Calc = Yield / Biomass (where biomass > 0)

Nitrogen_Use_Efficiency = Yield / N_applied (where N > 0)

Water_Use_Efficiency = Yield / Rainfall (where rainfall > 0)

Rainfall_Anomaly = Rainfall - mean(Rainfall)

Stress_Index = |Tmean - 25°C| / 25°C

Disease_Risk_Index:
    1.0 if Humidity > 80% and Tmax > 25°C
    0.5 if Humidity > 60% and Tmax > 20°C
    0.0 otherwise
```

---

## L. Leakage Labels (1 column)

Indicator of whether features were available at prediction time.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 123 | `Feature_Available_Before_Prediction` | integer | — | No | 1 = available before harvest, 0 = post-harvest variable |

**Leakage classification:**
- **POST_HARVEST** (leaked): Protein, Ash, Gluten, Fiber, Carbohydrates, Fat, Nitrogen_Content, Phosphorus_Content, Potassium_Content, Iron_Content, Copper_Content, Zinc_Content, Manganese_Content, Sulphur_Content, 100_Seed_Weight, Root_Weight, Pod_Weight, Harvest_Index, Biomass_Yield, Yield_per_Plot, Yield_per_Acre, Yield_per_Hectare, Fruit_Number, Fruit_Weight, Fruit_Diameter_mm, Spike_Length, Seeds_per_Spike, Target_* variables
- **PRE_HARVEST_MEASUREMENT** (safe): Leaf_Number, Tillers, Branches, Nodes, Flowers, Plant_Height_30/60/90_cm, Leaf_Area_30/60/90_cm2, SPAD, Moisture_Content
- **AVAILABLE_BEFORE_PREDICTION** (safe): All other columns

---

## M. Encoded Variables (6 columns)

Numeric encodings of categorical variables for ML consumption.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 124 | `Crop_Code` | integer | — | No | Numeric code for Crop |
| 125 | `Season_Code` | integer | — | No | Numeric code for Season |
| 126 | `Variety_Code` | integer | — | No | Numeric code for Variety |
| 127 | `Fertilizer_Code` | integer | — | No | Numeric code for Fertilizer_Name |
| 128 | `Soil_Texture_Code` | integer | — | No | Numeric code for Soil_Texture |
| 129 | `Country_Code` | integer | — | No | Numeric code for Country |

**Encoding method:** LabelEncoder (integer encoding, 0 to n_classes-1). Original values preserved in separate columns. Encoding map stored in `Encoding_Map.csv`.

---

## N. ML Predictions (9 columns)

Model outputs and recommendation fields written by the prediction, recommendation, and
fuzzy-logic agents. These are produced at prediction time and are never valid predictors
(see `agri_ai_agent/ml/leakage.py`).

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 130 | `Predicted_Yield` | float | kg/ha | No | Model-predicted yield |
| 131 | `Expected_Biomass` | float | g | No | Model-predicted biomass |
| 132 | `Expected_Plant_Height` | float | cm | No | Model-predicted plant height |
| 133 | `Recommended_Fertilizer` | string | — | No | Fertilizer recommended by the fuzzy expert system |
| 134 | `Recommended_Dose` | float | kg/ha | No | Recommended fertilizer dose |
| 135 | `Recommended_Application_Interval` | string | — | No | Recommended application interval |
| 136 | `Expected_Yield_Increase` | float | % | No | Expected yield increase from the recommendation |
| 137 | `Confidence_Score` | float | 0–1 | No | Recommendation confidence |
| 138 | `Recommendation_Summary` | string | — | No | Human-readable recommendation summary |

---

## Physical File Format

**Primary format:** UTF-8 CSV with header row.  
**Missing values:** Empty cells or explicit `NA`. Never zero-filled.  
**Float precision:** 4 decimal places for display, full precision in storage.  
**Row order:** No guaranteed order; sort by key columns as needed.  

### Export Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | `.csv` | Universal plain-text |
| Parquet | `.parquet` | Columnar, compressed, fast I/O |
| SQLite | `.sqlite` | Relational database |
| DuckDB | `.duckdb` | Analytical database |
| Excel | `.xlsx` | 15-sheet workbook |

### Excel Workbook Sheets

| Sheet | Content |
|-------|---------|
| 01_Metadata | Paper identifiers |
| 02_Field_Profile | Location, design, plot dimensions |
| 03_Crop_Profile | Crop identity, growth duration |
| 04_Soil_Profile | Full soil analysis |
| 05_Weather_TimeSeries | Weather and GDD |
| 06_Management_Events | Fertilizer and treatments |
| 07_Plant_Observations | Growth parameters |
| 08_Remote_Sensing | SPAD, NDVI |
| 09_Sensor_Data | Continuous sensor readings |
| 10_Laboratory_Analysis | Grain quality |
| 11_Derived_Features | Engineered features |
| 12_Targets | Yield and targets |
| 13_Data_Quality | Leakage labels |
| 14_Feature_Dictionary | Column metadata |
| 15_Evidence_Metadata | Source provenance |

## Versioning

This specification follows [Semantic Versioning 2.0.0](https://semver.org/):
- **MAJOR**: Breaking schema changes (column removal, type changes)
- **MINOR**: Backward-compatible additions (new columns, optional fields)
- **PATCH**: Clarifications, corrections, documentation improvements
