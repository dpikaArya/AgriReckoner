# Universal Agricultural Machine Learning Schema (UAMS) v1.0 — Full Specification

**Version:** 1.0  
**Status:** Stable  
**Total Columns:** 128  
**Column Groups:** 13 (A through M)

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

## B. Crop Information (5 columns)

Identity and phenological characteristics of the crop under study.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 7 | `Crop` | string | — | **Yes** | Common crop name (e.g., Wheat, Rice, Maize) |
| 8 | `Scientific_Name` | string | — | No | Binomial nomenclature (e.g., *Triticum aestivum*) |
| 9 | `Variety` | string | — | No | Cultivar or variety name |
| 10 | `Season` | string | — | No | Growing season (e.g., Rabi, Kharif, Spring) |
| 11 | `Growth_Duration_Days` | float | days | No | Days from sowing to harvest |

**Validation:** `Growth_Duration_Days` must be > 0 and typically < 365. `Crop` is required for schema compliance.

---

## C. Experimental Design (9 columns)

Trial design parameters and spatial layout information.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 12 | `Design` | string | — | No | Experimental design (e.g., RBD, CRD, Split-plot) |
| 13 | `Replications` | integer | — | No | Number of replications/blocks |
| 14 | `Plot_Size` | float | m² | No | Net plot area |
| 15 | `Spacing_Row` | float | cm | No | Row spacing |
| 16 | `Spacing_Plant` | float | cm | No | Plant-to-plant spacing |
| 17 | `Sample_Size` | integer | — | No | Number of plants sampled |
| 18 | `Location` | string | — | No | Experiment location name |
| 19 | `State` | string | — | No | State or administrative region |
| 20 | `Site` | string | — | No | Specific site/farm name |

**Validation:** `Replications` must be ≥ 2. `Plot_Size` must be > 0. `Spacing_Row` and `Spacing_Plant` must be ≥ 0.

---

## D. Environment (8 columns)

Geospatial coordinates and weather variables.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 21 | `Latitude` | float | decimal degrees | No | Latitude (-90 to 90) |
| 22 | `Longitude` | float | decimal degrees | No | Longitude (-180 to 180) |
| 23 | `Altitude` | float | m | No | Elevation above sea level |
| 24 | `Temperature_Max` | float | °C | No | Maximum temperature |
| 25 | `Temperature_Min` | float | °C | No | Minimum temperature |
| 26 | `Average_Temperature` | float | °C | No | Mean temperature |
| 27 | `Rainfall` | float | mm | No | Total precipitation |
| 28 | `Humidity` | float | % | No | Relative humidity |

**Validation:** `Temperature_Max` ≥ `Temperature_Min`. All temperatures between -20°C and 60°C. `Humidity` 0–100%. `Rainfall` ≥ 0. `Latitude` -90 to 90. `Longitude` -180 to 180.

---

## E. Soil Properties (16 columns)

Pre-sowing or pre-planting soil physico-chemical analysis.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 29 | `Soil_pH` | float | — | No | Soil pH (1:2.5 soil:water suspension) |
| 30 | `EC` | float | dS/m | No | Electrical conductivity |
| 31 | `Organic_Carbon` | float | % | No | Soil organic carbon |
| 32 | `Organic_Matter` | float | % | No | Soil organic matter |
| 33 | `Nitrogen` | float | kg/ha | No | Available nitrogen |
| 34 | `Phosphorus` | float | kg/ha | No | Available phosphorus (P₂O₅) |
| 35 | `Potassium` | float | kg/ha | No | Available potassium (K₂O) |
| 36 | `Sulphur` | float | ppm | No | Available sulphur |
| 37 | `Iron` | float | ppm | No | DTPA-extractable iron |
| 38 | `Copper` | float | ppm | No | DTPA-extractable copper |
| 39 | `Manganese` | float | ppm | No | DTPA-extractable manganese |
| 40 | `Zinc` | float | ppm | No | DTPA-extractable zinc |
| 41 | `Calcium` | float | ppm | No | Exchangeable calcium |
| 42 | `Magnesium` | float | ppm | No | Exchangeable magnesium |
| 43 | `Boron` | float | ppm | No | Hot-water soluble boron |
| 44 | `Molybdenum` | float | ppm | No | Available molybdenum |

**Validation:** `Soil_pH` 0–14. `EC` ≥ 0. `Organic_Carbon` and `Organic_Matter` typically 0–10%. All concentrations ≥ 0.

---

## F. Fertilizer Information (7 columns)

Fertilizer treatment and application details.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 45 | `Treatment` | string | — | No | Treatment identifier/name |
| 46 | `Fertilizer_Name` | string | — | No | Fertilizer product or type |
| 47 | `Organic_Fertilizer` | string | — | No | Organic amendment type |
| 48 | `Biofertilizer` | string | — | No | Biofertilizer type (e.g., Rhizobium, Azotobacter) |
| 49 | `Dose` | float | kg/ha | No | Application rate |
| 50 | `Application_Method` | string | — | No | Method (e.g., Broadcasting, Band placement, Foliar) |
| 51 | `Application_Interval` | string | — | No | Timing/frequency description |

**Validation:** `Dose` must be ≥ 0.

---

## G. Crop Growth Parameters (20 columns)

Biophysical measurements recorded during the growing season.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 52 | `Shoot_Length_cm` | float | cm | No | Shoot length |
| 53 | `Root_Length_cm` | float | cm | No | Root length |
| 54 | `Plant_Height_cm` | float | cm | No | Final plant height (120 DAS) |
| 55 | `Shoot_Biomass_g` | float | g | No | Shoot dry biomass |
| 56 | `Root_Biomass_g` | float | g | No | Root dry biomass |
| 57 | `Leaf_Area_cm2` | float | cm² | No | Leaf area (final, 120 DAS) |
| 58 | `Leaf_Number` | float | — | No | Number of leaves per plant |
| 59 | `Tillers` | float | — | No | Number of tillers per plant |
| 60 | `Root_Diameter_mm` | float | mm | No | Root diameter |
| 61 | `SPAD` | float | SPAD | No | Chlorophyll meter reading |
| 62 | `Moisture_Content` | float | % | No | Moisture content |
| 63 | `Dry_Matter` | float | % | No | Dry matter percentage |
| 64 | `Stem_Diameter_mm` | float | mm | No | Stem diameter |
| 65 | `Branches` | float | — | No | Number of branches per plant |
| 66 | `Nodes` | float | — | No | Number of nodes |
| 67 | `Flowers` | float | — | No | Number of flowers per plant |
| 68 | `Plant_Height_30_cm` | float | cm | No | Plant height at 30 DAS |
| 69 | `Plant_Height_60_cm` | float | cm | No | Plant height at 60 DAS |
| 70 | `Plant_Height_90_cm` | float | cm | No | Plant height at 90 DAS |
| 71 | `Leaf_Area_30_cm2` | float | cm² | No | Leaf area at 30 DAS |
| 72 | `Leaf_Area_60_cm2` | float | cm² | No | Leaf area at 60 DAS |
| 73 | `Leaf_Area_90_cm2` | float | cm² | No | Leaf area at 90 DAS |

**Validation:** All lengths ≥ 0. All biomass ≥ 0. `SPAD` typically 0–80. `Moisture_Content` 0–100%. Timepoint columns: `Plant_Height_30_cm ≤ Plant_Height_60_cm ≤ Plant_Height_90_cm ≤ Plant_Height_cm`.

---

## H. Yield Parameters (13 columns)

Yield and yield-attributing characters at harvest.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 74 | `Yield_per_Plot` | float | g/plot | No | Plot-level yield |
| 75 | `Yield_per_Acre` | float | kg/acre | No | Per-acre yield |
| 76 | `Yield_per_Hectare` | float | kg/ha | No | Per-hectare yield |
| 77 | `Fruit_Number` | float | — | No | Fruits per plant |
| 78 | `Fruit_Weight` | float | g | No | Individual fruit weight |
| 79 | `Fruit_Diameter_mm` | float | mm | No | Fruit diameter |
| 80 | `Spike_Length` | float | cm | No | Spike/panicle length |
| 81 | `Seeds_per_Spike` | float | — | No | Seeds per spike |
| 82 | `100_Seed_Weight` | float | g | No | 100-seed / test weight |
| 83 | `Root_Weight` | float | g | No | Root weight |
| 84 | `Pod_Weight` | float | g | No | Pod weight |
| 85 | `Harvest_Index` | float | ratio | No | Harvest index (yield / biomass) |
| 86 | `Biomass_Yield` | float | kg/ha | No | Total biomass yield |

**Validation:** All yield values ≥ 0. `Harvest_Index` 0–1. `100_Seed_Weight` > 0.

---

## I. Grain Quality (14 columns)

Post-harvest nutritional and quality analysis.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 87 | `Protein` | float | % | No | Crude protein content |
| 88 | `Ash` | float | % | No | Ash content |
| 89 | `Gluten` | float | % | No | Gluten content |
| 90 | `Fiber` | float | % | No | Crude fiber content |
| 91 | `Carbohydrates` | float | % | No | Carbohydrate content |
| 92 | `Fat` | float | % | No | Fat/oil content |
| 93 | `Nitrogen_Content` | float | % | No | Grain nitrogen content |
| 94 | `Phosphorus_Content` | float | % | No | Grain phosphorus content |
| 95 | `Potassium_Content` | float | % | No | Grain potassium content |
| 96 | `Iron_Content` | float | ppm | No | Grain iron content |
| 97 | `Copper_Content` | float | ppm | No | Grain copper content |
| 98 | `Zinc_Content` | float | ppm | No | Grain zinc content |
| 99 | `Manganese_Content` | float | ppm | No | Grain manganese content |
| 100 | `Sulphur_Content` | float | % | No | Grain sulphur content |

**Validation:** All percentages 0–100%. All concentrations ≥ 0. `Protein + Ash + Gluten + Fiber + Carbohydrates + Fat ≤ 100%` (approximate).

---

## J. ML Target Variables (5 columns)

Prediction target variables for supervised learning.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 101 | `Target_Yield` | float | kg/ha | No | Primary yield prediction target |
| 102 | `Target_Fertilizer` | float | kg/ha | No | Fertilizer dose prediction target |
| 103 | `Target_Nitrogen` | float | kg/ha | No | Nitrogen requirement target |
| 104 | `Target_Phosphorus` | float | kg/ha | No | Phosphorus requirement target |
| 105 | `Target_Potassium` | float | kg/ha | No | Potassium requirement target |

**Validation:** All targets ≥ 0.

---

## K. Engineered Features (16 columns)

Domain-specific derived features computed from raw measurements.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 106 | `Growing_Degree_Days` | float | °C·day | No | Accumulated thermal time above base (10°C) |
| 107 | `Heat_Units` | float | °C·day | No | Heat accumulation using Tmax |
| 108 | `Harvest_Index_Calc` | float | ratio | No | Computed harvest index |
| 109 | `Nitrogen_Use_Efficiency` | float | kg/kg | No | Yield per unit N applied |
| 110 | `Water_Use_Efficiency` | float | kg/ha·mm | No | Yield per unit rainfall |
| 111 | `Rainfall_Anomaly` | float | mm | No | Deviation from mean rainfall |
| 112 | `Stress_Index` | float | ratio | No | Thermal stress relative to optimum (25°C) |
| 113 | `Disease_Risk_Index` | float | 0–1 | No | Categorical disease risk score |
| 114 | `Yield_per_Plant` | float | g/plant | No | Per-plant yield estimate |
| 115 | `Yield_per_Plot_Calc` | float | g/plot | No | Direct plot yield |
| 116 | `Yield_per_Hectare_Calc` | float | kg/ha | No | Extrapolated hectare yield |
| 117 | `Temp_x_Rainfall` | float | °C·mm | No | Temperature–rainfall interaction |
| 118 | `N_x_P` | float | (kg/ha)² | No | Nitrogen–phosphorus interaction |
| 119 | `Temp_ squared` | float | °C² | No | Quadratic temperature term |
| 120 | `Rainfall_7d_MA` | float | mm | No | 7-day rainfall moving average |
| 121 | `Temp_7d_MA` | float | °C | No | 7-day temperature moving average |

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
| 122 | `Feature_Available_Before_Prediction` | integer | — | No | 1 = available before harvest, 0 = post-harvest variable |

**Leakage classification:**
- **POST_HARVEST** (leaked): Protein, Ash, Gluten, Fiber, Carbohydrates, Fat, Nitrogen_Content, Phosphorus_Content, Potassium_Content, Iron_Content, Copper_Content, Zinc_Content, Manganese_Content, Sulphur_Content, 100_Seed_Weight, Root_Weight, Pod_Weight, Harvest_Index, Biomass_Yield, Yield_per_Plot, Yield_per_Acre, Yield_per_Hectare, Fruit_Number, Fruit_Weight, Fruit_Diameter_mm, Spike_Length, Seeds_per_Spike, Target_* variables
- **PRE_HARVEST_MEASUREMENT** (safe): Leaf_Number, Tillers, Branches, Nodes, Flowers, Plant_Height_30/60/90_cm, Leaf_Area_30/60/90_cm2, SPAD, Moisture_Content
- **AVAILABLE_BEFORE_PREDICTION** (safe): All other columns

---

## M. Encoded Variables (6 columns)

Numeric encodings of categorical variables for ML consumption.

| # | Column | Type | Units | Required | Description |
|---|--------|------|-------|----------|-------------|
| 123 | `Crop_Code` | integer | — | No | Numeric code for Crop |
| 124 | `Season_Code` | integer | — | No | Numeric code for Season |
| 125 | `Variety_Code` | integer | — | No | Numeric code for Variety |
| 126 | `Fertilizer_Code` | integer | — | No | Numeric code for Fertilizer_Name |
| 127 | `Soil_Texture_Code` | integer | — | No | Numeric code for Soil_Texture |
| 128 | `Country_Code` | integer | — | No | Numeric code for Country |

**Encoding method:** LabelEncoder (integer encoding, 0 to n_classes-1). Original values preserved in separate columns. Encoding map stored in `Encoding_Map.csv`.

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
