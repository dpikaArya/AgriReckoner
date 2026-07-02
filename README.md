# Universal Agricultural Schema Generator (Merge Agent)

An AI-powered pipeline component that reads multiple crop master datasets, standardizes column names into a single unified schema, and produces ready-to-use machine learning datasets.

Part of an Agentic AI pipeline for **Crop Recommendation and Yield Prediction**.

## Features

- **Automatic schema alignment** — maps disparate column names (e.g. `Tmax_C`, `Available_N`, `PlantHeight_120_cm`) to a standardized 105-column universal schema
- **Multi-timepoint handling** — intermediate growth stages (30/60/90 DAS) are preserved in timepoint-specific columns; final stage maps to the base column
- **Duplicate resolution** — when multiple source columns map to the same standard name, the last/highest timepoint is retained
- **Missing data preserved** — unavailable values remain `NaN` (not zero-filled)
- **Data validation** — checks for impossible values (negative biomass, out-of-range pH, etc.)
- **Future-proof** — any new `.xlsx` dataset placed in `data/master_datasets/` is merged automatically without code changes

## Requirements

- Python 3.9+
- pandas >= 2.0
- numpy >= 1.24
- openpyxl >= 3.1

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place crop master dataset Excel files in `data/master_datasets/`
2. Run the generator:

```bash
python universal_schema_generator.py
```

Output files are written to `outputs/`:

| File | Description |
|---|---|
| `Universal_Agricultural_Schema.xlsx` | Merged dataset with all standardized columns |
| `MachineLearning_Dataset.csv` | Ready-to-use CSV for ML model training |
| `Schema_Metadata.json` | Full schema definition, column groups, and mapping history |
| `Validation_Report.xlsx` | Validation check results |

## Schema Overview (105 columns)

| Group | Columns |
|---|---|
| A. Paper Metadata | Paper_ID, DOI, Journal, Year, Authors, Country |
| B. Crop Information | Crop, Scientific_Name, Variety, Season, Growth_Duration_Days |
| C. Experimental Design | Design, Replications, Plot_Size, Spacing_Row, Spacing_Plant, Sample_Size, Location, State, Site |
| D. Environment | Latitude, Longitude, Altitude, Temperature_Max/Min/Avg, Rainfall, Humidity |
| E. Soil Properties | Soil_pH, EC, Organic_Carbon/Matter, N, P, K, S, Fe, Cu, Mn, Zn, Ca, Mg, B, Mo |
| F. Fertilizer | Treatment, Fertilizer_Name, Organic_Fertilizer, Biofertilizer, Dose, Application_Method/Interval |
| G. Growth Parameters | Shoot/Root Length, Plant Height, Biomass, Leaf Area, Leaf Number, Tillers, SPAD, Dry Matter, Branches, Flowers |
| H. Yield | Yield_per_Plot/Acre/Hectare, Fruit Number/Weight/Diameter, Spike Length, Seeds/Spike, 100-Seed Weight |
| I. Grain Quality | Protein, Ash, Gluten, Fiber, Carbohydrates, Fat, N/P/K/Fe/Cu/Zn Content |
| J. ML Targets | Target_Yield, Target_Fertilizer, Target_N/P/K |

## Column Name Standardization

The mapper recognises hundreds of synonymous column name variants:

| Source examples | Standard name |
|---|---|
| `Tmax_C`, `Temp_Max`, `Max_Temperature` | `Temperature_Max` |
| `Available_N`, `Nitrogen`, `N`, `Nitrogen_kg_ha` | `Nitrogen` |
| `PlantHeight_120_cm`, `Plant_Height`, `Plant_height_cm` | `Plant_Height_cm` |
| `Chlorophyll_SPAD`, `SPAD`, `Chlorophyll_Content` | `SPAD` |
| `Organic_C`, `Organic_Carbon`, `Organic_Carbon_%` | `Organic_Carbon` |
| `Yield_Plot`, `Yield_per_Plot_g`, `Fresh_Weight_g` | `Yield_per_Plot` |

## Adding a New Crop Dataset

Simply drop an `.xlsx` file into `data/master_datasets/` and re-run the script. The generator will:

1. Auto-detect the crop name from the filename or data
2. Map its columns via the pattern-and-dictionary matcher
3. Add any new standard columns automatically
4. Merge into the existing schema

No code changes required.

## Privacy

This repository contains **only** reusable source code and configuration. Actual crop datasets, generated outputs, and experimental results must be placed in `data/` and `outputs/` which are excluded by `.gitignore`.
