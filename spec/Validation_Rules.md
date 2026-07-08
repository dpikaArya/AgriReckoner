# UAMS Validation Rules

**Version:** 1.0  
**Scope:** Data quality constraints for all 128 UAMS columns

---

## Rule Categories

| Category | Description | Severity |
|----------|-------------|----------|
| R1 — Range | Numeric values outside physically possible ranges | ERROR |
| R2 — Consistency | Logical inconsistencies between related columns | ERROR |
| R3 — Completeness | Missing critical identifiers or required fields | WARNING |
| R4 — Uniqueness | Duplicate observations or identifiers | WARNING |
| R5 — Type | Data type mismatches or mixed types in a column | WARNING |
| R6 — Statistical | Statistical outliers beyond expected thresholds | INFO |

---

## R1 — Range Validation Rules

| Column | Rule | Valid Range | Rationale |
|--------|------|-------------|-----------|
| Soil_pH | Must be between 0 and 14 | [0.0, 14.0] | pH scale definition |
| EC | Must be non-negative | [0, ∞) | Physical constraint |
| Temperature_Max | Must be between -20 and 60 | [-20, 60] °C | Earth temperature limits |
| Temperature_Min | Must be between -20 and 60 | [-20, 60] °C | Earth temperature limits |
| Average_Temperature | Must be between -20 and 60 | [-20, 60] °C | Earth temperature limits |
| Humidity | Must be between 0 and 100 | [0.0, 100.0] % | Physical definition |
| Rainfall | Must be non-negative | [0, ∞) mm | Physical constraint |
| Latitude | Must be between -90 and 90 | [-90.0, 90.0] | Geographic definition |
| Longitude | Must be between -180 and 180 | [-180.0, 180.0] | Geographic definition |
| Altitude | Should be non-negative | [0, ∞) m | Sea level reference |
| Year | Must be between 1950 and current | [1950, current] | Publication history |
| Yield_per_Plot | Must be non-negative | [0, ∞) g | Physical constraint |
| Yield_per_Acre | Must be non-negative | [0, ∞) kg/acre | Physical constraint |
| Yield_per_Hectare | Must be non-negative | [0, ∞) kg/ha | Physical constraint |
| Harvest_Index | Must be between 0 and 1 | [0.0, 1.0] | Definitional |
| Dose | Must be non-negative | [0, ∞) kg/ha | Physical constraint |
| Replications | Must be ≥ 2 | [2, ∞) | Experimental design |
| Soil_Texture_Code | Must be non-negative | [0, ∞) | Encoding definition |

## R2 — Consistency Rules

| Rule | Condition | Error Message |
|------|-----------|---------------|
| C1 | `Temperature_Max ≥ Temperature_Min` | "Tmax < Tmin for {row}" |
| C2 | `Plant_Height_30_cm ≤ Plant_Height_60_cm ≤ Plant_Height_90_cm ≤ Plant_Height_cm` | "Monotonic height violation" |
| C3 | `Leaf_Area_30_cm2 ≤ Leaf_Area_60_cm2 ≤ Leaf_Area_90_cm2 ≤ Leaf_Area_cm2` | "Monotonic leaf area violation" |
| C4 | `Spacing_Row ≥ Spacing_Plant` | "Row spacing < plant spacing" |
| C5 | `Protein + Ash + Gluten + Fiber + Carbohydrates + Fat ≤ 100` (approximate) | "Quality sum exceeds 100%" |
| C6 | `Yield_per_Hectare ≥ Yield_per_Plot / Plot_Size × 10000` (approximate) | "Yield scaling mismatch" |
| C7 | `Latitude = 0 AND Longitude = 0` → warning | "Coordinates at null island" |

## R3 — Completeness Rules

| Rule | Column | Action |
|------|--------|--------|
| M1 | Crop | WARNING if missing (required for schema) |
| M2 | Paper_ID, DOI, Experiment_ID, Plot_ID, Sample_ID | WARNING if >10% missing (identifier columns) |
| M3 | All columns | Missing values are preserved as NaN; no automatic imputation |
| M4 | Target variables (Target_Yield, etc.) | INFO if missing (only needed for supervised learning) |

## R4 — Uniqueness Rules

| Rule | Check | Action |
|------|-------|--------|
| U1 | Exact duplicate rows | Remove duplicates (keep first occurrence) |
| U2 | Duplicate column names | Collapse duplicates (keep last occurrence) |
| U3 | Near-duplicate rows (e.g., all non-ID columns identical) | WARNING — flag for manual review |

## R5 — Type Validation

| Rule | Check | Action |
|------|-------|--------|
| T1 | Column has mixed numeric/string types | WARNING — report column |
| T2 | Date columns are parseable | WARNING — report unparseable dates |
| T3 | Encoded columns contain only integers | WARNING — report non-integer codes |

## R6 — Statistical Outlier Detection

Method: **Interquartile Range (IQR)** rule

```
IQR = Q3 - Q1
Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
Any value outside [lower, upper] flagged as outlier.
```

Applied to all numeric columns. Results are informational (not filtered).

---

## Validation Reports

Two reports are generated:

### Quality Report (`Quality_Report.md`)

- Summary counts by category
- Detailed list of each issue found
- Per-column issue breakdown

### Validation Report (`Validation_Report.csv`)

| Column | Data_Type | Non_Null_Count | Null_Count | Null_Pct | Unique_Values | Issues |
|--------|-----------|----------------|------------|----------|---------------|--------|

---

## Reference: Impossible Value Thresholds

| Variable | Impossible Threshold | Reasoning |
|----------|---------------------|-----------|
| Soil_pH > 14 | Extreme alkalinity not found in agricultural soils | pH scale maximum |
| Soil_pH < 0 | Extreme acidity not found in agricultural soils | pH scale minimum |
| Temperature > 60°C | No agricultural region exceeds this | World record: 56.7°C |
| Temperature < -20°C | Winter wheat survival limit | Crop physiology |
| Humidity > 100% | Physical impossibility | Saturation = 100% |
| Humidity < 0% | Physical impossibility | Definitional |
| Rainfall < 0 mm | Physical impossibility | Definitional |
| EC < 0 dS/m | Physical impossibility | Definitional |
| All biomass/yield < 0 | Physical impossibility | Definitional |
