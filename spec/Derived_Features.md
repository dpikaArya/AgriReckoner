# UAMS Derived Features — Specification

**Version:** 1.0  
**Engineered by:** Feature Engineering Agent (Agent 06)  
**Total derived features:** 16

---

## Feature Inventory

| # | Feature | Category | Inputs | Output Units |
|---|---------|----------|--------|-------------|
| 1 | Growing_Degree_Days | Thermal | Tmax, Tmin | °C·day |
| 2 | Heat_Units | Thermal | Tmax | °C·day |
| 3 | Harvest_Index_Calc | Ratio | Yield, Biomass | dimensionless |
| 4 | Nitrogen_Use_Efficiency | Efficiency | Yield, N | kg/kg |
| 5 | Water_Use_Efficiency | Efficiency | Yield, Rainfall | kg/ha·mm |
| 6 | Rainfall_Anomaly | Statistical | Rainfall | mm |
| 7 | Stress_Index | Stress | Tmax, Tmin | dimensionless |
| 8 | Disease_Risk_Index | Risk | Humidity, Tmax | categorical (0, 0.5, 1) |
| 9 | Yield_per_Plant | Yield | Fruit_Number, Fruit_Weight | g/plant |
| 10 | Yield_per_Plot_Calc | Yield | Yield_per_Plot | g/plot |
| 11 | Yield_per_Hectare_Calc | Yield | Yield_per_Plot, Plot_Size | kg/ha |
| 12 | Temp_x_Rainfall | Interaction | Tmean, Rainfall | °C·mm |
| 13 | N_x_P | Interaction | Nitrogen, Phosphorus | (kg/ha)² |
| 14 | Temp_ squared | Polynomial | Tmean | °C² |
| 15 | Rainfall_7d_MA | Rolling | Rainfall | mm |
| 16 | Temp_7d_MA | Rolling | Tmean | °C |

---

## 1. Growing Degree Days (GDD)

Accumulated thermal time above a base temperature. Measures heat units available for crop development.

**Formula:**
```
GDD = max(0, Tmean - Tbase)
```
where:
- `Tmean = (Tmax + Tmin) / 2`
- `Tbase = 10°C` (base temperature for most temperate crops)

**Units:** °C·day  
**Range:** ≥ 0  
**Use cases:** Phenology prediction, maturity estimation, thermal time modeling

**Crop-specific base temperatures:**

| Crop | Tbase (°C) |
|------|-----------|
| Wheat | 0–5 |
| Rice | 10 |
| Maize | 10 |
| Cotton | 15 |
| Potato | 7 |
| Tomato | 10 |

Note: UAMS uses Tbase = 10°C as default for broad applicability.

---

## 2. Heat Units (HU)

Alternative thermal metric using only maximum temperature.

**Formula:**
```
HU = max(0, Tmax - Tbase)
```
where `Tbase = 10°C`

**Units:** °C·day  
**Range:** ≥ 0

---

## 3. Harvest Index (Calculated)

Ratio of economic yield to total biological yield.

**Formula:**
```
Harvest_Index_Calc = Yield / Biomass  (if Biomass > 0)
```

**Units:** dimensionless (ratio)  
**Range:** 0 to 1 (typically 0.3–0.6 for cereal crops)  
**Use cases:** Yield partitioning efficiency, crop model calibration

---

## 4. Nitrogen Use Efficiency (NUE)

Amount of yield produced per unit of nitrogen applied.

**Formula:**
```
NUE = Yield / N_applied  (if N_applied > 0)
```

**Units:** kg yield per kg N  
**Range:** 10–100 kg/kg (typical)  
**Use cases:** Fertilizer recommendation, environmental impact assessment

---

## 5. Water Use Efficiency (WUE)

Yield produced per unit of water received.

**Formula:**
```
WUE = Yield / Rainfall  (if Rainfall > 0)
```

**Units:** kg/ha per mm  
**Range:** 5–30 kg/ha·mm (typical)  
**Use cases:** Drought tolerance evaluation, irrigation optimization

---

## 6. Rainfall Anomaly

Deviation of observed rainfall from the long-term mean.

**Formula:**
```
Rainfall_Anomaly = Rainfall - mean(Rainfall)
```

**Units:** mm  
**Range:** unbounded, centered at 0  
**Use cases:** Drought/flood detection, year-over-year comparison

---

## 7. Stress Index

Normalized thermal stress index measuring deviation from optimal temperature.

**Formula:**
```
Stress_Index = |Tmean - Topt| / Topt
```
where `Topt = 25°C` (optimal temperature for most crops)

**Units:** dimensionless (ratio)  
**Range:** 0 to ∞ (0 = no stress)  
**Use cases:** Heat stress quantification, climate impact assessment

---

## 8. Disease Risk Index

Categorical indicator of favorable conditions for fungal disease development.

**Formula:**
```
Disease_Risk_Index:
  1.0  if Humidity > 80% AND Tmax > 25°C   (high risk)
  0.5  if Humidity > 60% AND Tmax > 20°C   (moderate risk)
  0.0  otherwise                            (low risk)
```

**Units:** categorical (0.0, 0.5, 1.0)  
**Use cases:** Disease warning systems, fungicide timing decisions

---

## 9. Yield per Plant

Individual plant-level yield estimate.

**Formula:**
```
Yield_per_Plant = Fruit_Number × Fruit_Weight  (if Fruit_Number > 0)
```

**Units:** g/plant  
**Range:** ≥ 0  
**Use cases:** Per-plant productivity assessment

---

## 10. Yield per Plot (Calculated)

Direct plot yield (pass-through).

**Formula:**
```
Yield_per_Plot_Calc = Yield_per_Plot
```

**Units:** g/plot

---

## 11. Yield per Hectare (Calculated)

Extrapolated hectare yield from plot measurements.

**Formula:**
```
Yield_per_Hectare_Calc = Yield_per_Plot / Plot_Size × 10000  (if Plot_Size > 0)
```

**Units:** kg/ha  
**Range:** ≥ 0  
**Use cases:** Standardized yield comparison across experiments

---

## 12. Temperature × Rainfall Interaction

First-order interaction term for regression modeling.

**Formula:**
```
Temp_x_Rainfall = Tmean × Rainfall
```

**Units:** °C·mm  
**Use cases:** Captures combined temperature-moisture effects on yield

---

## 13. Nitrogen × Phosphorus Interaction

First-order interaction between macronutrients.

**Formula:**
```
N_x_P = Nitrogen × Phosphorus
```

**Units:** (kg/ha)²  
**Use cases:** Nutrient interaction effects on crop response

---

## 14. Temperature Squared

Quadratic polynomial term for non-linear modeling.

**Formula:**
```
Temp_ squared = Tmean²
```

**Units:** °C²  
**Use cases:** Captures optimal-temperature response curves

---

## 15. Rainfall 7-day Moving Average

Smoothed rainfall using a 7-day rolling window.

**Formula:**
```
Rainfall_7d_MA = rolling_mean(Rainfall, window=7, min_periods=1)
```

**Units:** mm  
**Use cases:** Soil moisture estimation, drought indices

---

## 16. Temperature 7-day Moving Average

Smoothed temperature using a 7-day rolling window.

**Formula:**
```
Temp_7d_MA = rolling_mean(Tmean, window=7, min_periods=1)
```

**Units:** °C  
**Use cases:** Heat wave detection, seasonal trend analysis

---

## Feature Selection Guidance

| ML Task | Recommended Features |
|---------|---------------------|
| Yield prediction | All engineered features |
| Fertilizer recommendation | GDD, NUE, N_x_P, Temp_x_Rainfall |
| Stress tolerance | Stress_Index, Disease_Risk_Index, WUE |
| Phenology | Growing_Degree_Days, Heat_Units |
| Classification | Stress_Index, Disease_Risk_Index, Rainfall_Anomaly |
| Time series | Temp_7d_MA, Rainfall_7d_MA |
| Causal inference | Temp_x_Rainfall, N_x_P (as instruments) |

## Notes

- All derived features are computed from available raw measurements
- Missing input values result in NaN for the derived feature
- Features are not scaled; downstream models should apply appropriate scaling
- Interaction features (12, 13) and polynomial features (14) improve non-linear model performance
- Rolling features (15, 16) are computed with `min_periods=1` to avoid NaN at sequence start
