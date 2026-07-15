# Fuzzy Logic System
Generated: 2026-07-15T14:20:27.614927
Input variables: 10
Rules: 16

## Membership Functions

### Nitrogen
  - low: (0, 30)
  - medium: (30, 60, 100)
  - high: (80, 120)

### Phosphorus
  - low: (0, 10)
  - medium: (10, 25, 40)
  - high: (30, 50)

### Potassium
  - low: (0, 80)
  - medium: (80, 150, 250)
  - high: (180, 300)

### Soil_pH
  - acidic: (0, 5.5)
  - neutral: (5.5, 6.5, 7.5, 8.0)
  - alkaline: (7.5, 14)

### Rainfall
  - low: (0, 400)
  - medium: (400, 750, 1100)
  - high: (900, 1200)

### Temperature_Max
  - cool: (0, 20)
  - moderate: (20, 25, 32, 35)
  - hot: (32, 38)

### Organic_Carbon
  - low: (0, 0.4)
  - medium: (0.4, 0.75, 1.0)
  - high: (0.8, 1.2)

## Rules
  - [10] IF Nitrogen=low; Rainfall=high; Organic_Carbon=low THEN Nitrogen: increase, Potassium: maintain
  - [9] IF Soil_pH=alkaline; Zinc=low THEN Zinc: apply_foliar_spray, Nitrogen: maintain
  - [8] IF Yield_Prediction=high; Potassium=medium THEN Potassium: maintain, Nitrogen: maintain
  - [8] IF Nitrogen=low; Rainfall=low THEN Nitrogen: increase
  - [9] IF Phosphorus=low; Soil_pH=acidic THEN Phosphorus: increase, Nitrogen: maintain
  - [7] IF Potassium=low THEN Potassium: increase
  - [9] IF Temperature_Max=hot; Rainfall=low THEN Nitrogen: maintain, Potassium: maintain
  - [6] IF Temperature_Max=cool; Rainfall=high THEN Nitrogen: maintain
  - [7] IF Organic_Carbon=low THEN Nitrogen: increase
  - [5] IF Soil_pH=alkaline; Nitrogen=high THEN Nitrogen: maintain, Zinc: monitor
  - [3] IF Nitrogen=medium; Phosphorus=medium; Potassium=medium; Soil_pH=neutral; Rainfall=medium THEN Nitrogen: maintain, Phosphorus: maintain, Potassium: maintain
  - [7] IF Yield_Prediction=low; Nitrogen=high THEN Nitrogen: review
  - [8] IF Zinc=low; Rainfall=high THEN Zinc: apply_foliar_spray
  - [4] IF Yield_Prediction=high; Nitrogen=medium; Phosphorus=medium; Potassium=medium; Soil_pH=neutral THEN Nitrogen: maintain, Phosphorus: maintain, Potassium: maintain
  - [4] IF Nitrogen=medium; Phosphorus=medium; Potassium=medium; Soil_pH=neutral; Rainfall=high THEN Nitrogen: maintain, Phosphorus: maintain, Potassium: maintain
  - [4] IF Nitrogen=medium; Phosphorus=medium; Potassium=medium; Soil_pH=neutral; Rainfall=low THEN Nitrogen: maintain, Phosphorus: maintain, Potassium: maintain