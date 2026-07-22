# Stage 08: Leakage Detection Report
Generated: 2026-07-20 17:17:08

## Summary
- Harvest/post-harvest variables in data: 17
- Leakage detected: 0
- Leakage prevented (flagged): 0
- False positives: 0
- Leakage detection rate: 0.0%

## Variables Flagged as Leakage (0)

## Harvest Variables in Dataset (Potential Leakage: 17)
- Yield_per_Plot
- Yield_per_Acre
- Yield_per_Hectare
- Fruit_Number
- Fruit_Weight
- Fruit_Diameter_mm
- Spike_Length
- Seeds_per_Spike
- 100_Seed_Weight
- Root_Weight
- Pod_Weight
- Harvest_Index
- Biomass_Yield
- Protein
- Ash
- Gluten
- Fiber

## Bottlenecks & Recommendations
1. **Undetected leakage**: Ensure all post-harvest measurement variables are flagged
2. **Target leakage**: Verify that target variables are not used as features
3. **Temporal leakage**: Add date-based leakage checks for time-series splits
4. Implement proper train/test separation with leakage awareness
5. Add cross-validation strategy that respects the leakage flag
