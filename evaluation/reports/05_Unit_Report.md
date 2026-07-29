# Stage 05: Unit Harmonization Report
Generated: 2026-07-28 15:48:24

## Summary
- Unit conversions applied: 0
- Conversion accuracy: 0.0%
- Unsupported units estimated: 10
- Columns with unit suffixes: 16
- Conversion consistency: 0.0%

## Conversions Applied

## Unit Columns Found (16)
- Shoot_Length_cm
- Root_Length_cm
- Plant_Height_cm
- Shoot_Biomass_g
- Root_Biomass_g
- Leaf_Area_cm2
- Root_Diameter_mm
- SPAD
- Stem_Diameter_mm
- Plant_Height_30_cm
- Plant_Height_60_cm
- Plant_Height_90_cm
- Leaf_Area_30_cm2
- Leaf_Area_60_cm2
- Leaf_Area_90_cm2
- Fruit_Diameter_mm

## Bottlenecks & Recommendations
1. **Missing conversions**: Extend the unit conversion dictionary with more crop-specific units
2. **Unsupported units**: Add support for local units (e.g., quintal, bigha, guntha)
3. **Consistency**: Ensure all converted values are rounded to consistent precision
4. Add unit validation against the UAMS expected unit registry
