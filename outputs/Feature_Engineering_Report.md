# Feature Engineering Report
Generated: 2026-07-07T22:01:32.232660
Features added: 4

## Features Added
- **Temp_squared**: Quadratic temperature term for non-linear modeling
- **Temperature_Max_7d_MA**: Engineered feature
- **Temperature_Min_7d_MA**: Engineered feature
- **Average_Temperature_7d_MA**: Engineered feature

## Feature Categories
- **Thermal**: Growing_Degree_Days, Heat_Units, Stress_Index
- **Efficiency**: Nitrogen_Use_Efficiency, Water_Use_Efficiency, Harvest_Index_Calc
- **Interaction**: Temp_x_Rainfall, N_x_P
- **Polynomial**: Temp_squared
- **Rolling**: *_7d_MA (moving averages)
- **Yield**: Yield_per_Plant, Yield_per_Plot_Calc, Yield_per_Hectare_Calc
- **Risk**: Disease_Risk_Index, Rainfall_Anomaly