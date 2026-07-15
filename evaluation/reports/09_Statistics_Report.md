# Stage 09: Statistical Diagnostics Report
Generated: 2026-07-04 22:30:10

## Summary
- Dataset shape: 3 rows × 131 columns
- Numeric columns: 129
- Missing values: 324/393 (82.4%)
- High correlation pairs (|r| > 0.8): 109
- High VIF features: 30
- Highly skewed features: 0
- High kurtosis features: 0
- Non-normal features: 0

## Missing Values (Top 20)
| Column | Missing | % |
|--------|---------|---|
| Paper_ID | 3 | 100.0% |
| DOI | 3 | 100.0% |
| Journal | 3 | 100.0% |
| Year | 3 | 100.0% |
| Authors | 3 | 100.0% |
| Country | 3 | 100.0% |
| Scientific_Name | 3 | 100.0% |
| Variety | 3 | 100.0% |
| Growth_Duration_Days | 3 | 100.0% |
| Design | 3 | 100.0% |
| Replications | 3 | 100.0% |
| Spacing_Row | 3 | 100.0% |
| Spacing_Plant | 3 | 100.0% |
| Sample_Size | 3 | 100.0% |
| Location | 3 | 100.0% |
| State | 3 | 100.0% |
| Site | 3 | 100.0% |
| Latitude | 3 | 100.0% |
| Longitude | 3 | 100.0% |
| Altitude | 3 | 100.0% |

## High Correlation Pairs (109)
- Plot_Size ↔ Temperature_Max: r=0.866
- Plot_Size ↔ Temperature_Min: r=0.961
- Plot_Size ↔ Rainfall: r=0.866
- Plot_Size ↔ Soil_pH: r=0.866
- Plot_Size ↔ EC: r=0.918
- Plot_Size ↔ Nitrogen: r=0.945
- Plot_Size ↔ Yield_per_Plot: r=0.866
- Plot_Size ↔ Growing_Degree_Days: r=0.912
- Plot_Size ↔ Heat_Units: r=0.866
- Plot_Size ↔ Rainfall_Anomaly: r=0.866
- Plot_Size ↔ Yield_per_Plot_Calc: r=0.866
- Plot_Size ↔ Temp_ squared: r=0.891
- Plot_Size ↔ Crop_Code: r=1.000
- Plot_Size ↔ Season_Code: r=1.000
- Plot_Size ↔ Temperature_Max_7d_MA: r=1.000
- Plot_Size ↔ Temperature_Min_7d_MA: r=0.918
- Temperature_Max ↔ Temperature_Min: r=0.971
- Temperature_Max ↔ Soil_pH: r=1.000
- Temperature_Max ↔ EC: r=0.993
- Temperature_Max ↔ Nitrogen: r=0.982

## Skewness (|skew| > 1: 0)

## Kurtosis (|kurt| > 3: 0)

## Non-Normal Features (Shapiro-Wilk p<0.05: 0)

## Bottlenecks & Recommendations
1. **Missing values**: Impute or drop columns with >10% missing before modeling
2. **High correlation**: Remove or combine highly correlated features
3. **High VIF**: Apply dimensionality reduction or regularization
4. **Skewness/Kurtosis**: Apply power transforms (Box-Cox, Yeo-Johnson)
5. **Normality**: Consider non-parametric tests or robust methods
