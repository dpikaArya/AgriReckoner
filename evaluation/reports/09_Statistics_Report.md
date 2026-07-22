# Stage 09: Statistical Diagnostics Report
Generated: 2026-07-20 17:17:16

## Summary
- Dataset shape: 43 rows × 138 columns
- Numeric columns: 132
- Missing values: 5572/5934 (93.9%)
- High correlation pairs (|r| > 0.8): 7
- High VIF features: 7
- Highly skewed features: 4
- High kurtosis features: 4
- Non-normal features: 5

## Missing Values (Top 20)
| Column | Missing | % |
|--------|---------|---|
| Journal | 43 | 100.0% |
| Year | 43 | 100.0% |
| Authors | 43 | 100.0% |
| Country | 43 | 100.0% |
| Scientific_Name | 43 | 100.0% |
| Variety | 43 | 100.0% |
| Season | 43 | 100.0% |
| Growth_Duration_Days | 43 | 100.0% |
| Growth_Stage | 43 | 100.0% |
| Plot_Size | 43 | 100.0% |
| Spacing_Row | 43 | 100.0% |
| Spacing_Plant | 43 | 100.0% |
| Sample_Size | 43 | 100.0% |
| Location | 43 | 100.0% |
| State | 43 | 100.0% |
| Site | 43 | 100.0% |
| Latitude | 43 | 100.0% |
| Longitude | 43 | 100.0% |
| Altitude | 43 | 100.0% |
| Temperature_Max | 43 | 100.0% |

## High Correlation Pairs (7)
- Rainfall ↔ Organic_Carbon: r=0.981
- Rainfall ↔ Iron: r=0.927
- Soil_pH ↔ Yield_per_Hectare: r=1.000
- EC ↔ Organic_Carbon: r=1.000
- EC ↔ Potassium: r=1.000
- EC ↔ Iron: r=0.971
- EC ↔ Yield_per_Hectare: r=1.000

## Skewness (|skew| > 1: 4)
- Replications: 6.557
- Soil_pH: 3.740
- EC: 2.125
- Organic_Carbon: 1.998

## Kurtosis (|kurt| > 3: 4)
- Replications: 43.000
- Soil_pH: 13.993
- EC: 3.228
- Organic_Carbon: 3.992

## Non-Normal Features (Shapiro-Wilk p<0.05: 5)
- Replications
- Rainfall
- Soil_pH
- EC
- Organic_Carbon

## Bottlenecks & Recommendations
1. **Missing values**: Impute or drop columns with >10% missing before modeling
2. **High correlation**: Remove or combine highly correlated features
3. **High VIF**: Apply dimensionality reduction or regularization
4. **Skewness/Kurtosis**: Apply power transforms (Box-Cox, Yeo-Johnson)
5. **Normality**: Consider non-parametric tests or robust methods
