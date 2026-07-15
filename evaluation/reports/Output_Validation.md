# Output Validation Report
Generated: 2026-07-04 22:30:11

## Schema XLSX Validation
| Check | Result |
|-------|--------|
| File exists | ✅ |
| Sheets present | 15/15 |

### Sheet Status
| Sheet | Present |
|-------|---------|
| ✅ | 01_Metadata |
| ✅ | 02_Field_Profile |
| ✅ | 03_Crop_Profile |
| ✅ | 04_Soil_Profile |
| ✅ | 05_Weather_TimeSeries |
| ✅ | 06_Management_Events |
| ✅ | 07_Plant_Observations |
| ✅ | 08_Remote_Sensing |
| ✅ | 09_Sensor_Data |
| ✅ | 10_Laboratory_Analysis |
| ✅ | 11_Derived_Features |
| ✅ | 12_Targets |
| ✅ | 13_Data_Quality |
| ✅ | 14_Feature_Dictionary |
| ✅ | 15_Evidence_Metadata |

### Missing Sheets (0)

## Column Validation
- Duplicated columns: 0
- Variables with missing ontology: 3
- Inconsistent units detected: 0
- Leakage issues: 0
- Model ready: ✅


### Variables Missing Ontology Mappings (3)
- Temperature_Max_7d_MA
- Temperature_Min_7d_MA
- Average_Temperature_7d_MA

### Leakage Issues (0)

## Overall Validation
**Output Validation: ✅ PASS**

## Recommendations
1. Add all 15 required sheets to the XLSX workbook
2. Remove duplicated columns from the output
3. Complete ontology mappings for all unmapped variables
4. Add Feature_Available_Before_Prediction flag
5. Ensure consistent units across all sheets
