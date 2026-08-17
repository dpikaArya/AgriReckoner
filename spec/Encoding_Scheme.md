# UAMS Encoding Scheme

**Version:** 1.0  
**Method:** Label Encoding (integer encoding)  
**Originals preserved:** Yes — original categorical columns are kept alongside encoded versions

---

## Encoding Specification

All categorical variables are encoded using `sklearn.preprocessing.LabelEncoder`. Encoded values are zero-based integers (0 to n-1 where n = number of unique classes). The encoding map is stored in `Encoding_Map.csv`.

| Original Column | Encoded Column | Description |
|-----------------|----------------|-------------|
| `Crop` | `Crop_Code` | Numeric code for crop type |
| `Season` | `Season_Code` | Numeric code for growing season |
| `Variety` | `Variety_Code` | Numeric code for variety/cultivar |
| `Fertilizer_Name` | `Fertilizer_Code` | Numeric code for fertilizer type |
| `Soil_Texture` | `Soil_Texture_Code` | Numeric code for soil texture class |
| `Country` | `Country_Code` | Numeric code for country |

---

## Encoding Map Format

```
Encoding_Map.csv
├── original_column: source column name
├── encoded_column: target column name
├── original_value: original categorical value
└── encoded_value: numeric code
```

### Example

| original_column | encoded_column | original_value | encoded_value |
|-----------------|----------------|----------------|---------------|
| Crop | Crop_Code | Wheat | 0 |
| Crop | Crop_Code | Rice | 1 |
| Crop | Crop_Code | Maize | 2 |
| Season | Season_Code | Rabi | 0 |
| Season | Season_Code | Kharif | 1 |
| Season | Season_Code | Spring | 2 |

---

## Supported Crops Codes

Standard codes for common crops (auto-detected from data, not predefined):

| Crop | Typical Code | Notes |
|------|-------------|-------|
| Wheat | 0 | *Triticum aestivum* |
| Rice | 1 | *Oryza sativa* |
| Maize | 2 | *Zea mays* |
| Gram | 3 | *Cicer arietinum* |
| BlackWheat | 4 | *Fagopyrum* |
| Carrot | 5 | *Daucus carota* |
| BellPepper | 6 | *Capsicum annuum* |
| Spinach | 7 | *Spinacia oleracea* |

---

## Supported Season Codes

| Season | Typical Code | Description |
|--------|-------------|-------------|
| Rabi | 0 | Winter season (Oct–Mar) |
| Kharif | 1 | Monsoon season (Jun–Oct) |
| Zaid | 2 | Summer season (Apr–Jun) |
| Spring | 3 | Spring planting |
| Autumn | 4 | Autumn planting |
| Perennial | 5 | Multi-year crop |

---

## Implementation Notes

1. **Deterministic**: Encodings are consistent within a single pipeline run
2. **Map persistence**: `Encoding_Map.csv` enables decoding back to original values
3. **Unknown values**: Unseen categories at prediction time will raise errors; handle via a strategy like "unknown" class or most-frequent fallback
4. **Original preserved**: Raw categorical columns are kept for interpretability and alternative encoding strategies (one-hot, target encoding, etc.)
5. **ML readiness**: Encoded integer columns are directly consumable by tree-based models; linear models and neural networks require further preprocessing (e.g., one-hot or embedding)
