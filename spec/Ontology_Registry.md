# UAMS Ontology Registry

**Version:** 1.0  
**Sources:** AGROVOC, Crop Ontology (CGIAR), Plant Ontology (PO), Environment Ontology (ENVO), FAO Vocabulary

Maps every UAMS variable to its corresponding term in international agricultural ontologies.

---

## Ontology Source Reference

| Source | Prefix | URL | Description |
|--------|--------|-----|-------------|
| AGROVOC | AGROVOC | https://agrovoc.fao.org/ | FAO multilingual agricultural thesaurus |
| Crop Ontology | CO | https://cropontology.org/ | CGIAR crop trait ontology |
| Plant Ontology | PO | https://obolibrary.org/obo/po/ | Plant anatomical and morphological terms |
| Environment Ontology | ENVO | https://obolibrary.org/obo/envo/ | Environmental terms |
| FAO Vocabulary | FAO | https://www.fao.org/faoterm/ | FAO terminology |

---

## Complete Variable-to-Ontology Mapping

### A. Paper Metadata

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Paper_ID | — | Paper Identifier | reference ID | — |
| DOI | — | Digital Object Identifier | — | — |
| Journal | — | Journal | publication | — |
| Year | — | Publication Year | — | — |
| Authors | — | Authors | — | — |
| Country | AGROVOC:c_1625 | Country | nation, region | AGROVOC |

### B. Crop Information

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Crop | AGROVOC:c_2556; CO:320:0000000 | Crop | crop type, cultivated plant | AGROVOC, Crop Ontology |
| Scientific_Name | AGROVOC:c_13864 | Scientific Name | binomial name, taxonomy | AGROVOC |
| Variety | AGROVOC:c_8151; CO:320:0000001 | Variety | cultivar, variety | AGROVOC, Crop Ontology |
| Season | AGROVOC:c_1762 | Growing Season | cropping season, rabi, kharif | AGROVOC |
| Growth_Duration_Days | AGROVOC:c_16317 | Growth Period | days to maturity, crop duration | AGROVOC |

### C. Experimental Design

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Design | AGROVOC:c_15888 | Experimental Design | trial design, RBD, CRD | AGROVOC |
| Replications | AGROVOC:c_6927 | Replication | replicate, block | AGROVOC |
| Plot_Size | AGROVOC:c_32651 | Plot Size | plot area, experimental unit | AGROVOC |
| Spacing_Row | AGROVOC:c_6729 | Row Spacing | row width, inter-row spacing | AGROVOC |
| Spacing_Plant | AGROVOC:c_5961 | Plant Spacing | intra-row spacing, plant distance | AGROVOC |
| Sample_Size | — | Sample Size | sample count, n | — |
| Location | AGROVOC:c_1965 | Location | experimental site, field location | AGROVOC |
| State | AGROVOC:c_4914 | State | province, administrative region | AGROVOC |
| Site | AGROVOC:c_4665 | Site | farm, station, research site | AGROVOC |

### D. Environment

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Latitude | ENVO:01000501 | Latitude | — | ENVO |
| Longitude | ENVO:01000502 | Longitude | — | ENVO |
| Altitude | ENVO:01000246 | Altitude | elevation | ENVO |
| Temperature_Max | ENVO:01000244 | Maximum Temperature | Tmax, max temp | ENVO |
| Temperature_Min | ENVO:01000243 | Minimum Temperature | Tmin, min temp | ENVO |
| Average_Temperature | ENVO:01000242 | Temperature | mean temperature, Tmean | ENVO |
| Rainfall | ENVO:01000507; AGROVOC:c_6435 | Rainfall | precipitation, rain | ENVO, AGROVOC |
| Humidity | ENVO:01000245; AGROVOC:c_3689 | Humidity | relative humidity | ENVO, AGROVOC |

### E. Soil Properties

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Soil_pH | ENVO:00001995; AGROVOC:c_7189 | Soil pH | pH, soil acidity | ENVO, AGROVOC |
| EC | AGROVOC:c_2482 | Electrical Conductivity | EC, conductivity | AGROVOC |
| Organic_Carbon | AGROVOC:c_5385 | Organic Carbon | OC, soil organic carbon | AGROVOC |
| Organic_Matter | AGROVOC:c_5387 | Organic Matter | OM, soil organic matter | AGROVOC |
| Nitrogen | AGROVOC:c_5188; CO:320:0000020 | Nitrogen | N, available nitrogen | AGROVOC, Crop Ontology |
| Phosphorus | AGROVOC:c_5801 | Phosphorus | P, available phosphorus | AGROVOC |
| Potassium | AGROVOC:c_6139 | Potassium | K, available potassium | AGROVOC |
| Sulphur | AGROVOC:c_7530 | Sulphur | S, sulfur | AGROVOC |
| Iron | AGROVOC:c_3958 | Iron | Fe | AGROVOC |
| Copper | AGROVOC:c_1863 | Copper | Cu | AGROVOC |
| Manganese | AGROVOC:c_4572 | Manganese | Mn | AGROVOC |
| Zinc | AGROVOC:c_8515 | Zinc | Zn | AGROVOC |
| Calcium | AGROVOC:c_1226 | Calcium | Ca | AGROVOC |
| Magnesium | AGROVOC:c_4518 | Magnesium | Mg | AGROVOC |
| Boron | AGROVOC:c_10020 | Boron | B | AGROVOC |
| Molybdenum | AGROVOC:c_4895 | Molybdenum | Mo | AGROVOC |

### F. Fertilizer Information

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Treatment | AGROVOC:c_7863 | Treatment | experimental treatment | AGROVOC |
| Fertilizer_Name | AGROVOC:c_15279 | Fertilizer | fertiliser, amendment | AGROVOC |
| Organic_Fertilizer | AGROVOC:c_24954 | Organic Fertilizer | organic amendment, manure | AGROVOC |
| Biofertilizer | AGROVOC:c_27528 | Biofertilizer | microbial inoculant | AGROVOC |
| Dose | AGROVOC:c_2363 | Dose | application rate, dose rate | AGROVOC |
| Application_Method | AGROVOC:c_5418 | Application Method | placement, broadcasting | AGROVOC |
| Application_Interval | — | Application Interval | timing, frequency | — |

### G. Crop Growth Parameters

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Shoot_Length_cm | PO:0009047 | Shoot Length | shoot height | Plant Ontology |
| Root_Length_cm | PO:0025203 | Root Length | root depth | Plant Ontology |
| Plant_Height_cm | PO:0009036; CO:320:0000012 | Plant Height | plant height, stem height | Plant Ontology, Crop Ontology |
| Shoot_Biomass_g | AGROVOC:c_8489 | Shoot Biomass | shoot dry weight | AGROVOC |
| Root_Biomass_g | AGROVOC:c_15974 | Root Biomass | root dry weight | AGROVOC |
| Leaf_Area_cm2 | PO:0025123; CO:320:0000028 | Leaf Area | leaf area | Plant Ontology, Crop Ontology |
| Leaf_Number | PO:0025070; CO:320:0000027 | Leaf Number | leaves per plant | Plant Ontology, Crop Ontology |
| Tillers | PO:0009049; CO:320:0000032 | Tiller Number | tillers per plant, shoots | Plant Ontology, Crop Ontology |
| Root_Diameter_mm | PO:0025002 | Root Diameter | root thickness | Plant Ontology |
| SPAD | CO:320:0000035 | SPAD Chlorophyll | chlorophyll, chlorophyll content | Crop Ontology |
| Moisture_Content | AGROVOC:c_4886 | Moisture Content | water content | AGROVOC |
| Dry_Matter | AGROVOC:c_2400 | Dry Matter | dry matter content | AGROVOC |
| Stem_Diameter_mm | PO:0025472 | Stem Diameter | stem thickness, culm diameter | Plant Ontology |
| Branches | PO:0009072 | Branch Number | branches per plant | Plant Ontology |
| Nodes | PO:0005005 | Node Number | nodes per plant | Plant Ontology |
| Flowers | PO:0009046 | Flower Number | flowers per plant | Plant Ontology |

### H. Yield Parameters

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Yield_per_Plot | AGROVOC:c_8486 | Plot Yield | plot harvest | AGROVOC |
| Yield_per_Acre | AGROVOC:c_8495 | Yield per Acre | acre yield | AGROVOC |
| Yield_per_Hectare | AGROVOC:c_8496 | Yield per Hectare | yield ha, productivity | AGROVOC |
| Fruit_Number | PO:0009011 | Fruit Number | fruits per plant | Plant Ontology |
| Fruit_Weight | AGROVOC:c_30789 | Fruit Weight | fruit mass | AGROVOC |
| Fruit_Diameter_mm | CO:320:0000037 | Fruit Diameter | fruit size, fruit width | Crop Ontology |
| Spike_Length | PO:0025396 | Spike Length | panicle length, ear length | Plant Ontology |
| Seeds_per_Spike | CO:320:0000042 | Seeds per Spike | grains per spike | Crop Ontology |
| 100_Seed_Weight | AGROVOC:c_2211; CO:320:0000040 | 100-Seed Weight | test weight, seed weight | AGROVOC, Crop Ontology |
| Root_Weight | AGROVOC:c_15976 | Root Weight | root mass | AGROVOC |
| Pod_Weight | — | Pod Weight | pod mass | — |
| Harvest_Index | AGROVOC:c_8483 | Harvest Index | HI | AGROVOC |
| Biomass_Yield | AGROVOC:c_8489 | Biomass Yield | total biomass, dry matter yield | AGROVOC |

### I. Grain Quality

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Protein | AGROVOC:c_6252; CO:320:0000045 | Protein Content | crude protein | AGROVOC, Crop Ontology |
| Ash | AGROVOC:c_676 | Ash Content | ash | AGROVOC |
| Gluten | AGROVOC:c_3296 | Gluten Content | gluten | AGROVOC |
| Fiber | AGROVOC:c_2893 | Crude Fiber | fiber content | AGROVOC |
| Carbohydrates | AGROVOC:c_1284 | Carbohydrate Content | carbs | AGROVOC |
| Fat | AGROVOC:c_2818 | Fat Content | oil content, crude fat | AGROVOC |
| Nitrogen_Content | AGROVOC:c_5194 | Grain Nitrogen | grain N | AGROVOC |
| Phosphorus_Content | AGROVOC:c_5802 | Grain Phosphorus | grain P | AGROVOC |
| Potassium_Content | AGROVOC:c_6140 | Grain Potassium | grain K | AGROVOC |
| Iron_Content | AGROVOC:c_3959 | Grain Iron | grain Fe, iron content | AGROVOC |
| Copper_Content | AGROVOC:c_1863 | Grain Copper | grain Cu | AGROVOC |
| Zinc_Content | AGROVOC:c_8515 | Grain Zinc | grain Zn | AGROVOC |
| Manganese_Content | AGROVOC:c_4572 | Grain Manganese | grain Mn | AGROVOC |
| Sulphur_Content | AGROVOC:c_5156 | Grain Sulphur | grain S | AGROVOC |

### J. ML Target Variables

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Target_Yield | AGROVOC:c_8496 | Target Yield | predicted yield | AGROVOC |
| Target_Fertilizer | AGROVOC:c_2363 | Target Fertilizer | recommended dose | AGROVOC |
| Target_Nitrogen | AGROVOC:c_5188 | Target Nitrogen | N recommendation | AGROVOC |
| Target_Phosphorus | AGROVOC:c_5801 | Target Phosphorus | P recommendation | AGROVOC |
| Target_Potassium | AGROVOC:c_6139 | Target Potassium | K recommendation | AGROVOC |

### K-M: Engineered Features, Leakage, Encoded Variables

| Variable | Ontology ID | Preferred Label | Synonyms | Source |
|----------|-------------|-----------------|----------|--------|
| Growing_Degree_Days | AGROVOC:c_15277 | Growing Degree Days | GDD, thermal time | AGROVOC |
| Heat_Units | AGROVOC:c_13612 | Heat Units | HU, growing degree units | AGROVOC |
| Nitrogen_Use_Efficiency | AGROVOC:c_27612 | Nitrogen Use Efficiency | NUE | AGROVOC |
| Water_Use_Efficiency | AGROVOC:c_16266 | Water Use Efficiency | WUE | AGROVOC |

---

## Crosswalk to Other Standards

### USDA-NRCS Soil Properties

| UAMS Variable | USDA-NRCS Code | Description |
|---------------|----------------|-------------|
| Soil_pH | pH | Soil reaction |
| EC | ECEC | Electrical conductivity |
| Organic_Carbon | SOC | Soil organic carbon |
| Nitrogen | N | Total nitrogen |
| Phosphorus | P | Available phosphorus |
| Potassium | K | Exchangeable potassium |

### ICAR-Crop Science Codes

| UAMS Variable | ICAR Code | Description |
|---------------|-----------|-------------|
| Crop | CS_001 | Crop name |
| Variety | CS_002 | Variety/cultivar |
| Season | CS_003 | Growing season |
| Design | CS_012 | Experimental design |

---

## Adding New Ontology Mappings

Mappings are maintained in `ades/agents/agent03_ontology.py`. To add a new mapping:

1. Add the entry to `ONTOLOGY_KNOWLEDGE_BASE` dictionary with ontology IDs
2. Update this registry document
3. Run tests to verify the mapping is exported correctly
