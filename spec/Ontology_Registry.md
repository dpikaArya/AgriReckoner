# UAMS Ontology Registry (generated)

**Version:** 1.0.0  
**Namespace:** https://w3id.org/uams#  
**Provenance note:** Terms migrated from agri_ai_agent/agents/extraction_agent.py (ONTOLOGY_KNOWLEDGE_BASE, authoritative) and spec/Ontology_Registry.md. CURIEs normalised to PREFIX:localid. Release versions unknown — TODO.

Generated from `spec/uams_ontology.yaml` by `scripts/gen_ontology_docs.py`. Do not edit by hand.

| Variable | Group | Terms (CURIE) | Unit (UCUM) | Range | Synonyms |
|----------|-------|---------------|-------------|-------|----------|
| Paper_ID | A. Paper Metadata | — | — | — | — |
| DOI | A. Paper Metadata | — | — | — | — |
| Journal | A. Paper Metadata | — | — | — | — |
| Year | A. Paper Metadata | — | — | — | — |
| Authors | A. Paper Metadata | — | — | — | — |
| Country | A. Paper Metadata | AGROVOC:c_1625 | — | — | — |
| Crop | B. Crop Information | AGROVOC:c_2556; CO_320:0000000; FOODON:00001002 | — | — | crop species, crop type, cultivated plant |
| Scientific_Name | B. Crop Information | AGROVOC:c_13864 | — | — | — |
| Variety | B. Crop Information | AGROVOC:c_8151; CO_320:0000001 | — | — | cultivar, variety name, variety type |
| Season | B. Crop Information | AGROVOC:c_1762 | — | — | cropping_season, growing_season, kharif, planting_date, rabi, seasonal … |
| Growth_Duration_Days | B. Crop Information | AGROVOC:c_16317 | d | — | crop_duration, days_to_harvest, days_to_maturity, growth_duration, harvest_das, harvest_date … |
| Growth_Stage | B. Crop Information | — | — | — | phenological_stage |
| Design | C. Experimental Design | AGROVOC:c_15888 | — | — | exp_design, experimental_design |
| Replications | C. Experimental Design | AGROVOC:c_6927 | — | — | no_of_replications, rep, repititions, replicate, replicates |
| Plot_Size | C. Experimental Design | AGROVOC:c_32651 | m2 | — | plot_area, plot_area_m2, plot_dimension, plot_size_m2 |
| Spacing_Row | C. Experimental Design | AGROVOC:c_6729 | cm | — | inter_row, row_spacing, row_spacing_cm, row_to_row |
| Spacing_Plant | C. Experimental Design | AGROVOC:c_5961 | cm | — | intra_row, plant_spacing, plant_spacing_cm, plant_to_plant |
| Sample_Size | C. Experimental Design | — | — | — | — |
| Location | C. Experimental Design | AGROVOC:c_1965 | — | — | — |
| State | C. Experimental Design | AGROVOC:c_4914 | — | — | — |
| Site | C. Experimental Design | AGROVOC:c_4665 | — | — | — |
| Latitude | D. Environment | ENVO:01000501 | deg | — | lat |
| Longitude | D. Environment | ENVO:01000502 | deg | — | lon, long |
| Altitude | D. Environment | ENVO:01000246 | m | — | elevation |
| Temperature_Max | D. Environment | ENVO:01000244; AGROVOC:c_7651 | Cel | -20.0–55.0 | daily maximum temperature, max temp, max_temperature, maximum temperature, temp_max, tmax … |
| Temperature_Min | D. Environment | ENVO:01000243; AGROVOC:c_7652 | Cel | -30.0–40.0 | daily minimum temperature, min temp, min_temperature, minimum temperature, temp_min, tmin … |
| Average_Temperature | D. Environment | ENVO:01000242 | Cel | — | avg_temp, temp, temperature |
| Rainfall | D. Environment | ENVO:01000507; AGROVOC:c_6435 | mm | 0.0–10000.0 | precipitation, precipitation amount, rain, rainfall amount, rainfall_mm |
| Humidity | D. Environment | ENVO:01000245; AGROVOC:c_3689 | % | 0.0–100.0 | air humidity, atmospheric humidity, relative_humidity |
| Soil_pH | E. Soil Properties | ENVO:00001995; AGROVOC:c_7189 | — | 3.0–10.0 | pH value, ph, soil acidity, soil reaction |
| EC | E. Soil Properties | AGROVOC:c_2482 | dS/m | 0.0–10.0 | ec_ds_m, electrical_conductivity, salinity, soil EC |
| Organic_Carbon | E. Soil Properties | AGROVOC:c_5385; ENVO:00002281 | % | 0.0–10.0 | OC, organic_c, organic_carbon_%, soil organic carbon, total organic carbon |
| Organic_Matter | E. Soil Properties | AGROVOC:c_5387 | % | — | organic_matter_% |
| Nitrogen | E. Soil Properties | AGROVOC:c_5188; CO_320:0000020; FOODON:00001310 | kg/ha | 0.0–1000.0 | available nitrogen, available_n, available_n_kg_ha, n, n_kg_ha, nitrogen_kg_ha … |
| Phosphorus | E. Soil Properties | AGROVOC:c_5801 | kg/ha | 0.0–500.0 | available phosphorus, available_p, available_p_kg_ha, p, p2o5, p2o5_kg_ha … |
| Potassium | E. Soil Properties | AGROVOC:c_6139 | kg/ha | 0.0–1000.0 | available potassium, available_k, available_k_kg_ha, k, k2o, k2o_kg_ha … |
| Sulphur | E. Soil Properties | AGROVOC:c_7530 | mg/kg | — | s, sulfur, sulphur_ppm |
| Iron | E. Soil Properties | AGROVOC:c_3958 | mg/kg | — | fe, iron_mgkg, iron_ppm |
| Copper | E. Soil Properties | AGROVOC:c_1863 | mg/kg | — | copper_ppm, cu, cu_ppm |
| Manganese | E. Soil Properties | AGROVOC:c_4572 | mg/kg | — | mn, mn_ppm |
| Zinc | E. Soil Properties | AGROVOC:c_8515 | mg/kg | — | zn, zn_ppm |
| Calcium | E. Soil Properties | AGROVOC:c_1226 | mg/kg | — | ca |
| Magnesium | E. Soil Properties | AGROVOC:c_4518 | mg/kg | — | mg |
| Boron | E. Soil Properties | AGROVOC:c_10020 | mg/kg | — | b |
| Molybdenum | E. Soil Properties | AGROVOC:c_4895 | mg/kg | — | mo |
| Treatment | F. Fertilizer Information | AGROVOC:c_7863 | — | — | treatments |
| Fertilizer_Name | F. Fertilizer Information | AGROVOC:c_15279 | — | — | amendment, fertiliser, fertilizer |
| Organic_Fertilizer | F. Fertilizer Information | AGROVOC:c_24954 | — | — | compost, farmyard_manure, fym, organic_amendments, vermicompost |
| Biofertilizer | F. Fertilizer Information | AGROVOC:c_27528 | — | — | biofertilizers, inoculation |
| Dose | F. Fertilizer Information | AGROVOC:c_2363 | kg/ha | — | dose_kg_acre, dose_kg_ha, fertilizer_dose, k_dose, n_dose, p_dose |
| Application_Method | F. Fertilizer Information | AGROVOC:c_5418 | — | — | — |
| Application_Interval | F. Fertilizer Information | — | — | — | — |
| Shoot_Length_cm | G. Crop Growth Parameters | PO:0009047 | cm | — | shoot_length, shoot_length_at_harvest, shootheight |
| Root_Length_cm | G. Crop Growth Parameters | PO:0025203 | cm | — | root_length, root_length_at_harvest |
| Plant_Height_cm | G. Crop Growth Parameters | CO_320:0000005; CO_320:0000012; AGROVOC:c_330960; PO:0009036 | cm | 0.0–500.0 | final_plant_height, height, plant_height, plant_height_at_harvest, shoot length |
| Shoot_Biomass_g | G. Crop Growth Parameters | AGROVOC:c_8489 | g | — | shoot_biomass |
| Root_Biomass_g | G. Crop Growth Parameters | AGROVOC:c_15974 | g | — | root_biomass |
| Leaf_Area_cm2 | G. Crop Growth Parameters | PO:0025123; CO_320:0000028 | cm2 | — | lai, leaf_area, leaf_area_index |
| Leaf_Number | G. Crop Growth Parameters | PO:0025070; CO_320:0000027 | — | — | leaves_plant, no_leaves, number_of_leaves |
| Tillers | G. Crop Growth Parameters | PO:0009049; CO_320:0000032 | — | — | no_tillers, number_of_tillers, tiller_number |
| Root_Diameter_mm | G. Crop Growth Parameters | PO:0025002 | mm | — | root_diameter |
| SPAD | G. Crop Growth Parameters | CO_320:0000035 | — | 0.0–80.0 | SPAD value, chlorophyll, chlorophyll_content, chlorophyll_content_spad, chlorophyll_spad, leaf greenness |
| Moisture_Content | G. Crop Growth Parameters | AGROVOC:c_4886 | % | — | moisture |
| Dry_Matter | G. Crop Growth Parameters | AGROVOC:c_2400 | % | — | dry_matter_%, dry_matter_pct |
| Stem_Diameter_mm | G. Crop Growth Parameters | PO:0025472 | mm | — | stem_diameter |
| Branches | G. Crop Growth Parameters | PO:0009072 | — | — | branches_per_plant, branches_plant, number_of_branches |
| Nodes | G. Crop Growth Parameters | PO:0005005 | — | — | number_of_nodes |
| Flowers | G. Crop Growth Parameters | PO:0009046 | — | — | flowers_per_plant, flowers_plant, number_of_flowers |
| Plant_Height_30_cm | G. Crop Growth Parameters | — | cm | — | — |
| Plant_Height_60_cm | G. Crop Growth Parameters | — | cm | — | — |
| Plant_Height_90_cm | G. Crop Growth Parameters | — | cm | — | — |
| Leaf_Area_30_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Leaf_Area_60_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Leaf_Area_90_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Yield_per_Plot | H. Yield Parameters | AGROVOC:c_8486 | g | — | economic_yield, fresh_weight, fresh_weight_g, grain_yield, seed_yield, yield … |
| Yield_per_Acre | H. Yield Parameters | AGROVOC:c_8495 | kg/[acr_us] | — | — |
| Yield_per_Hectare | H. Yield Parameters | AGROVOC:c_8496 | kg/ha | 0.0–50000.0 | crop yield, grain yield, yield ha, yield_per_ha |
| Fruit_Number | H. Yield Parameters | PO:0009011 | — | 0.0–10000.0 | fruits_plant |
| Fruit_Weight | H. Yield Parameters | AGROVOC:c_30789 | g | 0.0–5000.0 | fruit_weight_g |
| Fruit_Diameter_mm | H. Yield Parameters | CO_320:0000037 | mm | — | fruit_diameter |
| Spike_Length | H. Yield Parameters | PO:0025396 | cm | — | panicle_length, spike_length_cm |
| Seeds_per_Spike | H. Yield Parameters | CO_320:0000042 | — | — | grains_per_spike, seeds_spike |
| 100_Seed_Weight | H. Yield Parameters | AGROVOC:c_2211; CO_320:0000040 | g | 0.0–500.0 | 1000_grain_weight, hundred_seed_weight, test_weight, test_weight_g, thousand_grain_weight, weight_100_seeds_g |
| Root_Weight | H. Yield Parameters | AGROVOC:c_15976 | g | — | root_weight_g |
| Pod_Weight | H. Yield Parameters | — | g | — | pod_weight_g |
| Harvest_Index | H. Yield Parameters | AGROVOC:c_8483 | — | 0.0–1.5 | HI |
| Biomass_Yield | H. Yield Parameters | — | kg/ha | — | biological_yield, straw_yield, total_biomass |
| Protein | I. Grain Quality | AGROVOC:c_6252; FOODON:00001024; CO_320:0000045 | % | 0.0–60.0 | crude protein, grain protein, protein_%, protein_content |
| Ash | I. Grain Quality | AGROVOC:c_676 | % | — | ash_%, ash_pct |
| Gluten | I. Grain Quality | AGROVOC:c_3296 | % | — | — |
| Fiber | I. Grain Quality | AGROVOC:c_2893 | % | — | crude_fiber |
| Carbohydrates | I. Grain Quality | AGROVOC:c_1284 | % | — | carbs |
| Fat | I. Grain Quality | AGROVOC:c_2818 | % | — | crude_fat, oil |
| Nitrogen_Content | I. Grain Quality | AGROVOC:c_5194 | % | — | — |
| Phosphorus_Content | I. Grain Quality | AGROVOC:c_5802 | % | — | — |
| Potassium_Content | I. Grain Quality | AGROVOC:c_6140 | % | — | — |
| Iron_Content | I. Grain Quality | AGROVOC:c_3959 | mg/kg | — | — |
| Copper_Content | I. Grain Quality | — | mg/kg | — | — |
| Zinc_Content | I. Grain Quality | — | mg/kg | — | — |
| Manganese_Content | I. Grain Quality | — | mg/kg | — | — |
| Sulphur_Content | I. Grain Quality | AGROVOC:c_5156 | % | — | — |
| Target_Yield | J. ML Target Variables | — | kg/ha | — | — |
| Target_Fertilizer | J. ML Target Variables | — | kg/ha | — | — |
| Target_Nitrogen | J. ML Target Variables | — | kg/ha | — | — |
| Target_Phosphorus | J. ML Target Variables | — | kg/ha | — | — |
| Target_Potassium | J. ML Target Variables | — | kg/ha | — | — |
| Growing_Degree_Days | K. Engineered Features | AGROVOC:c_15277 | Cel.d | — | — |
| Heat_Units | K. Engineered Features | AGROVOC:c_13612 | Cel.d | — | — |
| Harvest_Index_Calc | K. Engineered Features | — | — | — | — |
| Nitrogen_Use_Efficiency | K. Engineered Features | AGROVOC:c_27612 | kg/kg | — | — |
| Water_Use_Efficiency | K. Engineered Features | AGROVOC:c_16266 | kg/m3 | — | — |
| Rainfall_Anomaly | K. Engineered Features | — | mm | — | — |
| Stress_Index | K. Engineered Features | — | — | — | — |
| Disease_Risk_Index | K. Engineered Features | — | — | — | — |
| Yield_per_Plant | K. Engineered Features | — | g | — | — |
| Yield_per_Plot_Calc | K. Engineered Features | — | g | — | — |
| Yield_per_Hectare_Calc | K. Engineered Features | — | kg/ha | — | — |
| Temp_x_Rainfall | K. Engineered Features | — | — | — | — |
| N_x_P | K. Engineered Features | — | — | — | — |
| Temp_squared | K. Engineered Features | — | — | — | — |
| Rainfall_7d_MA | K. Engineered Features | — | mm | — | — |
| Temp_7d_MA | K. Engineered Features | — | Cel | — | — |
| Feature_Available_Before_Prediction | L. Leakage Labels | — | — | — | — |
| Crop_Code | M. Encoded Variables | — | — | — | — |
| Season_Code | M. Encoded Variables | — | — | — | — |
| Variety_Code | M. Encoded Variables | — | — | — | — |
| Fertilizer_Code | M. Encoded Variables | — | — | — | — |
| Soil_Texture_Code | M. Encoded Variables | — | — | — | — |
| Country_Code | M. Encoded Variables | — | — | — | — |
| Predicted_Yield | N. ML Predictions | — | kg/ha | — | — |
| Expected_Biomass | N. ML Predictions | — | kg/ha | — | — |
| Expected_Plant_Height | N. ML Predictions | — | cm | — | — |
| Recommended_Fertilizer | N. ML Predictions | — | — | — | — |
| Recommended_Dose | N. ML Predictions | — | kg/ha | — | — |
| Recommended_Application_Interval | N. ML Predictions | — | d | — | — |
| Expected_Yield_Increase | N. ML Predictions | — | % | — | — |
| Confidence_Score | N. ML Predictions | — | — | — | confidence |
| Recommendation_Summary | N. ML Predictions | — | — | — | — |
