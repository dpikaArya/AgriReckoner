# UAMS Ontology Registry (generated)

**Version:** 2.0.0  
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
| Country | A. Paper Metadata | — | — | — | — |
| Crop | B. Crop Information | CO_320:0000000 | — | — | crop species, crop type, cultivated plant |
| Scientific_Name | B. Crop Information | — | — | — | — |
| Variety | B. Crop Information | CO_320:0000001 | — | — | cultivar, variety name, variety type |
| Season | B. Crop Information | — | — | — | cropping_season, growing_season, kharif, planting_date, rabi, seasonal … |
| Growth_Duration_Days | B. Crop Information | — | d | — | crop_duration, days_to_harvest, days_to_maturity, growth_duration, harvest_das, harvest_date … |
| Growth_Stage | B. Crop Information | — | — | — | phenological_stage |
| Design | C. Experimental Design | AGROVOC:c_2208 | — | — | exp_design, experimental_design |
| Replications | C. Experimental Design | — | — | — | no_of_replications, rep, repititions, replicate, replicates |
| Plot_Size | C. Experimental Design | — | m2 | — | plot_area, plot_area_m2, plot_dimension, plot_size_m2 |
| Spacing_Row | C. Experimental Design | — | cm | — | inter_row, row_spacing, row_spacing_cm, row_to_row |
| Spacing_Plant | C. Experimental Design | — | cm | — | intra_row, plant_spacing, plant_spacing_cm, plant_to_plant |
| Sample_Size | C. Experimental Design | — | — | — | — |
| Location | C. Experimental Design | AGROVOC:c_330988 | — | — | — |
| State | C. Experimental Design | AGROVOC:c_330998 | — | — | — |
| Site | C. Experimental Design | AGROVOC:c_331000 | — | — | — |
| Latitude | D. Environment | — | deg | — | lat |
| Longitude | D. Environment | — | deg | — | lon, long |
| Altitude | D. Environment | — | m | — | elevation |
| Temperature_Max | D. Environment | — | Cel | -20.0–55.0 | daily maximum temperature, max temp, max_temperature, maximum temperature, temp_max, tmax … |
| Temperature_Min | D. Environment | — | Cel | -30.0–40.0 | daily minimum temperature, min temp, min_temperature, minimum temperature, temp_min, tmin … |
| Average_Temperature | D. Environment | — | Cel | — | avg_temp, temp, temperature |
| Rainfall | D. Environment | AGROVOC:c_a060993c | mm | 0.0–10000.0 | precipitation, precipitation amount, rain, rainfall amount, rainfall_mm |
| Humidity | D. Environment | PATO:0015009; AGROVOC:c_3689 | % | 0.0–100.0 | air humidity, atmospheric humidity, relative_humidity |
| Soil_pH | E. Soil Properties | AGROVOC:c_34901 | — | 3.0–10.0 | pH value, ph, soil acidity, soil reaction |
| EC | E. Soil Properties | — | dS/m | 0.0–10.0 | ec_ds_m, electrical_conductivity, salinity, soil EC |
| Organic_Carbon | E. Soil Properties | AGROVOC:c_389fe908 | % | 0.0–10.0 | OC, organic_c, organic_carbon_%, soil organic carbon, total organic carbon |
| Organic_Matter | E. Soil Properties | AGROVOC:c_5387 | % | — | organic_matter_% |
| Nitrogen | E. Soil Properties | AGROVOC:c_5192; CO_320:0000020 | kg/ha | 0.0–1000.0 | available nitrogen, available_n, available_n_kg_ha, n, n_kg_ha, nitrogen_kg_ha … |
| Phosphorus | E. Soil Properties | AGROVOC:c_5804 | kg/ha | 0.0–500.0 | available phosphorus, available_p, available_p_kg_ha, p, p2o5, p2o5_kg_ha … |
| Potassium | E. Soil Properties | AGROVOC:c_6139 | kg/ha | 0.0–1000.0 | available potassium, available_k, available_k_kg_ha, k, k2o, k2o_kg_ha … |
| Sulphur | E. Soil Properties | — | mg/kg | — | s, sulfur, sulphur_ppm |
| Iron | E. Soil Properties | AGROVOC:c_3950 | mg/kg | — | fe, iron_mgkg, iron_ppm |
| Copper | E. Soil Properties | AGROVOC:c_1868 | mg/kg | — | copper_ppm, cu, cu_ppm |
| Manganese | E. Soil Properties | AGROVOC:c_4570 | mg/kg | — | mn, mn_ppm |
| Zinc | E. Soil Properties | AGROVOC:c_8517 | mg/kg | — | zn, zn_ppm |
| Calcium | E. Soil Properties | AGROVOC:c_1196 | mg/kg | — | ca |
| Magnesium | E. Soil Properties | AGROVOC:c_4517 | mg/kg | — | mg |
| Boron | E. Soil Properties | AGROVOC:c_1018 | mg/kg | — | b |
| Molybdenum | E. Soil Properties | AGROVOC:c_4901 | mg/kg | — | mo |
| Treatment | F. Fertilizer Information | — | — | — | treatments |
| Fertilizer_Name | F. Fertilizer Information | AGROVOC:c_2867 | — | — | amendment, fertiliser, fertilizer |
| Organic_Fertilizer | F. Fertilizer Information | — | — | — | compost, farmyard_manure, fym, organic_amendments, vermicompost |
| Biofertilizer | F. Fertilizer Information | — | — | — | biofertilizers, inoculation |
| Dose | F. Fertilizer Information | — | kg/ha | — | dose_kg_acre, dose_kg_ha, fertilizer_dose, k_dose, n_dose, p_dose |
| Application_Method | F. Fertilizer Information | — | — | — | — |
| Application_Interval | F. Fertilizer Information | — | — | — | — |
| Shoot_Length_cm | G. Crop Growth Parameters | — | cm | — | shoot_length, shoot_length_at_harvest, shootheight |
| Root_Length_cm | G. Crop Growth Parameters | — | cm | — | root_length, root_length_at_harvest |
| Plant_Height_cm | G. Crop Growth Parameters | CO_320:0000005; CO_320:0000012; AGROVOC:c_61f3cae5 | cm | 0.0–500.0 | final_plant_height, height, plant_height, plant_height_at_harvest, shoot length |
| Shoot_Biomass_g | G. Crop Growth Parameters | — | g | — | shoot_biomass |
| Root_Biomass_g | G. Crop Growth Parameters | — | g | — | root_biomass |
| Leaf_Area_cm2 | G. Crop Growth Parameters | CO_320:0000028 | cm2 | — | lai, leaf_area, leaf_area_index |
| Leaf_Number | G. Crop Growth Parameters | CO_320:0000027 | — | — | leaves_plant, no_leaves, number_of_leaves |
| Tillers | G. Crop Growth Parameters | CO_320:0000032 | — | — | no_tillers, number_of_tillers, tiller_number |
| Root_Diameter_mm | G. Crop Growth Parameters | — | mm | — | root_diameter |
| SPAD | G. Crop Growth Parameters | CO_320:0000035 | — | 0.0–80.0 | SPAD value, chlorophyll, chlorophyll_content, chlorophyll_content_spad, chlorophyll_spad, leaf greenness |
| Moisture_Content | G. Crop Growth Parameters | AGROVOC:c_4886 | % | — | moisture |
| Dry_Matter | G. Crop Growth Parameters | AGROVOC:c_331318 | % | — | dry_matter_%, dry_matter_pct |
| Stem_Diameter_mm | G. Crop Growth Parameters | — | mm | — | stem_diameter |
| Branches | G. Crop Growth Parameters | — | — | — | branches_per_plant, branches_plant, number_of_branches |
| Nodes | G. Crop Growth Parameters | — | — | — | number_of_nodes |
| Flowers | G. Crop Growth Parameters | — | — | — | flowers_per_plant, flowers_plant, number_of_flowers |
| Plant_Height_30_cm | G. Crop Growth Parameters | — | cm | — | — |
| Plant_Height_60_cm | G. Crop Growth Parameters | — | cm | — | — |
| Plant_Height_90_cm | G. Crop Growth Parameters | — | cm | — | — |
| Leaf_Area_30_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Leaf_Area_60_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Leaf_Area_90_cm2 | G. Crop Growth Parameters | — | cm2 | — | — |
| Yield_per_Plot | H. Yield Parameters | AGROVOC:c_10176 | g | — | economic_yield, fresh_weight, fresh_weight_g, grain_yield, seed_yield, yield … |
| Yield_per_Acre | H. Yield Parameters | AGROVOC:c_10176 | kg/[acr_us] | — | — |
| Yield_per_Hectare | H. Yield Parameters | AGROVOC:c_10176 | kg/ha | 0.0–50000.0 | crop yield, grain yield, yield ha, yield_per_ha |
| Fruit_Number | H. Yield Parameters | — | — | 0.0–10000.0 | fruits_plant |
| Fruit_Weight | H. Yield Parameters | AGROVOC:c_7db831f9 | g | 0.0–5000.0 | fruit_weight_g |
| Fruit_Diameter_mm | H. Yield Parameters | CO_320:0000037 | mm | — | fruit_diameter |
| Spike_Length | H. Yield Parameters | — | cm | — | panicle_length, spike_length_cm |
| Seeds_per_Spike | H. Yield Parameters | CO_320:0000042 | — | — | grains_per_spike, seeds_spike |
| 100_Seed_Weight | H. Yield Parameters | AGROVOC:c_36510; CO_320:0000040 | g | 0.0–500.0 | 1000_grain_weight, hundred_seed_weight, test_weight, test_weight_g, thousand_grain_weight, weight_100_seeds_g |
| Root_Weight | H. Yield Parameters | — | g | — | root_weight_g |
| Pod_Weight | H. Yield Parameters | — | g | — | pod_weight_g |
| Harvest_Index | H. Yield Parameters | AGROVOC:c_24854 | — | 0.0–1.5 | HI |
| Biomass_Yield | H. Yield Parameters | — | kg/ha | — | biological_yield, straw_yield, total_biomass |
| Protein | I. Grain Quality | AGROVOC:c_6251; CO_320:0000045 | % | 0.0–60.0 | crude protein, grain protein, protein_%, protein_content |
| Ash | I. Grain Quality | AGROVOC:c_665 | % | — | ash_%, ash_pct |
| Gluten | I. Grain Quality | AGROVOC:c_3294 | % | — | — |
| Fiber | I. Grain Quality | — | % | — | crude_fiber |
| Carbohydrates | I. Grain Quality | AGROVOC:c_1300 | % | — | carbs |
| Fat | I. Grain Quality | — | % | — | crude_fat, oil |
| Nitrogen_Content | I. Grain Quality | AGROVOC:c_5193 | % | — | — |
| Phosphorus_Content | I. Grain Quality | AGROVOC:c_5804 | % | — | — |
| Potassium_Content | I. Grain Quality | — | % | — | — |
| Iron_Content | I. Grain Quality | AGROVOC:c_3950 | mg/kg | — | — |
| Copper_Content | I. Grain Quality | — | mg/kg | — | — |
| Zinc_Content | I. Grain Quality | — | mg/kg | — | — |
| Manganese_Content | I. Grain Quality | — | mg/kg | — | — |
| Sulphur_Content | I. Grain Quality | AGROVOC:c_7514 | % | — | — |
| Target_Yield | J. ML Target Variables | — | kg/ha | — | — |
| Target_Fertilizer | J. ML Target Variables | — | kg/ha | — | — |
| Target_Nitrogen | J. ML Target Variables | — | kg/ha | — | — |
| Target_Phosphorus | J. ML Target Variables | — | kg/ha | — | — |
| Target_Potassium | J. ML Target Variables | — | kg/ha | — | — |
| Growing_Degree_Days | K. Engineered Features | — | Cel.d | — | — |
| Heat_Units | K. Engineered Features | — | Cel.d | — | — |
| Harvest_Index_Calc | K. Engineered Features | — | — | — | — |
| Nitrogen_Use_Efficiency | K. Engineered Features | AGROVOC:c_1e66facd | kg/kg | — | — |
| Water_Use_Efficiency | K. Engineered Features | — | kg/m3 | — | — |
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
| Title | A. Paper Metadata | — | — | — | title_of_article |
| Institution | A. Paper Metadata | — | — | — | institution |
| Cover_Crop | B. Crop Information | — | — | — | cover_crop, cover crop |
| Previous_Crop | B. Crop Information | — | — | — | previous_crop_name |
| Intercrop_Name | B. Crop Information | — | — | — | intercrop_name |
| Plot_ID | C. Experimental Design | — | — | — | plot_id |
| Block | C. Experimental Design | — | — | — | block |
| Experiment_Objective | C. Experimental Design | — | — | — | experimental_objective_description |
| Sowing_Date | C. Experimental Design | — | — | — | sowing_date |
| Harvest_Date | C. Experimental Design | — | — | — | harvest_date |
| Solar_Radiation | D. Environment | — | W/m2 | — | solar_radiation |
| Wind_Speed | D. Environment | — | m/s | — | wind_speed |
| ET0 | D. Environment | — | mm/d | — | potential_evapotranspiration |
| Climate_Zone | D. Environment | — | — | — | climate_zone |
| Sodium | E. Soil Properties | — | mg/kg | — | sodium, soil_sodium, soil_na_mg_kg |
| Chlorine | E. Soil Properties | — | mg/kg | — | chlorine, soil_cl_mg_kg |
| CEC | E. Soil Properties | — | cmol+/kg | — | soil_cec_cmol_kg |
| Soil_Bulk_Density | E. Soil Properties | — | Mg/m3 | 0.5–2.0 | soil_bulk_density_mg_cm3 |
| Soil_Clay_Pct | E. Soil Properties | — | % | 0–100 | soil_clay_% |
| Soil_Silt_Pct | E. Soil Properties | — | % | 0–100 | soil_silt_% |
| Soil_Sand_Pct | E. Soil Properties | — | % | 0–100 | soil_sand_% |
| Soil_Texture | E. Soil Properties | — | — | — | soil_texture_name |
| Soil_Class | E. Soil Properties | — | — | — | soil_class |
| Soil_Depth_cm | E. Soil Properties | — | cm | 0–None | soil_depth_cm |
| Fertilizer_Calcium | F. Fertilizer Information | — | kg/ha | — | fertiliser_ca_kg_ca_ha |
| Fertilizer_Magnesium | F. Fertilizer Information | — | kg/ha | — | fertiliser_mg_kg_mg_ha |
| Fertilizer_Boron | F. Fertilizer Information | — | kg/ha | — | fertiliser_b_kg_b_ha |
| Fertilizer_Chlorine | F. Fertilizer Information | — | kg/ha | — | fertiliser_cl_kg_cl_ha |
| Fertilizer_Molybdenum | F. Fertilizer Information | — | kg/ha | — | fertiliser_mo_kg_mo_ha |
| Plant_Height_120_cm | G. Crop Growth Parameters | — | cm | — | ph120, plant_height_120_cm |
| Leaf_Area_120_cm2 | G. Crop Growth Parameters | — | cm2 | — | la120, leaf_area_120_cm2 |
| Branches_60 | G. Crop Growth Parameters | — | 1 | — | branches_60 |
| Branches_90 | G. Crop Growth Parameters | — | 1 | — | branches_90 |
| Flowers_60 | G. Crop Growth Parameters | — | 1 | — | flowers_60 |
| FruitWeight_90_g | H. Yield Parameters | — | g | — | fruitweight_90_g |
| FruitWeight_120_g | H. Yield Parameters | — | g | — | fruitweight_120_g |
| FruitDiameter_90_mm | H. Yield Parameters | — | mm | — | fruitdiameter_90_mm |
| FruitDiameter_120_mm | H. Yield Parameters | — | mm | — | fruitdiameter_120_mm |
| YieldPlot_90_g | H. Yield Parameters | — | g | — | yieldplot_90_g |
| YieldPlot_120_g | H. Yield Parameters | — | g | — | yieldplot_120_g |
| Yield_13pct_Moisture | H. Yield Parameters | — | kg/ha | — | yield_13%_moisture |
| Marketable_Yield | H. Yield Parameters | — | kg/ha | — | marketable_yield |
| Biological_Yield | H. Yield Parameters | — | kg/ha | — | biological_yield |
| Stover_Yield | H. Yield Parameters | — | kg/ha | — | stover_yield |
| Seed_Yield | H. Yield Parameters | — | kg/ha | — | seed_yield |
| Pods_Number | H. Yield Parameters | — | 1 | — | pods_number |
| Seeds_per_Pod | H. Yield Parameters | — | 1 | — | seeds_per_pod |
| Weed_Biomass | H. Yield Parameters | — | g/m2 | — | weed_biomass |
| Calcium_Content | I. Grain Quality | — | mg/kg | — | calcium_content |
| Magnesium_Content | I. Grain Quality | — | mg/kg | — | magnesium_content |
| Boron_Content | I. Grain Quality | — | mg/kg | — | boron_content |
| Chlorine_Content | I. Grain Quality | — | mg/kg | — | chlorine_content |
| Molybdenum_Content | I. Grain Quality | — | mg/kg | — | molybdenum_content |
| Starch | I. Grain Quality | — | % | — | starch |
| Oil | I. Grain Quality | — | % | — | oil |
| SOC_Mineralization | O. Soil Biological Properties | — | mg/kg/d | — | soc_mineralization, basal soc mineralization |
| Residue_C_Decomposition | O. Soil Biological Properties | — | mg/kg/d | — | residue/root c decomposition, residue_c_decomposition |
| Potential_Evapotranspiration | O. Soil Biological Properties | — | mm | — | pe |
| Soil_Moisture | O. Soil Biological Properties | — | % | 0–100 | soil_moisture_%_whc, soil_moisture_% |
| Soil_Nitrate_N | O. Soil Biological Properties | — | mg/kg | — | water_extractable_nitrate, wec_no3_n_mg_n_kg |
| Soil_Ammonium_N | O. Soil Biological Properties | — | mg/kg | — | water_extractable_nh4, wec_nh4_mg_n_kg |
| Soil_Available_K | O. Soil Biological Properties | — | mg/kg | — | — |
| Soil_POC | O. Soil Biological Properties | — | g/kg | — | — |
| Soil_DOC | O. Soil Biological Properties | — | mg/kg | — | — |
| Soil_LOC | O. Soil Biological Properties | — | mg/kg | — | — |
| Soil_MBC | O. Soil Biological Properties | — | mg/kg | — | soil_mbc_mg_c_kg |
| Soil_MBN | O. Soil Biological Properties | — | mg/kg | — | soil_mbn_mg_n_kg |
| Soil_Carbon_Walkley_Black | O. Soil Biological Properties | — | % | — | soil_carbon_(walkley-black)_% |
| Soil_Nitrogen_Kjeldahl | O. Soil Biological Properties | — | g/kg | — | soil_nc_kjeldahl_g_n_kg |
| Soil_Nitrogen_Alkaline_Permanganate | O. Soil Biological Properties | — | kg/ha | — | soil_n_alkaline_permanganate_kg_n_ha |
| Soil_Sulphur_Available | O. Soil Biological Properties | — | mg/kg | — | soil_s_(cacl2)_mg_s_kg |
| Soil_Sulphur_Mehlich3 | O. Soil Biological Properties | — | mg/kg | — | soil_s_(mehlich-3)_mg_s_kg |
| Soil_NO3 | O. Soil Biological Properties | — | mg/kg | — | soil_nitrate |
| Soil_NH4 | O. Soil Biological Properties | — | mg/kg | — | soil_nh4 |
| Soil_N_Parco2020 | O. Soil Biological Properties | — | kg/ha | — | soil_n_(parco2020)_kg_n_ha |
| Soil_N_DeSilva2008 | O. Soil Biological Properties | — | kg/ha | — | soil_n_(de silva 2008)_kg_n_ha |
| Soil_Ca_Ammonium_Acetate | O. Soil Biological Properties | — | mg/kg | — | soil_ca_(ammonium_acetate)_mg_ca_kg |
| Soil_Mg_Ammonium_Acetate | O. Soil Biological Properties | — | mg/kg | — | soil_mg_(ammonium_acetate)_mg_k_kg |
| Soil_K_Ammonium_Acetate | O. Soil Biological Properties | — | mg/kg | — | soil_k_(ammonium_acetate)_mg_k_kg |
| Soil_Glucosidase | P. Soil Enzyme Activity | — | mg/kg/h | — | glucosidase |
| Soil_Xylosidase | P. Soil Enzyme Activity | — | mg/kg/h | — | xylosidase |
| Soil_Cellobiosidase | P. Soil Enzyme Activity | — | mg/kg/h | — | cellobiosidase |
| Beta_Glucosidase | P. Soil Enzyme Activity | — | mg/kg/h | — | beta_glucosidase |
| Peroxidase | P. Soil Enzyme Activity | — | mg/kg/h | — | peroxidase |
| Polyphenol_Oxidase | P. Soil Enzyme Activity | — | mg/kg/h | — | polyphenol_oxidase, ppo |
| N_Acetyl_Glucosaminidase | P. Soil Enzyme Activity | — | mg/kg/h | — | n-acetyl-glucosaminidase, nag |
| Leucine_Aminopeptidase | P. Soil Enzyme Activity | — | mg/kg/h | — | leucine_aminopeptidase, lap |
| Phosphatase | P. Soil Enzyme Activity | — | mg/kg/h | — | phosphatase, acid_phosphatase, alkaline_phosphatase |
| Catalase | P. Soil Enzyme Activity | — | mL/g/h | — | catalase |
| Urease | P. Soil Enzyme Activity | — | mg/kg/h | — | urease |
| Dehydrogenase | P. Soil Enzyme Activity | — | mg/kg/d | — | dehydrogenase |
| Bacterial_Diversity_Chao1 | Q. Soil Microbial Community | — | 1 | — | bacterial_chao1 |
| Bacterial_Diversity_Shannon | Q. Soil Microbial Community | — | 1 | — | bacterial_shannon |
| Fungal_Diversity_Chao1 | Q. Soil Microbial Community | — | 1 | — | fungal_chao1 |
| Fungal_Diversity_Shannon | Q. Soil Microbial Community | — | 1 | — | fungal_shannon |
| Total_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | total_plfas |
| Gram_Positive_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | gram_positive_plfas |
| Gram_Negative_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | gram_negative_plfas |
| Fungal_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | fungal_plfas |
| Bacterial_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | bacterial_plfas |
| Actinomycetes_PLFAs | Q. Soil Microbial Community | — | nmol/g | — | actinomycetes_plfas |
| NDVI | R. Remote Sensing Indices | — | 1 | -1–1 | ndvi, ndvi_green |
| EVI | R. Remote Sensing Indices | — | 1 | -1–1 | evi, enhanced_vegetation_index |
| SAVI | R. Remote Sensing Indices | — | 1 | -1–1 | savi, soil_adjusted_vegetation_index |
| NDWI | R. Remote Sensing Indices | — | 1 | -1–1 | ndwi |
| NDRE | R. Remote Sensing Indices | — | 1 | -1–1 | ndre |
| Canopy_Temperature | R. Remote Sensing Indices | — | degC | — | canopy_temperature, canopy_temp |
| CIG | R. Remote Sensing Indices | — | 1 | — | cig, chlorophyll_index_green |
| CIRE | R. Remote Sensing Indices | — | 1 | — | cire, chlorophyll_index_red_edge |
| Calcium_Mehlich3 | S. Soil Exchangeable Cations | — | mg/kg | — | calcium_mehlich3, soil_ca_mehlich-3_mg_ca_kg |
| Calcium_Saturation | S. Soil Exchangeable Cations | — | % | 0–100 | calcium_saturation_% |
| Magnesium_Saturation | S. Soil Exchangeable Cations | — | % | 0–100 | magnesium_saturation_% |
| Sodium_Saturation | S. Soil Exchangeable Cations | — | % | 0–100 | sodium_saturation_% |
| Potassium_Saturation | S. Soil Exchangeable Cations | — | % | 0–100 | potassium_saturation_% |
| Hydrogen_Saturation | S. Soil Exchangeable Cations | — | % | 0–100 | hydrogen_saturation_% |
| Water_Extractable_Nitrate | T. Water Extractable Organic | — | mg/kg | — | wec_n_mg_n_kg, wec_no3_n_mg_n_kg |
| Water_Extractable_Organic_Carbon | T. Water Extractable Organic | — | mg/kg | — | wec_oc_mg_c_kg, wec_oc_mg_oc_kg |
| Water_Extractable_Organic_Nitrogen | T. Water Extractable Organic | — | mg/kg | — | wec_on_mg_n_kg, wec_on_mg_on_kg |
| Water_Extractable_NH4 | T. Water Extractable Organic | — | mg/kg | — | wec_nh4_mg_n_kg, wec_nh4_n_mg_n_kg |
| Water_Extractable_CN_Ratio | T. Water Extractable Organic | — | 1 | — | wec_cn_ratio |
| Aggregate_Stability | U. Soil Physical Properties | — | % | — | aggregate_stability |
| Soil_Respiration | U. Soil Physical Properties | — | mg/kg/h | — | soil_respiration |
| Infiltration_Rate | U. Soil Physical Properties | — | mm/h | — | infiltration_rate |
| Nematode_Maturity_Index | V. Nematode Ecology | — | 1 | — | mi |
| Nematode_Plant_Parasitic_Index | V. Nematode Ecology | — | 1 | — | ppi |
| Nematode_Channel_Index | V. Nematode Ecology | — | 1 | — | ci |
| Nematode_Basal_Index | V. Nematode Ecology | — | 1 | — | bi |
| Nematode_Enrichment_Index | V. Nematode Ecology | — | 1 | — | ei |
| Nematode_Structure_Index | V. Nematode Ecology | — | 1 | — | si |
| Nematode_Total_Biomass_mg | V. Nematode Ecology | — | mg | — | nematode_total_biomass_mg |
| Total_Cost | W. Economic Parameters | — | USD/ha | — | total_cost |
| Total_Return | W. Economic Parameters | — | USD/ha | — | total_return |
| Net_Profit | W. Economic Parameters | — | USD/ha | — | net_profit |
| Benefit_Cost_Ratio | W. Economic Parameters | — | 1 | — | system benefit cost ratio, bc_ratio |
| System_Benefit_USD_ha | W. Economic Parameters | — | USD/ha | — | system benefit (usd/ha) |
| Off_Season_Contribution_Pct | W. Economic Parameters | — | % | — | off-season contribution to system benefit (%) |
| System_Water_Use_Efficiency | W. Economic Parameters | — | kg/m3/ha | — | system water use efficiency (kg/m3/ha) |
| Phosphorus_Uptake | X. Nutrient Uptake | — | kg/ha | — | p_uptake, phosphorus_uptake, p_uptake_kg_ha |
| Nitrogen_Uptake | X. Nutrient Uptake | — | kg/ha | — | n_uptake, nitrogen_uptake, n_uptake_kg_ha |
| Potassium_Uptake | X. Nutrient Uptake | — | kg/ha | — | k_uptake, potassium_uptake, k_uptake_kg_ha |
| Sulphur_Uptake | X. Nutrient Uptake | — | kg/ha | — | s_uptake, sulphur_uptake |
| Calcium_Uptake | X. Nutrient Uptake | — | kg/ha | — | ca_uptake, calcium_uptake |
| Magnesium_Uptake | X. Nutrient Uptake | — | kg/ha | — | mg_uptake, magnesium_uptake |
| Boron_Uptake | X. Nutrient Uptake | — | kg/ha | — | b_uptake, boron_uptake |
| Chlorine_Uptake | X. Nutrient Uptake | — | kg/ha | — | cl_uptake, chlorine_uptake |
| Copper_Uptake | X. Nutrient Uptake | — | kg/ha | — | cu_uptake, copper_uptake |
| Iron_Uptake | X. Nutrient Uptake | — | kg/ha | — | fe_uptake, iron_uptake |
| Manganese_Uptake | X. Nutrient Uptake | — | kg/ha | — | mn_uptake, manganese_uptake |
| Molybdenum_Uptake | X. Nutrient Uptake | — | kg/ha | — | mo_uptake, molybdenum_uptake |
| Zinc_Uptake | X. Nutrient Uptake | — | kg/ha | — | zn_uptake, zinc_uptake |
| Irrigation_Method | Y. Irrigation & Management | — | — | — | irrigation, irrigation_method_name |
| Irrigation_mm | Y. Irrigation & Management | — | mm | — | actual_irrigation_mm, water_amount_mm |
| Sowing_Density | Y. Irrigation & Management | — | seeds/m2 | — | sow_density_seeds_m2 |
| Harvest_Density | Y. Irrigation & Management | — | plants/m2 | — | harvest_density_plants_m2 |
| Cultivation_Method | Y. Irrigation & Management | — | — | — | cultivation_method_name |
| N_Fixation | Y. Irrigation & Management | — | kg/ha | — | n_fixation_kg_n_ha |
| Soil_pH_CaCl2 | Z. Soil pH Variants | — | 1 | 0–14 | soil_ph_cacl2, soil_ph_cacl2_number |
| Soil_pH_KCl | Z. Soil pH Variants | — | 1 | 0–14 | soil_ph_kcl, soil_ph_kcl_number |
