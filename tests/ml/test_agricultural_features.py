import pandas as pd
import pytest

from src.ml.agricultural_features.climate_indices import ClimateIndices
from src.ml.agricultural_features.crop_interaction import CropEnvironmentInteraction
from src.ml.agricultural_features.soil_indices import SoilIndices


@pytest.fixture
def climate_data():
    return pd.DataFrame(
        {
            "Average_Temperature": [25, 30, 35, 20, 15],
            "Temperature_Max": [30, 35, 40, 25, 20],
            "Temperature_Min": [20, 25, 30, 15, 10],
            "Rainfall_mm": [800, 1200, 600, 400, 1000],
        }
    )


@pytest.fixture
def soil_data():
    return pd.DataFrame(
        {
            "Soil_pH": [6.5, 7.0, 5.5, 8.0, 4.5],
            "Organic_Carbon_pct": [0.8, 1.2, 0.5, 1.5, 0.3],
            "Nitrogen_kg_ha": [120, 150, 90, 180, 60],
            "Phosphorus_kg_ha": [50, 30, 20, 60, 15],
            "Potassium_kg_ha": [250, 180, 150, 300, 100],
        }
    )


class TestClimateIndices:
    def test_growing_degree_days_added(self, climate_data):
        ci = ClimateIndices()
        df = ci.growing_degree_days(climate_data.copy())
        assert "GDD_Base10" in df.columns
        assert df["GDD_Base10"].iloc[0] > 0

    def test_heat_stress_index(self, climate_data):
        ci = ClimateIndices()
        df = ci.heat_stress_index(climate_data.copy())
        assert "Heat_Stress_Index" in df.columns
        assert df["Heat_Stress_Index"].between(0, 1).all()

    def test_rainfall_deficit_index(self, climate_data):
        ci = ClimateIndices()
        df = ci.rainfall_deficit_index(climate_data.copy())
        assert "Rainfall_Deficit_Index" in df.columns
        assert df["Rainfall_Deficit_Index"].between(0, 1).all()

    def test_compute_all_adds_features(self, climate_data):
        ci = ClimateIndices()
        df = ci.compute_all(climate_data.copy())
        generated = [c for c in df.columns if c not in climate_data.columns]
        assert len(generated) >= 5

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        ci = ClimateIndices()
        result = ci.compute_all(df)
        assert list(result.columns) == ["a"]


class TestSoilIndices:
    def test_soil_fertility_index(self, soil_data):
        si = SoilIndices()
        df = si.soil_fertility_index(soil_data.copy())
        assert "Soil_Fertility_Index" in df.columns
        assert df["Soil_Fertility_Index"].between(0, 1).all()

    def test_ph_suitability(self, soil_data):
        si = SoilIndices()
        df = si.soil_ph_suitability(soil_data.copy())
        assert "Soil_pH_Suitability_Score" in df.columns

    def test_npk_balance_index(self, soil_data):
        si = SoilIndices()
        df = si.npk_balance_index(soil_data.copy())
        assert "NPK_Balance_Index" in df.columns or "N_K_Ratio" in df.columns

    def test_compute_all_adds_features(self, soil_data):
        si = SoilIndices()
        df = si.compute_all(soil_data.copy())
        generated = [c for c in df.columns if c not in soil_data.columns]
        assert len(generated) >= 5

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        si = SoilIndices()
        result = si.compute_all(df)
        assert "Soil_Fertility_Index" not in result.columns


class TestCropEnvironmentInteraction:
    def test_crop_duration_temperature(self):
        df = pd.DataFrame(
            {
                "Average_Temperature": [25, 30],
                "Crop_Duration_days": [120, 140],
            }
        )
        cei = CropEnvironmentInteraction()
        result = cei.crop_duration_temperature(df.copy())
        assert "Crop_Duration_x_Temp" in result.columns

    def test_soil_climate_interaction(self):
        df = pd.DataFrame(
            {
                "Soil_Fertility_Index": [0.8, 0.5],
                "Average_Temperature": [25, 30],
                "Rainfall_mm": [800, 1200],
            }
        )
        cei = CropEnvironmentInteraction()
        result = cei.soil_climate_interaction(df.copy())
        assert "Soil_Climate_Interaction" in result.columns

    def test_compute_all(self):
        df = pd.DataFrame(
            {
                "Average_Temperature": [25, 30],
                "Crop_Duration_days": [120, 140],
                "Rainfall_mm": [800, 1200],
                "Soil_Fertility_Index": [0.8, 0.5],
                "Nitrogen_kg_ha": [120, 150],
                "Phosphorus_kg_ha": [50, 30],
                "Potassium_kg_ha": [250, 180],
                "Yield_per_Hectare": [4.5, 6.0],
            }
        )
        cei = CropEnvironmentInteraction()
        result = cei.compute_all(df)
        generated = [c for c in result.columns if c not in df.columns]
        assert len(generated) >= 3
