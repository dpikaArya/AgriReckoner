import pandas as pd
import pytest

from src.validation.range_validator import RangeValidator


@pytest.fixture
def validator():
    return RangeValidator()


class TestRangeValidator:
    def test_temperature_range(self, validator):
        df = pd.DataFrame({"Temperature": [-60, 0, 25, 70]})
        result = validator.validate(df)
        t = result["Temperature"]
        assert t["out_of_range_count"] == 2
        assert len(t["violations"]) == 2

    def test_rainfall_range(self, validator):
        df = pd.DataFrame({"Rainfall": [-10, 0, 500, 20000]})
        result = validator.validate(df)
        r = result["Rainfall"]
        assert r["out_of_range_count"] == 2

    def test_soil_ph_range(self, validator):
        df = pd.DataFrame({"Soil_pH": [-1, 0, 7, 14, 15]})
        result = validator.validate(df)
        s = result["Soil_pH"]
        assert s["out_of_range_count"] == 2

    def test_lat_lon_range(self, validator):
        lat = pd.Series([-90.0, 0.0, 91.0])
        lon = pd.Series([-180.0, 0.0, 200.0])
        result = validator.validate_coordinates(lat, lon)
        assert result["valid_pairs"] == 2
        assert result["invalid_pairs"] == 1

    def test_temperature_consistency(self, validator):
        df = pd.DataFrame(
            {
                "tmin": [10.0, 20.0, 5.0],
                "tmax": [15.0, 25.0, 3.0],
            }
        )
        result = validator.validate(df)
        tc = result["_temperature_consistency"]
        assert tc["inconsistent_count"] == 1
        assert tc["consistent_count"] == 2

    def test_temperature_consistency_pass(self, validator):
        df = pd.DataFrame(
            {
                "tmin": [10.0, 20.0],
                "tmax": [15.0, 25.0],
            }
        )
        result = validator.validate(df)
        tc = result["_temperature_consistency"]
        assert tc["inconsistent_count"] == 0

    def test_coordinate_validation(self, validator):
        lat = pd.Series([0.0, 100.0, -91.0])
        lon = pd.Series([0.0, 0.0, 0.0])
        result = validator.validate_coordinates(lat, lon)
        assert result["valid_pairs"] == 1
        assert result["invalid_pairs"] == 2

    def test_auto_column_matching(self, validator):
        df = pd.DataFrame(
            {
                "Temperature": [25.0, 30.0],
                "Rainfall": [100.0, 200.0],
            }
        )
        result = validator.validate(df)
        assert "Temperature" in result
        assert "Rainfall" in result

    def test_non_numeric_column(self, validator):
        df = pd.DataFrame({"Temperature": ["hot", "cold"]})
        result = validator.validate(df)
        assert "Temperature" in result
        assert "error" in result["Temperature"]
