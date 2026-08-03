"""Unit tests for agricultural data connectors (parsing, units, records)."""

from __future__ import annotations

import json

from agri_ai_agent.literature.agri.connector import (
    convert_unit,
    normalize_unit,
)
from agri_ai_agent.literature.agri.connectors.power import NasaPowerConnector
from agri_ai_agent.literature.agri.connectors.worldbank import WorldBankConnector
from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.literature.models import AgriculturalRecord
from agri_ai_agent.literature.state import ConnectorStateStore


def _make_connector(connector_cls, tmp_path):
    cfg = LiteratureConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        state_db=tmp_path / "state.sqlite",
        cache_dir=tmp_path / "cache",
    )
    state = ConnectorStateStore(cfg.state_db)
    return connector_cls(cfg, state)


# ---------------------------------------------------------------------- #
# unit normalization / conversion
# ---------------------------------------------------------------------- #
def test_normalize_unit():
    assert normalize_unit("kg/ha") == "kg/ha"
    assert normalize_unit("t/ha") == "kg/ha"
    assert normalize_unit("ton/ha") == "kg/ha"
    assert normalize_unit("g/m2") == "kg/ha"
    assert normalize_unit("percent") == "%"
    assert normalize_unit("ppm") == "mg/kg"
    assert normalize_unit(None) is None


def test_convert_unit():
    assert convert_unit(2.5, "t/ha", "kg/ha") == 2500.0
    assert convert_unit(100.0, "kg/ha", "kg/ha") == 100.0
    assert convert_unit(50.0, "g/m2", "kg/ha") == 500.0
    # unknown pair is left untouched
    assert convert_unit(7.0, "m", "kg/ha") == 7.0


# ---------------------------------------------------------------------- #
# base record validation
# ---------------------------------------------------------------------- #
def test_agri_validate_record(tmp_path):
    connector = _make_connector(WorldBankConnector, tmp_path)
    good = AgriculturalRecord(
        source="World_Bank", dataset_id="X", variable="yield", value=3.1
    )
    assert connector.validate_record(good) == []

    bad = AgriculturalRecord(source="", dataset_id="", variable="", value=None)
    assert len(connector.validate_record(bad)) >= 2


# ---------------------------------------------------------------------- #
# World Bank parsing
# ---------------------------------------------------------------------- #
def test_worldbank_to_records(tmp_path):
    connector = _make_connector(WorldBankConnector, tmp_path)
    path = tmp_path / "worldbank.json"
    payload = [
        {"page": 1, "pages": 1},
        [
            {"date": "2023", "country": {"value": "India"}, "countryiso3code": "IND", "value": 153868700.0},
            {"date": "2023", "country": {"value": "India"}, "countryiso3code": "IND", "value": None},
            {"date": "2022", "country": {"value": "India"}, "countryiso3code": "IND", "value": 154027533.3},
        ],
    ]
    path.write_text(json.dumps([payload]), encoding="utf-8")

    records = connector.to_records("AG.LND.ARBL.HA", path)
    assert len(records) == 2
    assert records[0].variable == "arable_land_ha"
    assert records[0].unit == "ha"
    assert records[0].country == "India"
    assert records[0].year == 2023
    assert records[0].extra["iso3"] == "IND"
    assert records[0].provenance.startswith("World Bank indicator")


def test_worldbank_search_falls_back_to_full_catalogue(tmp_path):
    connector = _make_connector(WorldBankConnector, tmp_path)
    # a query with no token matches must still return the fixed indicator set
    hits = connector.search("zebra unicorn")
    assert len(hits) == 5
    assert all(d.dataset_id for d in hits)


def test_worldbank_descriptor_metadata(tmp_path):
    connector = _make_connector(WorldBankConnector, tmp_path)
    descriptor = connector.fetch_metadata("AG.LND.ARBL.HA")
    assert descriptor is not None
    assert descriptor.variable == "arable_land_ha"
    assert descriptor.license == "World Bank terms of use"


# ---------------------------------------------------------------------- #
# NASA POWER parsing
# ---------------------------------------------------------------------- #
def test_power_to_records(tmp_path):
    connector = _make_connector(NasaPowerConnector, tmp_path)
    payload = {
        "geometry": {"coordinates": [77.9975, 29.9136]},
        "properties": {
            "parameter": {
                "T2M": {"20260501": 30.1, "20260502": None, "20260503": 31.0},
                "PRECTOTCORR": {"20260501": 0.0, "20260502": 5.0, "20260503": -999.0},
                "UNKNOWN": {"20260501": 1.0},
            }
        },
    }
    path = tmp_path / "power.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = connector.to_records("29.9136_77.9975", path)
    variables = {r.variable: r for r in records}
    assert "average_temperature" in variables
    assert variables["average_temperature"].value == 30.55  # None skipped
    assert variables["average_temperature"].unit == "degC"
    assert variables["precipitation"].value == 2.5  # -999 sentinel skipped
    assert variables["precipitation"].unit == "mm/day"
    assert "UNKNOWN" not in variables
    assert records[0].location_lat == 29.9136
    assert records[0].location_lon == 77.9975


def test_power_to_records_missing_file(tmp_path):
    connector = _make_connector(NasaPowerConnector, tmp_path)
    assert connector.to_records("x", tmp_path / "missing.json") == []


def test_power_descriptor_spatial(tmp_path):
    connector = _make_connector(NasaPowerConnector, tmp_path)
    d = connector.search("agriculture", max_results=1)[0]
    assert "lat" in d.spatial and "lon" in d.spatial
    assert d.license == "NASA OPEN DATA"
    assert d.url.startswith("https://power.larc.nasa.gov")
