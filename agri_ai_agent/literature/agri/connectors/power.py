"""NASA POWER connector (power.larc.nasa.gov, official public API)."""

from __future__ import annotations

from pathlib import Path

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord

_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
_API_ENDPOINT = "/api/temporal/daily/point"

# UAMS variable name per POWER parameter.
_PARAM_TO_VARIABLE = {
    "T2M": "average_temperature",
    "T2M_MAX": "temperature_max",
    "T2M_MIN": "temperature_min",
    "PRECTOTCORR": "precipitation",
    "RH2M": "humidity",
    "WS2M": "wind_speed",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
    "ET0": "evapotranspiration",
}

_DEFAULT_STATIONS = [
    (29.9136, 77.9975, "Haridwar, Uttarakhand"),
    (31.1048, 77.1734, "Shimla, Himachal Pradesh"),
    (13.0827, 80.2707, "Chennai, Tamil Nadu"),
]


def _stations() -> list[tuple[float, float, str]]:
    import os

    raw = os.getenv("AGRI_POWER_STATIONS", "").strip()
    out: list[tuple[float, float, str]] = []
    if raw:
        for part in raw.split(";"):
            coords = part.split(",")
            if len(coords) >= 2:
                try:
                    out.append((float(coords[0]), float(coords[1]), part))
                except ValueError:
                    continue
    return out or list(_DEFAULT_STATIONS)


class NasaPowerConnector(AgriculturalDataConnector):
    source_name = "NASA_POWER"
    display_name = "NASA POWER (Agroclimatology)"
    base_url = "https://power.larc.nasa.gov"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    def _descriptor(self, lat: float, lon: float, label: str, start: str, end: str) -> DatasetDescriptor:
        url = (
            f"{_API}?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,"
            f"ALLSKY_SFC_SW_DWN&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}&format=JSON"
        )
        return DatasetDescriptor(
            dataset_id=f"{lat:.4f}_{lon:.4f}",
            title=f"NASA POWER daily agroclimatology at {label}",
            url=url,
            description="Daily meteorology for agriculture (NASA POWER, AG community).",
            license="NASA OPEN DATA",
            variable=None,
            metadata={"lat": lat, "lon": lon, "label": label, "start": start, "end": end},
            spatial={"lat": lat, "lon": lon},
            temporal={"start": start, "end": end},
        )

    def _query_points(self) -> dict[str, tuple[float, float, str]]:
        import datetime

        end = datetime.date.today().replace(day=1)
        start = end - datetime.timedelta(days=91)
        window = (start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        descriptors: dict[str, tuple[float, float, str]] = {}
        for lat, lon, label in _stations():
            d = self._descriptor(lat, lon, label, *window)
            descriptors[d.dataset_id] = (lat, lon, label)
        return descriptors

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        points = self._query_points()
        return [
            self._descriptor(lat, lon, label, *self._window())
            for lat, lon, label in list(points.values())[:max_results]
        ]

    def _window(self) -> tuple[str, str]:
        import datetime

        end = datetime.date.today().replace(day=1)
        start = end - datetime.timedelta(days=91)
        return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        points = self._query_points()
        if dataset_id not in points:
            return None
        lat, lon, label = points[dataset_id]
        start, end = self._window()
        return self._descriptor(lat, lon, label, start, end)

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        meta = descriptor.metadata
        payload = self.http.get_json(
            _API_ENDPOINT,
            params={
                "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN",
                "community": "AG",
                "longitude": meta["lon"],
                "latitude": meta["lat"],
                "start": meta["start"],
                "end": meta["end"],
                "format": "JSON",
            },
            use_cache=False,
        )
        if not isinstance(payload, dict):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"power_{dataset_id}.json"
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        import json

        payload = json.loads(download_path.read_text(encoding="utf-8"))
        params = payload.get("properties", {}).get("parameter", {})
        lat = payload.get("geometry", {}).get("coordinates", [None, None])[1]
        lon = payload.get("geometry", {}).get("coordinates", [None, None])[0]
        records: list[AgriculturalRecord] = []
        for param, days in params.items():
            variable = _PARAM_TO_VARIABLE.get(param)
            if variable is None:
                continue
            values = [v for v in days.values() if v is not None and v != -999]
            if not values:
                continue
            mean = sum(values) / len(values)
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=f"{dataset_id}::{param}",
                    variable=variable,
                    value=round(mean, 4),
                    unit=_unit_for(param),
                    location_lat=lat,
                    location_lon=lon,
                    provenance=f"NASA POWER /api/temporal/daily/point parameter={param}",
                    license="NASA OPEN DATA",
                    extra={"parameter": param, "days": len(values)},
                )
            )
        return records


def _unit_for(param: str) -> str:
    return {
        "T2M": "degC",
        "T2M_MAX": "degC",
        "T2M_MIN": "degC",
        "PRECTOTCORR": "mm/day",
        "RH2M": "%",
        "WS2M": "m/s",
        "ALLSKY_SFC_SW_DWN": "kWh/m2/day",
        "ET0": "mm/day",
    }.get(param)
