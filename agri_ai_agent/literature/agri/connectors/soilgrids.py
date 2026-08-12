"""ISRIC SoilGrids connector (rest.isric.org/soilgrids/v2.0, official REST API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord

_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

_PROPERTIES = ["soc", "phh2o", "clay", "sand", "silt", "bdod", "nitrogen", "cec"]

_PROP_TO_VARIABLE = {
    "soc": "soil_organic_carbon",
    "phh2o": "soil_ph",
    "clay": "soil_clay_pct",
    "sand": "soil_sand_pct",
    "silt": "soil_silt_pct",
    "bdod": "soil_bulk_density",
    "nitrogen": "soil_nitrogen",
    "cec": "cec",
}

_DEFAULT_POINTS = [
    (29.9136, 77.9975, "Haridwar"),
    (31.1048, 77.1734, "Shimla"),
    (28.6139, 77.2090, "Delhi"),
]


class SoilGridsConnector(AgriculturalDataConnector):
    source_name = "SoilGrids"
    display_name = "ISRIC SoilGrids"
    base_url = "https://rest.isric.org/soilgrids/v2.0"
    default_rate_per_minute = 10

    auth = ConnectorAuth(env_vars=(), required=False)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # SoilGrids can be slow; relax the per-request timeout for this source.
        self.http._default_timeout = max(self.config.timeout_sec, 120)

    def _descriptor(self, lat: float, lon: float, label: str) -> DatasetDescriptor:
        url = f"{_API}?lon={lon}&lat={lat}&property={','.join(_PROPERTIES)}"
        return DatasetDescriptor(
            dataset_id=f"{lat:.4f}_{lon:.4f}",
            title=f"SoilGrids soil properties at {label}",
            url=url,
            description="SoilGrids 2.0 predicted soil properties (ISRIC).",
            license="CC BY 4.0",
            variable=None,
            metadata={"lat": lat, "lon": lon, "label": label},
            spatial={"lat": lat, "lon": lon},
        )

    def _points(self) -> list[tuple[float, float, str]]:
        import os

        raw = os.getenv("AGRI_SOILGRIDS_POINTS", "").strip()
        if not raw:
            return list(_DEFAULT_POINTS)
        out: list[tuple[float, float, str]] = []
        for part in raw.split(";"):
            coords = part.split(",")
            if len(coords) >= 2:
                try:
                    out.append((float(coords[0]), float(coords[1]), part))
                except ValueError:
                    continue
        return out or list(_DEFAULT_POINTS)

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        return [self._descriptor(lat, lon, label) for lat, lon, label in self._points()[:max_results]]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        for lat, lon, label in self._points():
            if f"{lat:.4f}_{lon:.4f}" == dataset_id:
                return self._descriptor(lat, lon, label)
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        meta = descriptor.metadata
        payload = self.http.get_json(
            "/properties/query",
            params={"lon": meta["lon"], "lat": meta["lat"], "property": ",".join(_PROPERTIES)},
        )
        # The v2.0 service may reject combined property queries (HTTP 500);
        # fall back to one property per request when that happens.
        if not isinstance(payload, dict):
            combined_layers: list = []
            for prop in _PROPERTIES:
                single = self.http.get_json(
                    "/properties/query",
                    params={"lon": meta["lon"], "lat": meta["lat"], "property": prop},
                )
                if isinstance(single, dict):
                    combined_layers.extend((single.get("properties", {}) or {}).get("layers", []) or [])
            if combined_layers:
                payload = {"properties": {"layers": combined_layers}}
        if not isinstance(payload, dict):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"soilgrids_{dataset_id}.json"
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        import json

        payload = json.loads(download_path.read_text(encoding="utf-8"))
        meta = self.fetch_metadata(dataset_id)
        lat = meta.metadata["lat"] if meta else None
        lon = meta.metadata["lon"] if meta else None

        # Phase 18: the v2.0 service now returns a "layers" array under
        # properties, e.g. properties.layers[].name / .unit_measure / .depths[].
        layers = ((payload.get("properties", {}) or {}).get("layers") or [])
        records: list[AgriculturalRecord] = []
        if layers:
            for layer in layers:
                prop = layer.get("name")
                variable = _PROP_TO_VARIABLE.get(prop)
                if variable is None:
                    continue
                um = layer.get("unit_measure") or {}
                d_factor = um.get("d_factor", 1)
                target_units = um.get("target_units") or um.get("mapped_units") or "unitless"
                mean, depth_label = None, None
                for depth in (layer.get("depths") or []):
                    vals = depth.get("values") or {}
                    m = vals.get("mean")
                    if m is None:
                        continue
                    mean = m
                    depth_label = depth.get("label") or ""
                    break
                if mean is None:
                    continue
                try:
                    scaled = float(mean) / float(d_factor)
                except (TypeError, ValueError):
                    continue
                records.append(
                    AgriculturalRecord(
                        source=self.source_name,
                        dataset_id=f"{dataset_id}::{prop}",
                        variable=variable,
                        value=round(scaled, 4),
                        unit=target_units,
                        location_lat=lat,
                        location_lon=lon,
                        provenance=f"SoilGrids v2.0 property={prop} depth={depth_label}",
                        license="CC BY 4.0",
                        extra={"property": prop, "depth": depth_label,
                               "mapped_units": um.get("mapped_units"), "d_factor": d_factor},
                    )
                )
            return records

        # Legacy response format: properties.<prop>.depth.<depth>.values[].mean
        props = payload.get("properties", {}) or {}
        for prop, data in props.items():
            variable = _PROP_TO_VARIABLE.get(prop)
            if variable is None:
                continue
            units = data.get("units")
            mean = None
            depth_label = None
            for depth, layer in (data.get("depth") or {}).items():
                values = [v.get("mean") for v in (layer.get("values") or []) if v.get("mean") is not None]
                if values:
                    mean = sum(values) / len(values)
                    depth_label = depth
                    break
            if mean is None:
                continue
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=f"{dataset_id}::{prop}",
                    variable=variable,
                    value=round(float(mean), 4),
                    unit=units,
                    location_lat=lat,
                    location_lon=lon,
                    provenance=f"SoilGrids v2.0 property={prop} depth={depth_label}",
                    license="CC BY 4.0",
                    extra={"property": prop, "depth": depth_label},
                )
            )
        return records
