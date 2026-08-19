"""Base connector for the Copernicus Climate Data Store (CDS).

Uses the official ``cdsapi`` Python client against the production CDS API
(``https://cds.climate.copernicus.eu/api``) with a personal access token.
Credentials come from ``CDSAPI_URL`` / ``CDSAPI_KEY`` or a standard
``~/.cdsapirc`` file.  Without valid credentials the connector is disabled.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord

# Official CDS dataset catalogue identifiers.
CDS_DATASETS: dict[str, dict[str, Any]] = {
    "reanalysis-era5-single-levels": {
        "title": "ERA5 hourly data on single levels from 1940 to present",
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        "description": "ECMWF ERA5 reanalysis, single levels (official Copernicus CDS dataset).",
        "license": "Licence to Use Copernicus Products",
        "variables": [
            "2m_temperature",
            "total_precipitation",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_solar_radiation_downwards",
        ],
        "request_keys": {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "time": "00:00",
        },
    },
    "sis-agrometeorological-indicators": {
        "title": "AgERA5: ECMWF Agrometeorological Indicators from 1979 to present",
        "url": "https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators",
        "description": (
            "AgERA5 daily agrometeorological indicators (temperature, precipitation, "
            "radiation, humidity) from ECMWF, official CDS dataset."
        ),
        "license": "Licence to Use Copernicus Products",
        "variables": [
            "2m_temperature",
            "2m_temperature_maximum",
            "2m_temperature_minimum",
            "precipitation_flux",
            "surface_net_solar_radiation",
        ],
        "request_keys": {
            "data_format": "zip",
        },
    },
}


class CDSConnector(AgriculturalDataConnector):
    """Copernicus CDS base connector; subclasses select dataset + variables."""

    cds_dataset: str = "reanalysis-era5-single-levels"
    source_name = "Copernicus_CDS"
    display_name = "Copernicus Climate Data Store"
    base_url = "https://cds.climate.copernicus.eu"
    auth = ConnectorAuth(
        env_vars=("CDSAPI_URL", "CDSAPI_KEY"),
        required=True,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cdsapi_import_error: Exception | None = None
        self._client = None
        if self.enabled:
            try:
                import cdsapi  # type: ignore

                url = self.auth.resolve_env("CDSAPI_URL") or "https://cds.climate.copernicus.eu/api"
                key = self.auth.resolve_env("CDSAPI_KEY") or os.getenv("CDSAPI_KEY")
                self._client = cdsapi.Client(url=url, key=key)
            except ImportError as exc:
                self._cdsapi_import_error = exc
                self.logger.warning(
                    "cdsapi not installed; %s connector cannot download (pip install cdsapi)",
                    self.source_name,
                )
            except Exception as exc:  # noqa: BLE001
                self._cdsapi_import_error = exc
                self.logger.warning("Failed to initialize cdsapi client: %s", exc)

    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        entry = CDS_DATASETS.get(self.cds_dataset)
        if entry is None:
            return []
        # simple keyword filter against the official catalogue title/description
        haystack = (entry["title"] + " " + entry["description"]).lower()
        if query and all(word not in haystack for word in query.lower().split()):
            return []
        return [
            DatasetDescriptor(
                dataset_id=self.cds_dataset,
                title=entry["title"],
                url=entry["url"],
                description=entry["description"],
                license=entry["license"],
                variable=", ".join(entry["variables"]),
                files=[],
                metadata=entry,
            )
        ]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        entry = CDS_DATASETS.get(dataset_id) or CDS_DATASETS.get(self.cds_dataset)
        if entry is None:
            return None
        return DatasetDescriptor(
            dataset_id=dataset_id,
            title=entry["title"],
            url=entry["url"],
            description=entry["description"],
            license=entry["license"],
            variable=", ".join(entry["variables"]),
            metadata=entry,
        )

    def _request_payload(
        self, year: int, month: int, day: int, variables: list[str]
    ) -> dict[str, Any]:
        payload = dict(CDS_DATASETS[self.cds_dataset]["request_keys"])
        payload.update(
            {
                "variable": variables[:1],
                "year": str(year),
                "month": str(month).zfill(2),
                "day": str(day).zfill(2),
                "area": [30, 70, 20, 90],  # small bounding box (south, west, north, east)
            }
        )
        return payload

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        if self._client is None:
            self.logger.warning(
                "CDS client unavailable (%s); cannot download %s",
                self._cdsapi_import_error,
                dataset_id,
            )
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        variables = CDS_DATASETS.get(dataset_id, {}).get("variables", ["2m_temperature"])
        # Minimal single-day test request keeps the first run cheap.
        payload = self._request_payload(2023, 1, 1, variables)
        target = target_dir / f"{dataset_id}_sample.netcdf"
        try:
            self._client.retrieve(dataset_id, payload, str(target))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("CDS retrieve failed for %s: %s", dataset_id, exc)
            return None
        return target if target.exists() and target.stat().st_size > 0 else None

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        """Parse a CDS download into AgriculturalRecords when possible;
        otherwise emit a provenance-only record."""
        entry = CDS_DATASETS.get(dataset_id, {})
        if download_path is not None and download_path.suffix.lower() in (".nc", ".netcdf"):
            try:
                import xarray as xr  # type: ignore

                ds = xr.open_dataset(download_path)
                records: list[AgriculturalRecord] = []
                for var_name in entry.get("variables", []):
                    if var_name not in ds:
                        continue
                    da = ds[var_name]
                    mean = float(da.mean().values) if da.size else None
                    records.append(
                        AgriculturalRecord(
                            source=self.source_name,
                            dataset_id=dataset_id,
                            variable=var_name,
                            value=mean,
                            unit=_unit_for(var_name),
                            provenance=f"{self.base_url}/datasets/{dataset_id}",
                            license=entry.get("license"),
                            extra={
                                "download_path": str(download_path),
                                "file_size_bytes": download_path.stat().st_size,
                                "reduction": "spatial-temporal mean",
                            },
                        )
                    )
                if records:
                    return records
            except ImportError:
                self.logger.warning("xarray not installed; emitting provenance-only record")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to parse CDS netcdf: %s", exc)
        return [
            AgriculturalRecord(
                source=self.source_name,
                dataset_id=dataset_id,
                variable=entry.get("variables", ["2m_temperature"])[0],
                value=None,
                unit=None,
                provenance=f"{self.base_url}/datasets/{dataset_id}",
                license=entry.get("license"),
                extra={
                    "download_path": str(download_path) if download_path else None,
                    "note": "metadata record; dataset download requires CDS account + licence acceptance",
                },
            )
        ]


def _unit_for(var_name: str) -> str | None:
    if "temperature" in var_name:
        return "degC"
    if "precipitation" in var_name or "radiation" in var_name:
        return "W/m2" if "radiation" in var_name else "kg/m2/s"
    return None


class ERA5Connector(CDSConnector):
    source_name = "ERA5"
    display_name = "ERA5 (Copernicus CDS)"
    cds_dataset = "reanalysis-era5-single-levels"


class AgERA5Connector(CDSConnector):
    source_name = "AgERA5"
    display_name = "AgERA5 Agrometeorological Indicators (Copernicus CDS)"
    cds_dataset = "sis-agrometeorological-indicators"
