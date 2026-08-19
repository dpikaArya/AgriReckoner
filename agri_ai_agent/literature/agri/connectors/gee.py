"""Google Earth Engine connector (official EE Python client).

Requires a service account (GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY or
GOOGLE_APPLICATION_CREDENTIALS).  Disabled when the ``ee`` package or
credentials are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord


class GoogleEarthEngineConnector(AgriculturalDataConnector):
    source_name = "Google_Earth_Engine"
    display_name = "Google Earth Engine"
    base_url = "https://earthengine.googleapis.com"
    default_rate_per_minute = 30

    auth = ConnectorAuth(
        env_vars=(
            "GEE_SERVICE_ACCOUNT",
            "GEE_PRIVATE_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ),
        required=True,
    )

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        try:
            import ee  # noqa: F401

            self._ee_available = True
        except ImportError:
            self._ee_available = False
            self.enabled = False
            self.logger.warning("Google_Earth_Engine disabled: the 'ee' package is not installed")

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        if not self._ee_available:
            return []
        # MODIS NDVI and CHIRPS are the canonical public agricultural images.
        return [
            DatasetDescriptor(
                dataset_id="MODIS/061/MOD13A1",
                title="MODIS Terra Vegetation Indices 16-Day (MOD13A1)",
                url="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A1",
                description="16-day 500m NDVI/EVI composites (official GEE catalog).",
                license="NASA data license",
                variable="ndvi",
                files=[],
                metadata={"ee_image": "MODIS/061/MOD13A1"},
            ),
            DatasetDescriptor(
                dataset_id="UCSB-CHG/CHIRPS/DAILY",
                title="CHIRPS daily precipitation (official GEE catalog)",
                url="https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY",
                description="CHIRPS daily 0.05deg precipitation on Google Earth Engine.",
                license="CHC data license",
                variable="precipitation",
                files=[],
                metadata={"ee_image": "UCSB-CHG/CHIRPS/DAILY"},
            ),
        ][:max_results]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        for d in self.search("", 20):
            if d.dataset_id == dataset_id:
                return d
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        if not self._ee_available:
            return None
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        try:
            import ee

            ee.Initialize(project=descriptor.metadata.get("ee_image").split("/")[0])
            image = ee.Image(descriptor.metadata["ee_image"])
            info = image.select(0).getInfo()
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / f"gee_{dataset_id.replace('/', '_')}.json"
            path.write_text(json.dumps(info, default=str), encoding="utf-8")
            return path
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("GEE download failed: %s", exc)
            return None

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        info = json.loads(download_path.read_text(encoding="utf-8"))
        descriptor = self.fetch_metadata(dataset_id)
        return [
            AgriculturalRecord(
                source=self.source_name,
                dataset_id=dataset_id,
                variable=descriptor.variable if descriptor else "ndvi",
                value=None,
                unit=None,
                provenance=download_path.as_posix(),
                license="NASA data license",
                extra={
                    "id": info.get("id"),
                    "bands": len(info.get("bands", []) or []),
                    "rows": 1,
                    "requires_extra": "ee",
                },
            )
        ]
