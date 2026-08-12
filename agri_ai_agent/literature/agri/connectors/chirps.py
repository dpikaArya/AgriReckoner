"""CHIRPS precipitation connector (data.chc.ucsb.edu, official data portal).

Downloads a recent CHIRPS global monthly GeoTIFF into the dataset cache.
Full raster parsing requires the optional ``rasterio`` extra; without it the
connector records the acquisition (provenance + file) as metadata records.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord

_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"


class ChirpsConnector(AgriculturalDataConnector):
    source_name = "CHIRPS"
    display_name = "CHIRPS (UCSB/CHC precipitation)"
    base_url = _BASE
    default_rate_per_minute = 20

    auth = ConnectorAuth(env_vars=(), required=False)

    def _recent_months(self, n: int = 3) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        today = datetime.date.today()
        year, month = today.year, today.month
        for _ in range(n):
            month -= 1
            if month == 0:
                year -= 1
                month = 12
            out.append((str(year), f"{month:02d}"))
        return out

    def _descriptor(self, year: str, month: str) -> DatasetDescriptor:
        filename = f"chirps-v2.0.{year}.{month}.tif.gz"
        url = f"{_BASE}/global_monthly/tifs/{filename}"
        return DatasetDescriptor(
            dataset_id=f"{year}-{month}",
            title=f"CHIRPS v2.0 monthly precipitation {year}-{month}",
            url=url,
            description="CHIRPS (Climate Hazards Center) global monthly precipitation (mm).",
            license="CHC data license",
            variable="precipitation",
            files=[{"url": url, "name": filename, "format": "geotiff-gz"}],
            metadata={"year": year, "month": month},
            temporal={"start": f"{year}-{month}", "end": f"{year}-{month}"},
        )

    def _descriptors(self) -> list[DatasetDescriptor]:
        return [self._descriptor(y, m) for y, m in self._recent_months()]

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        return self._descriptors()[:max_results]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        for d in self._descriptors():
            if d.dataset_id == dataset_id:
                return d
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None or not descriptor.files:
            return None
        file_info = descriptor.files[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / file_info["name"]
        if not path.exists():
            data = self.http.get_bytes(file_info["url"])
            if data is None:
                return None
            path.write_bytes(data)
        return path

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        size_mb = round(download_path.stat().st_size / 1_000_000, 2)
        year, month = dataset_id.split("-", 1)
        parsed = 0
        try:
            import rasterio  # optional extra

            with rasterio.open(download_path) as src:
                band = src.read(1)
                valid = band[band != -9999]
                if valid.size:
                    parsed = valid.size
        except ImportError:
            self.logger.info(
                "rasterio not installed; CHIRPS %s recorded as acquired file only", dataset_id
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("CHIRPS raster parse failed: %s", exc)
        return [
            AgriculturalRecord(
                source=self.source_name,
                dataset_id=dataset_id,
                variable="precipitation",
                value=None,
                unit="mm",
                year=int(year),
                provenance=download_path.as_posix(),
                license="CHC data license",
                extra={
                    "file_size_mb": size_mb,
                    "raster_pixels_parsed": parsed,
                    "rows": 1,
                    "requires_extra": "rasterio" if not parsed else "",
                },
            )
        ]
