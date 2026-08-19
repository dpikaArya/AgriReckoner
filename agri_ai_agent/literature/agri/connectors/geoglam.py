"""GEOGLAM Crop Monitor connector (earthmap.org API).

The GEOGLAM Crop Monitor for Early Warning publishes its official indicator
maps through the EarthMap platform API (``https://earthmap.org/api``).  The
API requires an API key; without one the connector is disabled.
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


class GeoglamConnector(AgriculturalDataConnector):
    source_name = "GEOGLAM"
    display_name = "GEOGLAM Crop Monitor"
    base_url = "https://earthmap.org/api"
    default_rate_per_minute = 20

    auth = ConnectorAuth(
        env_vars=("EARTHMAP_API_KEY", "AGRI_GEOGLAM_API_KEY"),
        required=True,
        header="X-Api-Key",
    )

    def _descriptor(self, endpoint: str, label: str) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id=endpoint,
            title=f"GEOGLAM Crop Monitor - {label}",
            url=f"{self.base_url}/{endpoint}",
            description="GEOGLAM Crop Monitor for Early Warning official indicator data.",
            license="GEOGLAM terms of use",
            variable="crop_condition",
            files=[],
            metadata={"endpoint": endpoint},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        return [
            self._descriptor("cropreports", "crop reports"),
            self._descriptor("indicators", "indicators"),
        ][:max_results]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        for d in self.search("", 20):
            if d.dataset_id == dataset_id:
                return d
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        payload = self.http.get_json(f"/{dataset_id}", use_cache=False)
        if not isinstance(payload, (dict, list)):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"geoglam_{dataset_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        payload = json.loads(download_path.read_text(encoding="utf-8"))
        records: list[AgriculturalRecord] = []
        items = payload if isinstance(payload, list) else [payload]
        for idx, item in enumerate(items[:2000]):
            country = item.get("country") or item.get("region")
            indicator = item.get("indicator") or item.get("name") or f"record-{idx}"
            condition = item.get("condition") or item.get("crop_condition")
            value = _to_float(condition if isinstance(condition, (int, float)) else None)
            if value is None and condition is not None:
                value = _condition_score(str(condition))
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=dataset_id,
                    variable="crop_condition",
                    value=value,
                    unit=None,
                    year=_to_int(item.get("year")),
                    crop=item.get("crop") or item.get("crop_name"),
                    country=country,
                    provenance=f"GEOGLAM earthmap /api/{dataset_id}",
                    license="GEOGLAM terms of use",
                    extra={"condition": condition, "indicator": indicator},
                )
            )
        return records


def _condition_score(condition: str) -> float:
    mapping = {
        "favourable": 5.0,
        "favorable": 5.0,
        "good": 4.0,
        "watch": 3.0,
        "poor": 2.0,
        "severe": 1.0,
    }
    return mapping.get(condition.strip().lower(), None)


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None
