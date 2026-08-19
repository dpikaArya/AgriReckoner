"""USDA NASS Quick Stats connector (quickstats.nass.usda.gov, official API)."""

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

_API = "https://quickstats.nass.usda.gov/api/api_GET/"

_STAT_TO_VARIABLE = {
    "YIELD": "yield_per_hectare",
    "PRODUCTION": "production_tonnes",
    "AREA HARVESTED": "area_harvested",
    "AREA PLANTED": "area_planted",
}

# BU / ACRE -> kg/ha per commodity (official conversion factors).
_GRAIN_CONVERSION = {
    "CORN": 62.77,
    "WHEAT": 67.25,
    "SOYBEANS": 67.25,
    "SORGHUM": 53.74,
    "BARLEY": 53.74,
    "RICE": 53.74,
    "OATS": 32.15,
}


class NassConnector(AgriculturalDataConnector):
    source_name = "USDA_NASS"
    display_name = "USDA NASS Quick Stats"
    base_url = _API
    default_rate_per_minute = 20

    auth = ConnectorAuth(env_vars=("NASS_API_KEY", "AGRI_NASS_API_KEY"), required=True)

    def _key(self) -> str:
        return self.auth.resolve() or ""

    def _headers(self) -> dict[str, str]:
        return {}

    def _descriptor(self, commodity: str, statistic: str) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id=f"{commodity}::{statistic}",
            title=f"USDA NASS Quick Stats - {commodity} {statistic}",
            url="https://quickstats.nass.usda.gov",
            description="Official USDA NASS Quick Stats survey data.",
            license="Public domain (USDA)",
            variable=None,
            files=[],
            metadata={"commodity": commodity, "statisticcat_desc": statistic},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        commodity = query.strip().upper().replace(" ", "_")
        return [
            self._descriptor(commodity, "YIELD"),
            self._descriptor(commodity, "PRODUCTION"),
            self._descriptor(commodity, "AREA HARVESTED"),
        ]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        if "::" in dataset_id:
            commodity, statistic = dataset_id.split("::", 1)
            return self._descriptor(commodity, statistic)
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        params = {
            "key": self._key(),
            "commodity_desc": descriptor.metadata["commodity"],
            "statisticcat_desc": descriptor.metadata["statisticcat_desc"],
            "year__GE": "2015",
            "format": "JSON",
        }
        payload = self.http.get_json("", params=params, use_cache=False)
        if not isinstance(payload, dict):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"nass_{dataset_id.replace('::', '_')}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        payload = json.loads(download_path.read_text(encoding="utf-8"))
        data = payload.get("data", []) or []
        commodity = (payload.get("source_desc", "") or "").split(" ")[0]
        descriptor = self.fetch_metadata(dataset_id)
        commodity = descriptor.metadata["commodity"] if descriptor else commodity
        statistic = descriptor.metadata["statisticcat_desc"] if descriptor else "YIELD"
        variable = _STAT_TO_VARIABLE.get(statistic, "yield_per_hectare")
        records: list[AgriculturalRecord] = []
        for row in data[:5000]:
            value = _to_float(row.get("Value"))
            if value is None:
                continue
            unit = row.get("unit_desc")
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=f"{dataset_id}",
                    variable=variable,
                    value=value,
                    unit=unit,
                    year=_to_int(row.get("year")),
                    crop=commodity,
                    country="USA",
                    state=str(row.get("state_name", "")) or None,
                    provenance=f"NASS Quick Stats commodity={commodity} statistic={statistic}",
                    license="Public domain (USDA)",
                    extra={
                        "state_alpha": row.get("state_alpha"),
                        "county": row.get("county_name"),
                        "class_desc": row.get("class_desc"),
                        "unit": unit,
                    },
                )
            )
        return records


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "(D)", "(L)", "(Z)") else None
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
