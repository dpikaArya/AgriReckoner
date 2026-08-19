"""World Bank Open Data / Climate Change connectors (api.worldbank.org)."""

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

_API = "https://api.worldbank.org/v2"

# indicator -> (UAMS variable, unit)
_INDICATORS = {
    "AG.LND.ARBL.HA": ("arable_land_ha", "ha"),
    "AG.CON.FERT.ZS": ("fertilizer_consumption", "kg/ha"),
    "AG.PRD.CROP.XD": ("crop_production_index", "index"),
    "AG.PRD.FOOD.XD": ("food_production_index", "index"),
    "SP.RUR.TOTL": ("rural_population", "persons"),
}

# Major agriculture producers / economies — country-specific requests are fast
# and return real values (the country/ALL aggregate endpoint is slow and
# returns nulls for the most recent reported years).
_COUNTRIES = ["IND", "CHN", "USA", "BRA", "NGA", "PAK", "BGD", "IDN", "RUS", "UKR", "FRA", "DEU"]


def _countries() -> list[str]:
    import os

    raw = os.getenv("AGRI_WORLDBANK_COUNTRIES", "").strip()
    if raw:
        return [c.strip().upper() for c in raw.split(",") if c.strip()]
    return list(_COUNTRIES)


class WorldBankConnector(AgriculturalDataConnector):
    source_name = "World_Bank"
    display_name = "World Bank Open Data"
    base_url = _API
    default_rate_per_minute = 20

    auth = ConnectorAuth(env_vars=(), required=False)

    def _descriptor(self, indicator: str, variable: str) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id=indicator,
            title=f"World Bank indicator {indicator} ({variable})",
            url=f"{_API}/country/IND/indicator/{indicator}?format=json",
            description="Official World Bank development indicator (country-year series).",
            license="World Bank terms of use",
            variable=variable,
            files=[],
            metadata={"indicator": indicator, "variable": variable},
        )

    def _descriptors(self) -> list[DatasetDescriptor]:
        return [self._descriptor(ind, var) for ind, (var, _unit) in _INDICATORS.items()]

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        tokens = query.lower().split()
        hits = [
            d
            for d in self._descriptors()
            if all(t in d.title.lower() for t in tokens) or query in d.dataset_id
        ]
        return (hits or self._descriptors())[:max_results]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        for d in self._descriptors():
            if d.dataset_id == dataset_id:
                return d
        return None

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        pages: list[Any] = []
        for iso3 in _countries():
            payload = self.http.get_json(
                f"/country/{iso3}/indicator/{dataset_id}",
                params={"format": "json", "per_page": 50, "date": "2015:2024"},
                timeout=90,
            )
            if isinstance(payload, list) and len(payload) > 1:
                pages.append(payload)
        if not pages:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"worldbank_{dataset_id}.json"
        path.write_text(json.dumps(pages), encoding="utf-8")
        return path

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        pages = json.loads(download_path.read_text(encoding="utf-8"))
        descriptor = self.fetch_metadata(dataset_id)
        variable, unit = (
            _INDICATORS.get(dataset_id, (descriptor.variable, None))[:2]
            if descriptor
            else (None, None)
        )
        records: list[AgriculturalRecord] = []
        for payload in pages or []:
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            for row in payload[1] or []:
                value = _to_float(row.get("value"))
                if value is None:
                    continue
                records.append(
                    AgriculturalRecord(
                        source=self.source_name,
                        dataset_id=dataset_id,
                        variable=variable,
                        value=value,
                        unit=unit,
                        year=_to_int(row.get("date")),
                        country=row.get("country", {}).get("value"),
                        provenance=f"World Bank indicator {dataset_id}",
                        license="World Bank terms of use",
                        extra={"iso3": row.get("countryiso3code")},
                    )
                )
        return records


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
