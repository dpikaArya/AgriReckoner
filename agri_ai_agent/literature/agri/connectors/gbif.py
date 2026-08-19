"""GBIF occurrence connector (api.gbif.org, official public API)."""

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

_API = "https://api.gbif.org/v1/occurrence/search"


class GbifConnector(AgriculturalDataConnector):
    source_name = "GBIF"
    display_name = "GBIF (biodiversity occurrences)"
    base_url = "https://api.gbif.org/v1"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    def _descriptor(self, query: str, country: str | None) -> DatasetDescriptor:
        country = country or "IN"
        return DatasetDescriptor(
            dataset_id=f"{query}::{country}",
            title=f"GBIF occurrences for '{query}' in {country}",
            url=f"{_API}?q={query}&country={country}&limit=25",
            description="Cultivated crop occurrence records from the GBIF network.",
            license="CC0 / dataset-dependent",
            variable="occurrence",
            files=[],
            metadata={"q": query, "country": country},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        return [self._descriptor(query, "IN")]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        if "::" in dataset_id:
            q, country = dataset_id.split("::", 1)
            return self._descriptor(q, country)
        return self._descriptor(dataset_id, "IN")

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        payload = self.http.get_json(
            "/occurrence/search",
            params={
                "q": descriptor.metadata["q"],
                "country": descriptor.metadata["country"],
                "limit": 25,
            },
        )
        if not isinstance(payload, dict):
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"gbif_{dataset_id.replace('::', '_')}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        payload = json.loads(download_path.read_text(encoding="utf-8"))
        records: list[AgriculturalRecord] = []
        for occ in payload.get("results", []) or []:
            species = occ.get("species") or occ.get("scientificName") or occ.get("taxonKey", "")
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=dataset_id,
                    variable="occurrence",
                    value=1.0,
                    unit=None,
                    location_lat=occ.get("decimalLatitude"),
                    location_lon=occ.get("decimalLongitude"),
                    year=_to_int(occ.get("year")),
                    crop=species,
                    country=occ.get("country"),
                    provenance=f"GBIF occurrence {occ.get('key', '')}",
                    license="CC0 / dataset-dependent",
                    extra={
                        "datasetKey": occ.get("datasetKey"),
                        "speciesKey": occ.get("speciesKey"),
                        "basisOfRecord": occ.get("basisOfRecord"),
                    },
                )
            )
        return records


def _to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
