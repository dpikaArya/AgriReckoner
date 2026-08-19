"""MapSPAM (Spatial Production Allocation Model) connector.

Official gridded maps are distributed by IFPRI/HarvestChoice on Zenodo and the
HarvestChoice dataverse.  This connector discovers the official MapSPAM
deposits on Zenodo and downloads tabular/geotiff assets.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord


class MapSpamConnector(AgriculturalDataConnector):
    source_name = "MapSPAM"
    display_name = "MapSPAM (IFPRI spatial production)"
    base_url = "https://zenodo.org/api/records"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    def _records(self, page: int = 1, size: int = 10) -> list[dict[str, Any]]:
        payload = self.http.get_json(
            "",
            params={"q": "communities:ifpri MapSPAM", "size": size, "page": page},
        )
        if not isinstance(payload, dict):
            return []
        return list(payload.get("hits", {}).get("hits", []) or [])

    def _descriptor(self, record: dict[str, Any]) -> DatasetDescriptor | None:
        rec_id = record.get("id")
        if not rec_id:
            return None
        metadata = record.get("metadata", {}) or {}
        files = [
            {
                "url": f.get("links", {}).get("self"),
                "name": f.get("key"),
                "format": (f.get("key", "") or "").rsplit(".", 1)[-1],
            }
            for f in record.get("files", []) or []
            if f.get("links", {}).get("self")
        ]
        return DatasetDescriptor(
            dataset_id=str(rec_id),
            title=metadata.get("title") or f"MapSPAM record {rec_id}",
            url=record.get("links", {}).get("html") or f"https://zenodo.org/records/{rec_id}",
            description=(metadata.get("description") or "")[:500],
            license=metadata.get("license", {}).get("id"),
            variable="harvested_area",
            files=files,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        out: list[DatasetDescriptor] = []
        for record in self._records(size=max_results):
            descriptor = self._descriptor(record)
            if descriptor:
                out.append(descriptor)
        return out

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        payload = self.http.get_json(f"/records/{dataset_id}")
        if not isinstance(payload, dict):
            return None
        return self._descriptor(payload)

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        chosen = next(
            (f for f in descriptor.files if (f.get("name") or "").lower().endswith(".zip")),
            descriptor.files[0] if descriptor.files else None,
        )
        if chosen is None or not chosen.get("url"):
            return None
        path = target_dir / f"mapspam_{dataset_id}"
        data = self.http.get_bytes(chosen["url"])
        if data is None:
            return None
        path.write_bytes(data)
        return path

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        df = None
        try:
            if (download_path.name or "").lower().endswith(".zip"):
                with zipfile.ZipFile(download_path) as zf:
                    inner = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                    with zf.open(inner) as src:
                        df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
            else:
                df = pd.read_csv(download_path, encoding="utf-8-sig", low_memory=False)
        except Exception:  # noqa: BLE001
            return []
        if df is None:
            return []
        records: list[AgriculturalRecord] = []
        for col in df.columns:
            if "area" not in str(col).lower():
                continue
            total = pd.to_numeric(df[col], errors="coerce").sum()
            if pd.isna(total):
                continue
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=dataset_id,
                    variable="harvested_area",
                    value=round(float(total), 4),
                    unit="ha",
                    provenance=download_path.as_posix(),
                    license="MapSPAM data license",
                    extra={"layer": col, "rows": int(len(df)), "requires_extra": "rasterio"},
                )
            )
        return records
