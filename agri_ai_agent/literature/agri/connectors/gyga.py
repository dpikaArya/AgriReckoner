"""GYGA (Global Yield Gap Atlas) connector.

The official yield-gap dataset is published by the GYGA project on Zenodo
(community ``global-yield-gap-atlas``); records are also downloadable from
``www.yieldgap.org``.  This connector uses the official Zenodo community API.
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


class GygaConnector(AgriculturalDataConnector):
    source_name = "GYGA"
    display_name = "Global Yield Gap Atlas"
    base_url = "https://zenodo.org/api/records"
    default_rate_per_minute = 30

    auth = ConnectorAuth(env_vars=(), required=False)

    def _community_records(self, page: int = 1, size: int = 10) -> list[dict[str, Any]]:
        payload = self.http.get_json(
            "",
            params={"q": "communities:global-yield-gap-atlas", "size": size, "page": page},
        )
        if not isinstance(payload, dict):
            return []
        return list(payload.get("hits", {}).get("hits", []) or [])

    def _descriptor(self, record: dict[str, Any]) -> DatasetDescriptor | None:
        metadata = record.get("metadata", {}) or {}
        rec_id = record.get("id")
        if not rec_id:
            return None
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
            title=metadata.get("title") or f"GYGA record {rec_id}",
            url=record.get("links", {}).get("html") or f"https://zenodo.org/records/{rec_id}",
            description=(metadata.get("description") or "")[:500],
            license=metadata.get("license", {}).get("id"),
            variable="yield_gap",
            files=files,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        out: list[DatasetDescriptor] = []
        for record in self._community_records(size=max_results):
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
        # prefer the first tabular file in the deposit
        chosen = None
        for f in descriptor.files:
            if (f.get("name") or "").lower().endswith((".csv", ".xlsx", ".xls", ".zip")):
                chosen = f
                break
        if chosen is None and descriptor.files:
            chosen = descriptor.files[0]
        if chosen is None or not chosen.get("url"):
            return None
        path = target_dir / f"gyga_{dataset_id}"
        data = self.http.get_bytes(chosen["url"])
        if data is None:
            return None
        path.write_bytes(data)
        return path

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        df = _load_tabular(download_path)
        if df is None:
            return []
        records: list[AgriculturalRecord] = []
        yield_col = next(
            (c for c in df.columns if "yield" in str(c).lower() and "gap" in str(c).lower()),
            None,
        )
        country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
        crop_col = next((c for c in df.columns if "crop" in str(c).lower()), None)
        if yield_col is None:
            return records
        for _, row in df.head(20000).iterrows():
            value = _to_float(row.get(yield_col))
            if value is None:
                continue
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=dataset_id,
                    variable="yield_gap",
                    value=value,
                    unit="t/ha",
                    crop=str(row.get(crop_col)) if crop_col else None,
                    country=str(row.get(country_col)) if country_col else None,
                    provenance=download_path.as_posix(),
                    license="GYGA data license",
                    extra={"row": dict(row.dropna().to_dict())},
                )
            )
        return records


def _load_tabular(path: Path) -> pd.DataFrame | None:
    suffix = (path.name or "").lower().rsplit(".", 1)[-1]
    try:
        if suffix == "csv":
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if suffix in ("xlsx", "xls"):
            return pd.read_excel(path, engine="openpyxl")
        if suffix == "zip":
            with zipfile.ZipFile(path) as zf:
                inner = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(inner) as src:
                    return pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    except Exception:  # noqa: BLE001
        return None
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
