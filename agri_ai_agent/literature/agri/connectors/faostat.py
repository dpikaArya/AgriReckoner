"""FAOSTAT connector using the official bulk-download service.

The interactive FAOSTAT API (``fenixservices.fao.org``) is bot-protected
(HTTP 521 to non-browser clients), so this connector uses the official
bulk-download host ``https://bulks-faostat.fao.org/production/`` (verified
reachable) and the published normalized CSV files.
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

_BULK_HOST = "https://bulks-faostat.fao.org/production"
_CATALOGUE_URL = f"{_BULK_HOST}/datasets_E.json"

# FAOSTAT domain code -> official normalized bulk filename.
# Phase 18 recovery: catalogue codes/filenames refreshed from
# https://bulks-faostat.fao.org/production/datasets_E.json (verified 2026-08-11).
_KNOWN_DATASETS: dict[str, tuple[str, str]] = {
    "QCL": ("Production - Crops, Livestock and Livestock Products", "Production_Crops_Livestock_E_All_Data_(Normalized).zip"),
    "RFN": ("Fertilizers - Consumption by Nutrient", "Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip"),
    "RP": ("Pesticides - Use", "Inputs_Pesticides_Use_E_All_Data_(Normalized).zip"),
    "RL": ("Land Use Indicators", "Inputs_LandUse_E_All_Data_(Normalized).zip"),
}

# FAOSTAT element names we can map to UAMS variables.
_ELEMENT_TO_VARIABLE = {
    "Yield": "yield_per_hectare",
    "Area harvested": "area_harvested",
    "Production": "production_tonnes",
    "Fertilizers - Nitrogen Consumption": "nitrogen_consumption",
    "Fertilizers - Phosphate Consumption": "phosphorus_consumption",
    "Fertilizers - Potash Consumption": "potassium_consumption",
}


class FaostatConnector(AgriculturalDataConnector):
    source_name = "FAOSTAT"
    display_name = "FAOSTAT (FAO)"
    base_url = _BULK_HOST
    default_rate_per_minute = 20

    auth = ConnectorAuth(env_vars=(), required=False)

    def _catalogue(self) -> list[dict[str, Any]]:
        payload = self.http.get_json("/datasets_E.json", use_cache=True)
        if not isinstance(payload, dict):
            return []
        datasets = payload.get("Datasets", {}).get("Dataset", []) or []
        return list(datasets)

    def _descriptors(self) -> list[DatasetDescriptor]:
        out: list[DatasetDescriptor] = []
        catalogue = {str(d.get("DatasetCode")): d for d in self._catalogue()}
        for code, (title, filename) in _KNOWN_DATASETS.items():
            meta = catalogue.get(code, {})
            url = f"{_BULK_HOST}/{filename}"
            out.append(
                DatasetDescriptor(
                    dataset_id=code,
                    title=title,
                    url=url,
                    description=meta.get("Description", ""),
                    license="FAOSTAT terms of use",
                    variable=None,
                    files=[{"url": url, "name": filename, "format": "zip"}],
                    metadata=meta,
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        tokens = query.lower().split()
        hits = [
            d
            for d in self._descriptors()
            if any(t in d.title.lower() or t in (d.description or "").lower() for t in tokens)
        ]
        return hits[:max_results] or self._descriptors()[:max_results]

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
        zip_path = target_dir / file_info["name"]
        if not zip_path.exists():
            data = self.http.get_bytes(file_info["url"])
            if data is None:
                return None
            zip_path.write_bytes(data)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = next(
                    n for n in zf.namelist()
                    if n.lower().endswith((".csv", ".txt")) and not n.startswith("__MACOSX")
                )
                out = target_dir / f"{dataset_id}.csv"
                with zf.open(csv_name) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                return out
        except (zipfile.BadZipFile, StopIteration):
            return None

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        try:
            df = pd.read_csv(download_path, encoding="utf-8-sig", low_memory=False)
        except Exception:  # noqa: BLE001
            return []
        records: list[AgriculturalRecord] = []
        need = {"Element", "Item", "Year", "Value", "Unit"}
        if not need.issubset(set(df.columns)):
            return records
        descriptor = self.fetch_metadata(dataset_id)
        provenance = descriptor.url if descriptor else _BULK_HOST
        for _, row in df.head(200000).iterrows():
            element = str(row.get("Element", ""))
            variable = _ELEMENT_TO_VARIABLE.get(element)
            if variable is None:
                continue
            value = _to_float(row.get("Value"))
            if value is None:
                continue
            unit = str(row.get("Unit", "")).strip() or None
            year = _to_int(row.get("Year"))
            if variable == "yield_per_hectare" and unit:
                # FAOSTAT yield unit is hg/ha -> kg/ha (1 hg = 0.1 kg).
                if unit.lower() in ("hg/ha", "100 g/ha"):
                    value = value * 0.1
                    unit = "kg/ha"
            records.append(
                AgriculturalRecord(
                    source=self.source_name,
                    dataset_id=f"{dataset_id}::{element}",
                    variable=variable,
                    value=value,
                    unit="kg/ha" if variable == "yield_per_hectare" else unit,
                    year=year,
                    country=str(row.get("Area", "")) or None,
                    crop=str(row.get("Item", "")) or None,
                    provenance=provenance,
                    license="FAOSTAT terms of use",
                    extra={
                        "area_code": row.get("Area Code"),
                        "item_code": row.get("Item Code"),
                        "element_code": row.get("Element Code"),
                        "flag": row.get("Flag"),
                    },
                )
            )
        return records[:50000]


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
