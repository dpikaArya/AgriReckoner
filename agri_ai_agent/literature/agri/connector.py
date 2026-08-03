"""Contract for agricultural data connectors.

Every connector reaches an *official* production data source, downloads the
metadata and (when available) the dataset itself, normalizes units, records
provenance, and maps variables into the Universal Agricultural Schema (UAMS).

Connectors that require credentials are disabled automatically when those
credentials are absent — the remainder of the pipeline continues.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth, LiteratureConfig
from agri_ai_agent.literature.http import ConnectorHttpClient
from agri_ai_agent.literature.models import AgriculturalRecord, SyncResult
from agri_ai_agent.literature.state import ConnectorStateStore

logger = logging.getLogger(__name__)


@dataclass
class DatasetDescriptor:
    """Metadata describing one downloadable dataset from an official source."""

    dataset_id: str
    title: str
    url: str  # landing page / provenance URL
    description: str = ""
    license: str | None = None
    variable: str | None = None  # UAMS canonical variable when single-variable
    files: list[dict[str, Any]] = field(default_factory=list)  # [{url, name, format}]
    metadata: dict[str, Any] = field(default_factory=dict)
    spatial: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "license": self.license,
            "variable": self.variable,
            "files": self.files,
            "metadata": self.metadata,
            "spatial": self.spatial,
            "temporal": self.temporal,
        }


# Common unit conversions toward SI / agronomic canon.
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("kg/ha", "kg/ha"): 1.0,
    ("t/ha", "kg/ha"): 1000.0,
    ("tonnes/ha", "kg/ha"): 1000.0,
    ("kg per ha", "kg/ha"): 1.0,
    ("g/m2", "kg/ha"): 10.0,
    ("g/m²", "kg/ha"): 10.0,
    ("q/ha", "kg/ha"): 100.0,
    ("quintal/ha", "kg/ha"): 100.0,
    ("ton/ha", "kg/ha"): 1000.0,
    ("mg/kg", "mg/kg"): 1.0,
    ("ppm", "mg/kg"): 1.0,
    ("%", "%"): 1.0,
    ("percent", "%"): 1.0,
    ("mm", "mm"): 1.0,
    ("°c", "degC"): 1.0,
    ("c", "degC"): 1.0,
}


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().lower()
    for (src, canon) in UNIT_CONVERSIONS:
        if u == src or u == src.lower():
            return canon
    return unit.strip()


def convert_unit(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """Convert a numeric value between recognized unit pairs."""
    if from_unit == to_unit:
        return value
    factor = UNIT_CONVERSIONS.get((from_unit, to_unit))
    if factor is None:
        return value  # unknown pair: leave untouched, provenance records original
    return value * factor


class AgriculturalDataConnector(ABC):
    source_name: str = ""
    display_name: str = ""
    base_url: str = ""
    default_rate_per_minute: int = 20
    auth: ConnectorAuth = ConnectorAuth()

    def __init__(
        self,
        config: LiteratureConfig,
        state: ConnectorStateStore,
        *,
        cache_dir: str | Path | None = None,
    ):
        self.config = config
        self.state = state
        self.logger = logging.getLogger(
            f"literature.agri.{self.source_name or self.__class__.__name__}"
        )
        source_name = self.source_name or self.__class__.__name__
        self.http = ConnectorHttpClient(
            base_url=self.base_url,
            api_key=None,
            headers=self._auth_headers(),
            rate_limit_per_minute=config.limit(
                source_name,
                "rate_limit_per_minute",
                config.rate_limit_per_minute or self.default_rate_per_minute,
            ),
            timeout_sec=config.limit(source_name, "timeout_sec", config.timeout_sec),
            max_retries=config.limit(source_name, "retry_max", config.retry_max),
            cache_dir=cache_dir or config.cache_dir,
        )
        self.enabled = self._detect_enabled()
        if not self.enabled:
            self.logger.warning(
                "Agricultural connector %s disabled: missing credentials (%s)",
                self.source_name or self.__class__.__name__,
                self.auth.describe(),
            )

    # ------------------------------------------------------------------ #
    # auth
    # ------------------------------------------------------------------ #
    def _auth_headers(self) -> dict[str, str]:
        if self.auth.header and self.auth.available:
            return {self.auth.header: self.auth.resolve() or ""}
        return {}

    def _detect_enabled(self) -> bool:
        override = self.config.source_enabled(self.source_name)
        if override is not None:
            return override
        if not self.auth.required:
            return True
        return self.auth.available

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        """Search the official source; return dataset descriptors."""

    @abstractmethod
    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        """Fetch full metadata for a dataset."""

    @abstractmethod
    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        """Download the dataset (or its metadata) to ``target_dir``."""

    @abstractmethod
    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        """Parse a downloaded dataset into normalized AgriculturalRecords."""

    def validate_record(self, record: AgriculturalRecord) -> list[str]:
        """Structural validation; empty list means valid."""
        errors: list[str] = []
        if not record.source:
            errors.append("missing source")
        if not record.dataset_id:
            errors.append("missing dataset_id")
        if record.value is None and not record.extra.get("rows", 0):
            errors.append("no data parsed")
        return errors

    # ------------------------------------------------------------------ #
    # incremental sync (default)
    # ------------------------------------------------------------------ #
    def incremental_sync(
        self,
        queries: tuple[str, ...] | None = None,
        max_datasets: int = 10,
    ) -> SyncResult:
        result = SyncResult(connector=self.source_name)
        if not self.enabled:
            result.errors.append("connector disabled: credentials missing")
            result.completed_at = result.started_at
            return result
        queries = tuple(queries or ("agriculture", "crop", "soil", "yield"))
        downloaded = 0
        try:
            for query in queries:
                if downloaded >= max_datasets:
                    break
                for descriptor in self.search(query, max_results=max_datasets):
                    if downloaded >= max_datasets:
                        break
                    key = f"downloaded::{descriptor.dataset_id}"
                    if self.state.get_cursor(self.source_name, key) and not self.config.incremental_mode:
                        continue
                    target = self.config.data_dir / "datasets" / self.source_name
                    target.mkdir(parents=True, exist_ok=True)
                    path = self.download(descriptor.dataset_id, target)
                    if path is None:
                        result.errors.append(f"download failed: {descriptor.dataset_id}")
                        continue
                    records = self.to_records(descriptor.dataset_id, path)
                    for rec in records:
                        self.state.cache_put(
                            self.source_name, f"{rec.dataset_id}::{rec.variable}", rec.to_dict()
                        )
                    result.fetched_records += len(records)
                    result.new_records += len(records)
                    downloaded += 1
                    self.state.set_cursor(self.source_name, key, True)
            self.state.set_cursor(self.source_name, "last_sync_at", result.started_at)
            result.cursor = result.started_at
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("incremental_sync failed for %s", self.source_name)
            result.errors.append(str(exc))
        result.completed_at = result.started_at
        return result

    def get_cursor(self, key: str, default: Any = None) -> Any:
        return self.state.get_cursor(self.source_name, key, default)

    def set_cursor(self, key: str, value: Any) -> None:
        self.state.set_cursor(self.source_name, key, value)

    def close(self) -> None:  # noqa: B027 - optional resource release, no-op base
        pass
