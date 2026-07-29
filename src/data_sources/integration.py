from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.connector_manager import ConnectorManager
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.registry import ConnectorRegistry
from agri_ai_agent.external_data.registry_db import DatasetRegistry

from src.data_sources.config_loader import (
    get_pipeline_config,
    get_source_config,
    get_storage_config,
    is_source_enabled,
)
from src.data_sources.storage import ImmutableStorage
from src.data_sources.sync import SyncManager


class DataSourceIntegration:
    def __init__(
        self,
        registry: DatasetRegistry,
        download_dir: Path,
        storage_dir: Optional[Path] = None,
        max_workers: Optional[int] = None,
        retry_max_attempts: Optional[int] = None,
        retry_base_delay: Optional[float] = None,
    ):
        pipe_cfg = get_pipeline_config()
        self._download_dir = download_dir
        self._storage = ImmutableStorage(storage_dir or Path(
            get_storage_config().get("base_dir", "external_data/raw")
        ))
        self._sync = SyncManager(registry)
        self._manager = ConnectorManager(
            registry=registry,
            download_dir=download_dir,
            max_workers=max_workers or pipe_cfg.get("max_workers", 4),
            retry_max_attempts=retry_max_attempts or pipe_cfg.get("retry_max_attempts", 3),
            retry_base_delay=retry_base_delay or pipe_cfg.get("retry_base_delay_sec", 2.0),
        )

    def list_sources(self) -> list[str]:
        all_sources = self._manager.list_sources()
        return [s for s in all_sources if is_source_enabled(s)]

    def health_checks(self, sources: Optional[list[str]] = None
                      ) -> dict[str, object]:
        return self._manager.health_checks(sources or self.list_sources())

    def run_all(self, sources: Optional[list[str]] = None
                ) -> tuple[dict[str, list[DatasetPackage]], list[object]]:
        selected = sources if sources is not None else self.list_sources()
        packages_by_source, run_logs = self._manager.run_all(sources=selected)

        for src, pkgs in packages_by_source.items():
            for pkg in pkgs:
                self._immutable_store(pkg)

        return packages_by_source, run_logs

    def _immutable_store(self, pkg: DatasetPackage):
        if pkg.download_path and pkg.download_path.exists():
            checksum = self._sync.sha256(pkg.download_path)
            if self._sync.skip_if_unchanged(
                pkg.source or pkg.provider,
                pkg.resource_id,
                pkg.version,
                checksum,
            ):
                return
            stored = self._storage.store(
                source=pkg.source or pkg.provider,
                resource_id=pkg.resource_id,
                version=pkg.version,
                file_path=pkg.download_path,
                checksum=checksum,
            )
            pkg.checksum = checksum
            pkg.supplementary_files.append(stored)

    @staticmethod
    def summarize_runs(logs: list[object]) -> dict:
        return ConnectorManager.summarize_runs(logs)

    @staticmethod
    def merge_packages(packages_by_source: dict[str, list[DatasetPackage]],
                       existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return ConnectorManager.merge_packages(packages_by_source, existing_df)
