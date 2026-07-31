from pathlib import Path

import pandas as pd

from agri_ai_agent.external_data.connector_manager import (
    ConnectorHealth,
    ConnectorManager,
    ConnectorRunLog,
)
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.registry_db import DatasetRegistry
from src.data_sources.config_loader import (
    get_pipeline_config,
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
        storage_dir: Path | None = None,
        max_workers: int | None = None,
        retry_max_attempts: int | None = None,
        retry_base_delay: float | None = None,
    ):
        pipe_cfg = get_pipeline_config()
        self._download_dir = download_dir
        self._storage = ImmutableStorage(
            storage_dir or Path(get_storage_config().get("base_dir", "external_data/raw"))
        )
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

    def health_checks(self, sources: list[str] | None = None) -> dict[str, ConnectorHealth]:
        return self._manager.health_checks(sources or self.list_sources())

    def run_all(
        self, sources: list[str] | None = None
    ) -> tuple[dict[str, list[DatasetPackage]], list[ConnectorRunLog]]:
        selected = sources if sources is not None else self.list_sources()
        packages_by_source, run_logs = self._manager.run_all(sources=selected)

        for _, pkgs in packages_by_source.items():
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
    def summarize_runs(logs: list[ConnectorRunLog]) -> dict:
        return ConnectorManager.summarize_runs(logs)

    @staticmethod
    def merge_packages(
        packages_by_source: dict[str, list[DatasetPackage]],
        existing_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return ConnectorManager.merge_packages(packages_by_source, existing_df)
