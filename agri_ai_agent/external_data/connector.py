from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.download_strategy import FormatFilter
from agri_ai_agent.external_data.registry_db import DatasetRegistry


class ExternalDataConnector(ABC):
    def __init__(self):
        self._registry: Optional[DatasetRegistry] = None
        self._format_filter = FormatFilter()

    def set_registry(self, registry: DatasetRegistry):
        self._registry = registry

    @property
    def registry(self) -> Optional[DatasetRegistry]:
        return self._registry

    @property
    def format_filter(self) -> FormatFilter:
        return self._format_filter

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @property
    @abstractmethod
    def base_url(self) -> str:
        ...

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def discover(self, query: Optional[str] = None) -> list[dict]:
        ...

    @abstractmethod
    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        ...

    @abstractmethod
    def validate(self, package: DatasetPackage) -> bool:
        ...

    def register(self, package: DatasetPackage) -> str:
        if self._registry is None:
            return ""
        dataset_id, _ = self._registry.register(
            package,
            version=package.version,
            license=package.license,
            source_url=package.source_url or self.base_url,
        )
        return dataset_id

    @abstractmethod
    def update(self) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def fetch(self, resource_id: str, target_dir: Path) -> Optional[DatasetPackage]:
        if self._registry:
            existing = self._registry.find(self.source_name, resource_id)
            if existing:
                version_check = self._registry.check_update(
                    self.source_name, resource_id, existing["version"]
                )
                if version_check is None:
                    return None

        self.connect()
        try:
            path = self.download(resource_id, target_dir)
            if path is None:
                return None
            package = DatasetPackage(
                source=self.source_name,
                resource_id=resource_id,
                name=path.stem,
                download_path=path,
            )
            package.data = package.to_dataframe()
            package.is_valid = self.validate(package)
            if package.is_valid:
                self.register(package)
            return package
        finally:
            self.close()
