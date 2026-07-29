from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from agri_ai_agent.external_data.dataset_package import DatasetPackage


class ExternalDataConnector(ABC):
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

    @abstractmethod
    def register(self, package: DatasetPackage) -> str:
        ...

    @abstractmethod
    def update(self) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def fetch(self, resource_id: str, target_dir: Path) -> Optional[DatasetPackage]:
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
