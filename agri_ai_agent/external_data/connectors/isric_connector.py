import hashlib
from pathlib import Path
from typing import Optional

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class ISRICConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "ISRIC"

    @property
    def base_url(self) -> str:
        return "https://files.isric.org"

    def connect(self) -> bool:
        try:
            resp = requests.head(self.base_url, timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: Optional[str] = None) -> list[dict]:
        collections = [
            {
                "id": "soilgrids",
                "name": "SoilGrids 250m",
                "description": "Global soil property maps at 250m resolution",
            },
            {
                "id": "wosis",
                "name": "WoSIS Soil Profile Database",
                "description": "World Soil Information Service standardized soil profile data",
            },
            {
                "id": "isric-soil-data-files",
                "name": "ISRIC Soil Data Files",
                "description": "Collection of legacy soil data compilations",
            },
        ]
        if query:
            q = query.lower()
            collections = [c for c in collections if q in c["name"].lower() or q in c["id"]]
        return collections

    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/public/{resource_id}.zip"
        local_path = target_dir / f"isric_{resource_id}.zip"
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return local_path
        except requests.RequestException:
            return None

    def validate(self, package: DatasetPackage) -> bool:
        package.validation_errors.clear()
        if package.download_path is None:
            package.validation_errors.append("No download path")
            return False
        if not package.download_path.exists() or package.download_path.stat().st_size == 0:
            package.validation_errors.append("File missing or empty")
            return False
        package.is_valid = True
        return True

    def register(self, package: DatasetPackage) -> str:
        checksum = hashlib.sha256(str(package.download_path).encode()).hexdigest()[:16]
        package.checksum = checksum
        return checksum

    def update(self) -> int:
        return 0

    def close(self) -> None:
        pass
