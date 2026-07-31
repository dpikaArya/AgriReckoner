import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class MendeleyConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "Mendeley_Data"

    @property
    def base_url(self) -> str:
        return "https://api.data.mendeley.com/v1"

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/catalogs", timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        params = {"q": query or "agriculture", "limit": 20}
        try:
            resp = requests.get(
                f"{self.base_url}/catalogs",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "id": item.get("id", ""),
                    "name": item.get("title", ""),
                    "description": (item.get("description") or "")[:200],
                    "doi": item.get("doi", ""),
                    "authors": item.get("authors", []),
                    "version": item.get("version", ""),
                }
                for item in data.get("results", data.get("data", []))
            ]
        except requests.RequestException:
            return []

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / f"mendeley_{resource_id}.csv"
        try:
            resp = requests.get(
                f"{self.base_url}/catalogs/{resource_id}/files",
                timeout=60,
            )
            resp.raise_for_status()
            files_data = resp.json()
            files = files_data.get("results", files_data.get("data", []))
            if not files:
                return None
            file_url = files[0].get("download_url") or files[0].get("url", "")
            if not file_url:
                return None
            file_resp = requests.get(file_url, timeout=300)
            file_resp.raise_for_status()
            local_path.write_bytes(file_resp.content)
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
