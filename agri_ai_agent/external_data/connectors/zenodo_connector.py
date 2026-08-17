import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class ZenodoConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "Zenodo"

    @property
    def base_url(self) -> str:
        return "https://zenodo.org/api"

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/records", params={"size": 1}, timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        params = {"q": query or "agriculture", "size": 20, "sort": "mostrecent"}
        try:
            resp = requests.get(f"{self.base_url}/records", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "id": str(rec.get("id")),
                    "name": rec.get("metadata", {}).get("title", ""),
                    "description": rec.get("metadata", {}).get("description", "")[:200],
                    "doi": rec.get("doi", ""),
                    "created_at": rec.get("created"),
                    "files": [f.get("links", {}).get("download", "") for f in rec.get("files", [])],
                }
                for rec in data.get("hits", {}).get("hits", [])
            ]
        except requests.RequestException:
            return []

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(f"{self.base_url}/records/{resource_id}", timeout=30)
            resp.raise_for_status()
            record = resp.json()
            files = record.get("files", [])
            if not files:
                return None
            download_url = files[0].get("links", {}).get("download", "")
            if not download_url:
                return None
            filename = files[0].get("key", f"zenodo_{resource_id}.csv")
            local_path = target_dir / f"zenodo_{resource_id}_{filename}"
            file_resp = requests.get(download_url, timeout=300)
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
