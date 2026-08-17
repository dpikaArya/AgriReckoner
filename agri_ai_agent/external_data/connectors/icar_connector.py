import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.download_strategy import try_priority_downloads


class ICARConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "ICAR"

    @property
    def base_url(self) -> str:
        return "https://krishikosh.egranth.ac.in/api"

    def connect(self) -> bool:
        try:
            resp = requests.head("https://krishikosh.egranth.ac.in", timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        search_term = query or "agriculture"
        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params={"q": search_term, "limit": 20},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", data.get("records", [])):
                results.append(
                    {
                        "id": item.get("id", ""),
                        "name": item.get("title", ""),
                        "description": (item.get("abstract") or item.get("description", ""))[:200],
                        "authors": item.get("authors", []),
                        "year": item.get("year", ""),
                        "type": item.get("type", "thesis"),
                    }
                )
            return results
        except requests.RequestException:
            return []

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)

        result = try_priority_downloads(
            [
                (
                    f"{self.base_url}/records/{resource_id}/metadata/csv",
                    "csv",
                    target_dir / f"icar_{resource_id}.csv",
                ),
                (
                    f"https://krishikosh.egranth.ac.in/bitstream/{resource_id}/1/fulltext.pdf",
                    "pdf",
                    target_dir / f"icar_{resource_id}.pdf",
                ),
            ]
        )
        return result

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
