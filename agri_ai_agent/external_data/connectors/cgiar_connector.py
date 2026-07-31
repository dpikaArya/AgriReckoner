import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.download_strategy import (
    download_huggingface_parquet,
    try_priority_downloads,
)


class CGIARConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "CGIAR"

    @property
    def base_url(self) -> str:
        return "https://huggingface.co/api/datasets/CGIAR"

    def connect(self) -> bool:
        try:
            resp = requests.head(self.base_url, timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        url = "https://huggingface.co/api/datasets?search=CGIAR"
        if query:
            url += f"+{query}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return [
                {
                    "id": ds.get("id"),
                    "name": ds.get("id", "").split("/")[-1],
                    "description": ds.get("description", ""),
                    "downloads": ds.get("downloads", 0),
                    "updated_at": ds.get("lastModified"),
                }
                for ds in resp.json()
            ]
        except requests.RequestException:
            return []

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)

        path = download_huggingface_parquet(resource_id, target_dir)
        if path is not None:
            return path

        safe_name = resource_id.replace("/", "_")
        return try_priority_downloads(
            [
                (
                    f"https://huggingface.co/datasets/{resource_id}/resolve/main/data/train-00000-of-00001.parquet",
                    "parquet",
                    target_dir / f"cgiar_{safe_name}.parquet",
                ),
                (
                    f"https://huggingface.co/datasets/{resource_id}/resolve/main/data.csv",
                    "csv",
                    target_dir / f"cgiar_{safe_name}.csv",
                ),
            ]
        )

    def validate(self, package: DatasetPackage) -> bool:
        package.validation_errors.clear()
        if package.data is None and package.download_path is None:
            package.validation_errors.append("No data or download path")
            return False
        df = package.data if package.data is not None else package.to_dataframe()
        if df.empty:
            package.validation_errors.append("Empty dataset")
            return False
        package.is_valid = True
        return True

    def register(self, package: DatasetPackage) -> str:
        checksum = hashlib.sha256(str(package.download_path).encode()).hexdigest()[:16]
        package.checksum = checksum
        return checksum

    def update(self) -> int:
        new_items = 0
        datasets = self.discover()
        for ds in datasets:
            cache_key = f"cgiar_{ds['name']}"
            if not Path(f".cache/{cache_key}").exists():
                new_items += 1
        return new_items

    def close(self) -> None:
        pass
