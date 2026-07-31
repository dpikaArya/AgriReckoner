import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class FAOSTATConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "FAOSTAT"

    @property
    def base_url(self) -> str:
        return "https://fenixservices.fao.org/faostat/api/v1"

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/QA/QA", timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        try:
            resp = requests.get(f"{self.base_url}/QA/QA", timeout=30)
            resp.raise_for_status()
            datasets = resp.json()
            results = []
            for ds in datasets:
                code = ds.get("DomainCode", "")
                name = ds.get("DomainName", "")
                if (
                    query
                    and query.lower() not in name.lower()
                    and query.lower() not in code.lower()
                ):
                    continue
                results.append(
                    {
                        "id": code,
                        "name": name,
                        "description": ds.get("Description", ""),
                        "updated_at": ds.get("UpdateDate"),
                    }
                )
            return results
        except requests.RequestException:
            return []

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_url = f"https://fenixservices.fao.org/faostat/static/bulkdownloads/{resource_id}.zip"
        local_path = target_dir / f"faostat_{resource_id}.zip"
        try:
            resp = requests.get(zip_url, timeout=300)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return local_path
        except requests.RequestException:
            csv_url = (
                f"https://fenixservices.fao.org/faostat/static/bulkdownloads/{resource_id}.csv"
            )
            local_path = local_path.with_suffix(".csv")
            try:
                resp = requests.get(csv_url, timeout=120)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
                return local_path
            except requests.RequestException:
                return None

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
        checksum = hashlib.md5(str(package.download_path).encode()).hexdigest()[:16]
        package.checksum = checksum
        return checksum

    def update(self) -> int:
        return 0

    def close(self) -> None:
        pass
