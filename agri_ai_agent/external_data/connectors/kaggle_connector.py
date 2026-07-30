import hashlib
import os
from pathlib import Path
from typing import Optional

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.download_strategy import FormatFilter


class KaggleConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "Kaggle"

    @property
    def base_url(self) -> str:
        return "https://www.kaggle.com/api/v1"

    def connect(self) -> bool:
        username = os.getenv("KAGGLE_USERNAME", "")
        key = os.getenv("KAGGLE_KEY", "")
        return bool(username and key)

    def discover(self, query: Optional[str] = None) -> list[dict]:
        import subprocess
        search_term = query or "agriculture"
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list", "--search", search_term, "--csv"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return []
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return []
            header = lines[0].split(",")
            datasets = []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 2:
                    datasets.append({
                        "id": parts[0].strip('"'),
                        "name": parts[0].strip('"').split("/")[-1] if "/" in parts[0] else parts[0],
                        "description": parts[1].strip('"') if len(parts) > 1 else "",
                        "size": parts[3].strip('"') if len(parts) > 3 else "",
                    })
            return datasets
        except Exception as e:
            self.log.debug("Kaggle list_datasets failed: %s", e)
            return []

    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        import subprocess
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", resource_id, "-p", str(target_dir), "--unzip"],
                capture_output=True, text=True, timeout=300,
            )
            all_files = []
            for ext in ["parquet", "arrow", "csv", "json", "xml", "html", "pdf"]:
                all_files.extend(target_dir.glob(f"*.{ext}"))
            all_files.extend(target_dir.glob("*.zip"))

            file_infos = []
            for fp in all_files:
                fmt = fp.suffix.lower().lstrip(".")
                file_infos.append({"path": fp, "format": fmt})
            if not file_infos:
                return None

            ordered = FormatFilter.sorted_files(file_infos)
            return ordered[0]["path"] if ordered else None
        except Exception as e:
            self.log.debug("Kaggle download failed for %s: %s", resource_id, e)
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
        """No resources to release for Kaggle API connector."""
