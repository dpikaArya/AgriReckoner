import hashlib
import os
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.download_strategy import (
    try_priority_downloads,
)
from agri_ai_agent.utils.logging_utils import get_logger

logger = get_logger("HuggingFaceConnector")

DEFAULT_SEARCH_TERMS = [
    "agriculture",
    "crop",
    "soil",
    "climate",
    "yield",
    "weather",
    "farm",
    "plant",
    "vegetation",
    "land+cover",
    "remote+sensing+agriculture",
    "crop+type",
    "soil+moisture",
    "precipitation",
    "temperature+agriculture",
    "agronomy",
    "fertilizer",
    "irrigation",
    "pest",
    "phenology",
]


class HuggingFaceConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "HuggingFace"

    @property
    def base_url(self) -> str:
        return "https://huggingface.co/api/datasets"

    def connect(self) -> bool:
        try:
            resp = requests.head(self.base_url, timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        if query:
            search_terms = [query]
        else:
            search_terms = os.environ.get(
                "AGRI_HUGGINGFACE_SEARCH_TERMS",
                ",".join(DEFAULT_SEARCH_TERMS[:5]),
            ).split(",")

        seen_ids: set[str] = set()
        results: list[dict] = []

        for term in search_terms:
            term = term.strip()
            if not term:
                continue
            try:
                url = f"{self.base_url}?search={term}&sort=downloads&direction=-1&limit=20"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                for ds in resp.json():
                    ds_id = ds.get("id")
                    if ds_id and ds_id not in seen_ids:
                        seen_ids.add(ds_id)
                        results.append(
                            {
                                "id": ds_id,
                                "name": ds_id.split("/")[-1] if "/" in ds_id else ds_id,
                                "description": ds.get("description", ""),
                                "downloads": ds.get("downloads", 0),
                                "updated_at": ds.get("lastModified"),
                                "tags": ds.get("tags", []),
                            }
                        )
            except requests.RequestException:
                continue

        results.sort(key=lambda x: x.get("downloads", 0), reverse=True)
        return results

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = resource_id.replace("/", "_").replace("-", "_")

        # First check dataset size via HF API to avoid downloading large datasets
        max_bytes = int(os.environ.get("AGRI_HF_MAX_DOWNLOAD_BYTES", "50000000"))  # 50MB default
        try:
            info_url = f"https://huggingface.co/api/datasets/{resource_id}"
            resp = requests.get(info_url, timeout=15)
            if resp.status_code == 200:
                info = resp.json()
                siblings = info.get("siblings", [])
                total_bytes = sum(s.get("size", 0) or 0 for s in siblings)
                if total_bytes > max_bytes:
                    return None
        except requests.RequestException:
            logger.debug("HuggingFace API info request failed for %s", resource_id)

        # Try direct parquet download first (fastest)
        safe_name = resource_id.replace("/", "_").replace("-", "_")
        result = try_priority_downloads(
            [
                (
                    f"https://huggingface.co/datasets/{resource_id}/resolve/main/data/train-00000-of-00001.parquet",
                    "parquet",
                    target_dir / f"hf_{safe_name}.parquet",
                ),
                (
                    f"https://huggingface.co/datasets/{resource_id}/resolve/main/data.csv",
                    "csv",
                    target_dir / f"hf_{safe_name}.csv",
                ),
                (
                    f"https://huggingface.co/datasets/{resource_id}/resolve/main/train.csv",
                    "csv",
                    target_dir / f"hf_{safe_name}.csv",
                ),
            ]
        )
        if result is not None:
            return result

        # Fallback: use huggingface datasets library with a row limit
        try:
            from datasets import load_dataset

            ds = load_dataset(
                resource_id,
                split="train",
                streaming=True,
            )
            rows = []
            for i, row in enumerate(ds):
                if i >= 1000:
                    break
                rows.append(row)
            if rows:
                import pandas as pd

                df = pd.DataFrame(rows)
                parquet_path = target_dir / f"hf_{safe_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                return parquet_path
        except Exception:
            logger.warning(
                "HuggingFace streaming download failed for %s", resource_id, exc_info=True
            )
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
        checksum = hashlib.sha256(str(package.download_path).encode()).hexdigest()[:16]
        package.checksum = checksum
        return checksum

    def update(self) -> int:
        new_items = 0
        datasets = self.discover()
        for ds in datasets:
            cache_key = f"hf_{ds['name']}"
            if not Path(f".cache/{cache_key}").exists():
                new_items += 1
        return new_items

    def close(self) -> None:
        pass
