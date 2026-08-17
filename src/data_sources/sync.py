import hashlib
from pathlib import Path

import requests

from agri_ai_agent.external_data.registry_db import DatasetRegistry


class SyncManager:
    def __init__(
        self, registry: DatasetRegistry, chunk_size: int = 8192, partial_suffix: str = ".partial"
    ):
        self._registry = registry
        self._chunk_size = chunk_size
        self._partial_suffix = partial_suffix

    def download_resumable(self, url: str, target_path: Path, timeout: int = 300) -> Path | None:
        partial_path = target_path.with_suffix(target_path.suffix + self._partial_suffix)
        headers = {}
        existing_size = 0
        if partial_path.exists():
            existing_size = partial_path.stat().st_size
            headers["Range"] = f"bytes={existing_size}-"

        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            if resp.status_code == 416:
                partial_path.rename(target_path)
                return target_path
            if resp.status_code == 206:
                mode = "ab"
            elif resp.status_code == 200 and existing_size == 0:
                mode = "wb"
            else:
                return None

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(partial_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=self._chunk_size):
                    if chunk:
                        f.write(chunk)
            partial_path.rename(target_path)
            return target_path
        except (requests.RequestException, OSError):
            return None

    def skip_if_unchanged(self, source: str, resource_id: str, version: str, checksum: str) -> bool:
        record = self._registry.find(source, resource_id, version)
        if record and record.get("checksum") == checksum:
            self._registry.update_timestamp(record["dataset_id"])
            return True
        return False

    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def cache_path(self, base_cache: Path, source: str, resource_id: str, version: str) -> Path:
        cache_dir = base_cache / source / resource_id / f"v{version}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def cache_hit(
        self, base_cache: Path, source: str, resource_id: str, version: str, checksum: str
    ) -> Path | None:
        cache_dir = self.cache_path(base_cache, source, resource_id, version)
        for fp in cache_dir.iterdir():
            if fp.stem == checksum and fp.suffix not in (".metadata.json", ".partial"):
                return fp
        return None
