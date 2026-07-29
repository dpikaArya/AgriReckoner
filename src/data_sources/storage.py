import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.data_sources.config_loader import get_storage_config


class ImmutableStorage:
    def __init__(self, base_dir: Optional[Path] = None):
        cfg = get_storage_config()
        self._base = base_dir or Path(cfg.get("base_dir", "external_data/raw"))
        self._versioned = cfg.get("versioned", True)

    def store(self, source: str, resource_id: str, version: str,
              file_path: Path, checksum: Optional[str] = None) -> Path:
        if checksum is None:
            checksum = self._sha256(file_path)
        ext = file_path.suffix.lower()

        if self._versioned:
            dest_dir = self._base / source / resource_id / f"v{version}"
        else:
            dest_dir = self._base / source / resource_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"{checksum}{ext}"
        if dest_path.exists():
            return dest_path

        shutil.copy2(file_path, dest_path)

        meta = {
            "original_name": file_path.name,
            "source": source,
            "resource_id": resource_id,
            "version": version,
            "checksum": checksum,
            "stored_at": datetime.now().isoformat(),
            "size_bytes": file_path.stat().st_size,
        }
        meta_path = dest_dir / f"{checksum}.metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return dest_path

    def resolve(self, source: str, resource_id: str,
                version: str, checksum: str) -> Optional[Path]:
        if self._versioned:
            search_dir = self._base / source / resource_id / f"v{version}"
        else:
            search_dir = self._base / source / resource_id
        if not search_dir.exists():
            return None
        for fp in search_dir.iterdir():
            if fp.stem == checksum and fp.suffix != ".metadata.json":
                return fp
        return None

    def version_path(self, source: str, resource_id: str,
                     version: str) -> Optional[Path]:
        if self._versioned:
            p = self._base / source / resource_id / f"v{version}"
        else:
            p = self._base / source / resource_id
        return p if p.exists() else None

    def list_versions(self, source: str, resource_id: str) -> list[str]:
        source_dir = self._base / source / resource_id
        if not source_dir.exists():
            return []
        versions = []
        for entry in sorted(source_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("v"):
                versions.append(entry.name[1:])
        return versions

    def dedup_check(self, source: str, resource_id: str,
                    version: str, file_path: Path) -> Optional[str]:
        checksum = self._sha256(file_path)
        existing = self.resolve(source, resource_id, version, checksum)
        return checksum if existing is None else None

    @staticmethod
    def _sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    @staticmethod
    def file_checksum(file_path: Path) -> str:
        return ImmutableStorage._sha256(file_path)
