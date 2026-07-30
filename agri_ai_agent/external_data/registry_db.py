import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.external_data.dataset_package import DatasetPackage

logger = logging.getLogger(__name__)


class DatasetRegistry:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_registry (
                    dataset_id          TEXT PRIMARY KEY,
                    dataset_name        TEXT NOT NULL,
                    provider            TEXT NOT NULL,
                    source_url          TEXT,
                    version             TEXT NOT NULL,
                    license             TEXT,
                    download_date       TEXT NOT NULL,
                    last_checked        TEXT,
                    record_count        INTEGER DEFAULT 0,
                    schema_hash         TEXT,
                    storage_format      TEXT,
                    storage_location    TEXT NOT NULL,
                    status              TEXT NOT NULL DEFAULT 'active',
                    previous_version_id TEXT,
                    superseded_by       TEXT,
                    resource_id         TEXT NOT NULL,
                    checksum            TEXT,
                    metadata_json       TEXT,
                    created_at          TEXT NOT NULL,
                    updated_at          TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_provider
                    ON dataset_registry(provider)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_status
                    ON dataset_registry(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_lookup
                    ON dataset_registry(provider, resource_id, status)
            """)

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    @staticmethod
    def generate_dataset_id(provider: str, resource_id: str, version: str) -> str:
        raw = f"{provider}/{resource_id}/{version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def compute_schema_hash(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return hashlib.md5(b"empty").hexdigest()[:16]
        schema_str = json.dumps(
            {col: str(dtype) for col, dtype in df.dtypes.items()},
            sort_keys=True,
        )
        return hashlib.md5(schema_str.encode()).hexdigest()[:16]

    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        parts1 = [p for p in v1.replace("-", ".").split(".") if p.isdigit()]
        parts2 = [p for p in v2.replace("-", ".").split(".") if p.isdigit()]
        for a, b in zip(parts1, parts2):
            if int(a) < int(b):
                return -1
            if int(a) > int(b):
                return 1
        if len(parts1) < len(parts2):
            return -1
        if len(parts1) > len(parts2):
            return 1
        return 0

    def register(self, package: DatasetPackage, version: Optional[str] = None,
                 license: str = "", source_url: str = "") -> tuple[str, bool]:
        version = version or package.version or "1.0.0"
        existing = self.find(package.source, package.resource_id, version)
        if existing is not None:
            return existing["dataset_id"], False

        prev = self._latest(package.source, package.resource_id)
        dataset_id = self.generate_dataset_id(
            package.source, package.resource_id, version
        )
        df = package.to_dataframe()
        schema_hash = self.compute_schema_hash(df)
        storage_format = (
            package.download_path.suffix.lstrip(".")
            if package.download_path else "unknown"
        )
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO dataset_registry (
                    dataset_id, dataset_name, provider, source_url,
                    version, license, download_date, last_checked,
                    record_count, schema_hash, storage_format,
                    storage_location, status, previous_version_id,
                    resource_id, checksum, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    package.name,
                    package.source,
                    source_url or getattr(package, "source_url", ""),
                    version,
                    license,
                    now,
                    now,
                    package.row_count,
                    schema_hash,
                    storage_format,
                    str(package.download_path) if package.download_path else "",
                    "active",
                    prev["dataset_id"] if prev else None,
                    package.resource_id,
                    package.checksum or "",
                    json.dumps(package.metadata) if package.metadata else "{}",
                    now,
                    now,
                ),
            )
            if prev:
                conn.execute(
                    "UPDATE dataset_registry SET status='superseded', "
                    "superseded_by=?, updated_at=? WHERE dataset_id=?",
                    (dataset_id, now, prev["dataset_id"]),
                )
        return dataset_id, True

    def find(self, provider: str, resource_id: str,
             version: Optional[str] = None) -> Optional[dict]:
        with self._conn() as conn:
            if version:
                row = conn.execute(
                    "SELECT * FROM dataset_registry WHERE provider=? "
                    "AND resource_id=? AND version=? AND status='active'",
                    (provider, resource_id, version),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM dataset_registry WHERE provider=? "
                    "AND resource_id=? AND status='active'",
                    (provider, resource_id),
                ).fetchone()
            return self._row_to_dict(row, conn) if row else None

    def _latest(self, provider: str, resource_id: str) -> Optional[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM dataset_registry WHERE provider=? "
                "AND resource_id=? AND status='active' ORDER BY created_at DESC",
                (provider, resource_id),
            ).fetchall()
            if not rows:
                return None
            return self._row_to_dict(rows[0], conn)

    def get(self, dataset_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM dataset_registry WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            return self._row_to_dict(row, conn) if row else None

    def list_datasets(self, provider: Optional[str] = None,
                      status: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            query = "SELECT * FROM dataset_registry WHERE 1=1"
            params = []
            if provider:
                query += " AND provider=?"
                params.append(provider)
            if status:
                query += " AND status=?"
                params.append(status)
            query += " ORDER BY provider, download_date DESC"
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(r, conn) for r in rows]

    def check_update(self, provider: str, resource_id: str,
                     current_version: str) -> Optional[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version FROM dataset_registry WHERE provider=? "
                "AND resource_id=? AND status='active'",
                (provider, resource_id),
            ).fetchall()
            for (version,) in rows:
                if self.compare_versions(version, current_version) > 0:
                    return version
            return None

    def mark_superseded(self, dataset_id: str, new_version_id: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE dataset_registry SET status='superseded', "
                "superseded_by=?, updated_at=? WHERE dataset_id=?",
                (new_version_id, now, dataset_id),
            )

    def rollback(self, dataset_id: str) -> Optional[dict]:
        current = self.get(dataset_id)
        if current is None:
            return None
        prev_id = current.get("previous_version_id")
        if not prev_id:
            return None
        with self._conn() as conn:
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE dataset_registry SET status='rolled_back', "
                "superseded_by=NULL, updated_at=? WHERE dataset_id=?",
                (now, dataset_id),
            )
            conn.execute(
                "UPDATE dataset_registry SET status='active', "
                "superseded_by=NULL, updated_at=? WHERE dataset_id=?",
                (now, prev_id),
            )
        return self.get(prev_id)

    def update_timestamp(self, dataset_id: str):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE dataset_registry SET last_checked=?, updated_at=? "
                "WHERE dataset_id=?",
                (now, now, dataset_id),
            )

    def summary(self) -> dict:
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM dataset_registry"
            ).fetchone()[0]
            by_status = conn.execute(
                "SELECT status, COUNT(*) FROM dataset_registry "
                "GROUP BY status"
            ).fetchall()
            by_provider = conn.execute(
                "SELECT provider, COUNT(*) FROM dataset_registry "
                "GROUP BY provider"
            ).fetchall()
            return {
                "total_datasets": total,
                "by_status": dict(by_status),
                "by_provider": dict(by_provider),
            }

    def close(self):
        """Close registry — connections are per-operation via context manager."""
        logger.debug("DatasetRegistry closed")

    def _row_to_dict(self, row, conn) -> dict:
        if row is None:
            return None
        if not hasattr(self, "_columns"):
            cur = conn.execute("SELECT * FROM dataset_registry LIMIT 0")
            self._columns = [desc[0] for desc in cur.description]
        return dict(zip(self._columns, row))
