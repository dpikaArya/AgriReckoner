import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path


VERSIONS_DDL = """
CREATE TABLE IF NOT EXISTS resource_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_type   TEXT    NOT NULL,
    resource_id     TEXT    NOT NULL,
    version         TEXT    NOT NULL DEFAULT '1.0.0',
    version_hash    TEXT    NOT NULL,
    checksum        TEXT    DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'active',
    metadata_json   TEXT    DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    UNIQUE(resource_type, resource_id, version)
);

CREATE TABLE IF NOT EXISTS sync_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    repos_checked   INTEGER DEFAULT 0,
    repos_changed   INTEGER DEFAULT 0,
    stages_executed TEXT    DEFAULT '[]',
    status          TEXT    DEFAULT 'running',
    summary_json    TEXT    DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_rv_type_id ON resource_versions(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_rv_status ON resource_versions(status);
"""


class VersionHistory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript(VERSIONS_DDL)
        conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def register_version(self, resource_type: str, resource_id: str,
                         version: str = "", checksum: str = "",
                         metadata: Optional[dict] = None) -> dict:
        conn = self._connect()
        now = datetime.now().isoformat()
        content = f"{resource_type}:{resource_id}:{version}:{checksum}"
        version_hash = self.compute_hash(content)
        if not version:
            latest = self.get_latest(resource_type, resource_id)
            if latest:
                parts = latest["version"].split(".")
                version = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else "2.0.0"
            else:
                version = "1.0.0"

        try:
            conn.execute(
                "INSERT INTO resource_versions "
                "(resource_type, resource_id, version, version_hash, checksum, metadata_json, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (resource_type, resource_id, version, version_hash, checksum,
                 json.dumps(metadata or {}), now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.execute(
                "UPDATE resource_versions SET version_hash=?, checksum=?, "
                "metadata_json=?, created_at=? WHERE resource_type=? AND resource_id=? AND version=?",
                (version_hash, checksum, json.dumps(metadata or {}), now,
                 resource_type, resource_id, version),
            )
            conn.commit()

        return {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "version": version,
            "version_hash": version_hash,
            "checksum": checksum,
            "created_at": now,
        }

    def get_latest(self, resource_type: str, resource_id: str) -> dict:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM resource_versions WHERE resource_type=? AND resource_id=? AND status='active' "
            "ORDER BY rowid DESC LIMIT 1",
            (resource_type, resource_id),
        ).fetchone()
        if row is None:
            return {}
        return dict(row)

    def has_changed(self, resource_type: str, resource_id: str, checksum: str) -> bool:
        latest = self.get_latest(resource_type, resource_id)
        if not latest:
            return True
        return latest.get("checksum", "") != checksum

    def start_sync(self) -> int:
        conn = self._connect()
        now = datetime.now().isoformat()
        cur = conn.execute(
            "INSERT INTO sync_history (started_at, status) VALUES (?, 'running')",
            (now,),
        )
        conn.commit()
        return cur.lastrowid

    def complete_sync(self, sync_id: int, repos_checked: int, repos_changed: int,
                      stages_executed: list[str], summary: dict):
        conn = self._connect()
        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE sync_history SET completed_at=?, repos_checked=?, repos_changed=?, "
            "stages_executed=?, status=?, summary_json=? WHERE id=?",
            (now, repos_checked, repos_changed, json.dumps(stages_executed),
             "completed", json.dumps(summary), sync_id),
        )
        conn.commit()

    def list_syncs(self, limit: int = 10) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM sync_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
