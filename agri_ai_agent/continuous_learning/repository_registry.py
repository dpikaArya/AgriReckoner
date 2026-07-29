import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


REPOSITORIES_DDL = """
CREATE TABLE IF NOT EXISTS repositories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    url             TEXT    NOT NULL,
    repo_type       TEXT    NOT NULL DEFAULT 'generic',
    connector_name  TEXT    NOT NULL DEFAULT '',
    enabled         INTEGER NOT NULL DEFAULT 1,
    sync_interval_sec INTEGER DEFAULT 3600,
    last_sync_at    TEXT,
    last_version    TEXT    DEFAULT '',
    metadata_json   TEXT    DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
"""


class RepositoryRegistry:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self):
        conn = self._connect()
        conn.execute(REPOSITORIES_DDL)
        conn.commit()

    def register(self, name: str, url: str, repo_type: str = "generic",
                 connector_name: str = "", metadata: Optional[dict] = None) -> dict:
        conn = self._connect()
        now = datetime.now().isoformat()
        existing = self.get(name)
        if existing:
            conn.execute(
                "UPDATE repositories SET url=?, repo_type=?, connector_name=?, "
                "metadata_json=?, updated_at=? WHERE name=?",
                (url, repo_type, connector_name,
                 json.dumps(metadata or {}), now, name),
            )
        else:
            conn.execute(
                "INSERT INTO repositories (name, url, repo_type, connector_name, "
                "metadata_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (name, url, repo_type, connector_name,
                 json.dumps(metadata or {}), now, now),
            )
        conn.commit()
        return self.get(name)

    def get(self, name: str) -> dict:
        conn = self._connect()
        row = conn.execute("SELECT * FROM repositories WHERE name=?", (name,)).fetchone()
        if row is None:
            return {}
        return dict(row)

    def list_enabled(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM repositories WHERE enabled=1 ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM repositories ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def mark_synced(self, name: str, version: str = ""):
        conn = self._connect()
        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE repositories SET last_sync_at=?, last_version=?, updated_at=? WHERE name=?",
            (now, version, now, name),
        )
        conn.commit()

    def check_updates(self, name: str) -> bool:
        repo = self.get(name)
        if not repo:
            return False
        return True

    def enable(self, name: str, enabled: bool = True):
        conn = self._connect()
        conn.execute("UPDATE repositories SET enabled=? WHERE name=?", (1 if enabled else 0, name))
        conn.commit()

    def delete(self, name: str):
        conn = self._connect()
        conn.execute("DELETE FROM repositories WHERE name=?", (name,))
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
