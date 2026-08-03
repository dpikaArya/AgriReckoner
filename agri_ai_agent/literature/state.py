"""Local connector state: incremental-sync cursors and a metadata cache.

Each connector keeps two kinds of persistent state in a single SQLite file:

* ``connector_state`` — arbitrary key/value cursors (e.g. ``last_sync_at``,
  ``last_pubmed_uid``, ``last_created_gte``) used to make
  ``incremental_sync`` resumable and cheap across runs.
* ``raw_cache`` — the local metadata cache mandated by the spec: every raw
  API response payload is stored here so re-runs do not refetch unchanged
  records, and so provenance can be reconstructed offline.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConnectorStateStore:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS connector_state (
                    connector  TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (connector, key)
                );
                CREATE TABLE IF NOT EXISTS raw_cache (
                    source     TEXT NOT NULL,
                    source_id  TEXT NOT NULL,
                    payload    TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (source, source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_raw_cache_source
                    ON raw_cache (source, fetched_at);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # incremental-sync cursors
    # ------------------------------------------------------------------ #
    def set_cursor(self, connector: str, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO connector_state (connector, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(connector, key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (connector, key, json.dumps(value, default=str), self._now()),
            )

    def get_cursor(self, connector: str, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM connector_state WHERE connector = ? AND key = ?",
                (connector, key),
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def list_cursors(self, connector: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM connector_state WHERE connector = ?",
                (connector,),
            ).fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (json.JSONDecodeError, TypeError):
                out[r["key"]] = r["value"]
        return out

    # ------------------------------------------------------------------ #
    # local metadata cache
    # ------------------------------------------------------------------ #
    def cache_put(self, source: str, source_id: str, payload: dict[str, Any]) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO raw_cache (source, source_id, payload, fetched_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source, source_id)
                    DO UPDATE SET payload = excluded.payload, fetched_at = excluded.fetched_at
                    """,
                    (source, source_id, json.dumps(payload, default=str), self._now()),
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("raw_cache write failed for %s/%s: %s", source, source_id, exc)

    def cache_get(self, source: str, source_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM raw_cache WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            return None

    def cache_has(self, source: str, source_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM raw_cache WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
        return row is not None

    def cache_count(self, source: str | None = None) -> int:
        with self._connect() as conn:
            if source:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM raw_cache WHERE source = ?", (source,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM raw_cache").fetchone()
        return int(row["n"]) if row else 0

    def iter_cache(self, source: str) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(source_id, payload)`` pairs for a given source."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_id, payload FROM raw_cache WHERE source = ?", (source,)
            ).fetchall()
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                continue
            out.append((row["source_id"], payload))
        return out

    def iter_sources(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT source FROM raw_cache").fetchall()
        return [r["source"] for r in rows]

    # ------------------------------------------------------------------ #
    # stats / bookkeeping
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            cursors = conn.execute(
                "SELECT connector, COUNT(*) AS n FROM connector_state GROUP BY connector"
            ).fetchall()
            cached = conn.execute("SELECT COUNT(*) AS n FROM raw_cache").fetchone()
        return {
            "cursors_by_connector": {r["connector"]: r["n"] for r in cursors},
            "cached_records": int(cached["n"]) if cached else 0,
        }

    def close(self) -> None:
        """No-op: connections are short-lived."""
