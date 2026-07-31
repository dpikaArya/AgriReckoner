import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class RequestCache:
    def __init__(self, cache_dir: Path = Path("data/cache"), ttl_hours: int = 1):
        self._cache_dir = cache_dir
        self._ttl_hours = ttl_hours
        self._lock = threading.Lock()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        cache_path = self._path_for_key(key)
        if not cache_path.exists():
            return None
        try:
            with self._lock:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
                if age_hours >= self._ttl_hours:
                    cache_path.unlink(missing_ok=True)
                    return None
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
            return data
        except Exception as e:
            logger.debug("Cache read error for key %s: %s", key, e)
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def set(self, key: str, response: dict):
        cache_path = self._path_for_key(key)
        try:
            with self._lock:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(response, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug("Cache write error for key %s: %s", key, e)

    def make_key(self, endpoint: str, params: dict) -> str:
        serialized = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        raw = f"{endpoint}:{serialized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear(self, older_than_hours: int | None = None):
        threshold = older_than_hours if older_than_hours is not None else self._ttl_hours
        cutoff = time.time() - (threshold * 3600)
        with self._lock:
            for fname in os.listdir(str(self._cache_dir)):
                fpath = self._cache_dir / fname
                if fpath.suffix == ".json":
                    try:
                        if fpath.stat().st_mtime < cutoff:
                            fpath.unlink()
                    except OSError:
                        pass

    def stats(self) -> dict:
        total_entries = 0
        total_size = 0
        oldest = None
        newest = None
        with self._lock:
            for fname in os.listdir(str(self._cache_dir)):
                fpath = self._cache_dir / fname
                if fpath.suffix == ".json":
                    total_entries += 1
                    try:
                        st = fpath.stat()
                        total_size += st.st_size
                        mt = st.st_mtime
                        if oldest is None or mt < oldest:
                            oldest = mt
                        if newest is None or mt > newest:
                            newest = mt
                    except OSError:
                        pass
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "oldest": (
                datetime.fromtimestamp(oldest, tz=timezone.utc).isoformat()
                if oldest is not None
                else None
            ),
            "newest": (
                datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
                if newest is not None
                else None
            ),
        }
