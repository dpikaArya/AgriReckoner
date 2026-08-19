"""HTTP transport for connectors.

Wraps the existing ``src.data_sources.http_client.HttpClient`` (which already
provides sliding-window rate limiting, exponential backoff with jitter, a disk
response cache, and structured ``log_api_call`` logging) with JSON convenience
methods tailored to API connectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.data_sources.http_client import HttpClient

logger = logging.getLogger(__name__)


class ConnectorHttpClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        rate_limit_per_minute: int = 30,
        timeout_sec: int = 45,
        max_retries: int = 3,
        cache_dir: str | Path = "literature_data/cache",
    ):
        # HttpClient's rate limiter is expressed as (rate_limit, rate_period).
        # For sub-second-tolerant APIs the base client sleeps before the
        # request; we pass per-minute budgets directly.
        self._client = HttpClient(
            base_url=base_url,
            api_key=api_key,
            rate_limit=max(rate_limit_per_minute, 1),
            rate_period=60,
            cache_dir=Path(cache_dir),
            default_timeout=timeout_sec,
            max_retries=max_retries,
        )
        self._base_url = base_url.rstrip("/")
        self._static_headers = dict(headers or {})
        self._default_timeout = timeout_sec

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------ #
    # GET helpers
    # ------------------------------------------------------------------ #
    def get_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        *,
        use_cache: bool = True,
        timeout: int | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        """GET a JSON payload; returns parsed object or None on non-2xx."""
        merged = dict(self._static_headers)
        if headers:
            merged.update(headers)
        resp = self._client.get(
            endpoint,
            params=params,
            headers=merged or None,
            timeout=timeout or self._default_timeout,
            use_cache=use_cache,
        )
        if resp is None or resp.status_code >= 400:
            if resp is not None and resp.status_code >= 400:
                logger.debug("GET %s -> %d %s", endpoint, resp.status_code, resp.text[:200])
            return None
        try:
            return resp.json()
        except ValueError:
            logger.debug("GET %s -> non-JSON response", endpoint)
            return None

    def get_text(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        *,
        use_cache: bool = True,
        timeout: int | None = None,
    ) -> str | None:
        merged = dict(self._static_headers)
        if headers:
            merged.update(headers)
        resp = self._client.get(
            endpoint,
            params=params,
            headers=merged or None,
            timeout=timeout or self._default_timeout,
            use_cache=use_cache,
        )
        if resp is None or resp.status_code >= 400:
            return None
        return resp.text

    def get_bytes(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        timeout: int | None = None,
    ) -> bytes | None:
        """Download raw bytes from an absolute URL (for PDFs / datasets)."""
        merged = dict(self._static_headers)
        if headers:
            merged.update(headers)
        resp = self._client.get(
            url,
            headers=merged or None,
            timeout=timeout or self._default_timeout,
            use_cache=False,
        )
        if resp is None or resp.status_code >= 400:
            return None
        return resp.content

    def head(self, url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
        merged = dict(self._static_headers)
        if headers:
            merged.update(headers)
        try:
            resp = self._client.get(url, headers=merged or None, use_cache=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("HEAD-ish GET failed for %s: %s", url, exc)
            return 0, ""
        content_type = resp.headers.get("Content-Type", "")
        return resp.status_code, content_type

    # ------------------------------------------------------------------ #
    # POST helpers
    # ------------------------------------------------------------------ #
    def post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        merged = dict(self._static_headers)
        if headers:
            merged.update(headers)
        resp = self._client.post(
            endpoint,
            json=payload,
            headers=merged or None,
            timeout=timeout or self._default_timeout,
        )
        if resp is None or resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def clear_cache(self, older_than_hours: int = 24) -> None:
        self._client.clear_cache(older_than_hours=older_than_hours)

    def stats(self) -> dict:
        return self._client.stats()
