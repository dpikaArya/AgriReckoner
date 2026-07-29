import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from src.data_sources.cache import RequestCache
from src.utils.logging_config import log_api_call

logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = None,
        rate_limit: int = 10,
        rate_period: int = 60,
        cache_dir: Path = Path("data/cache"),
        default_timeout: int = 30,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._rate_limit = rate_limit
        self._rate_period = rate_period
        self._default_timeout = default_timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._request_timestamps: list[float] = []

        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        self._total_response_time = 0.0

        self._cache = RequestCache(cache_dir=Path(cache_dir), ttl_hours=1)

        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self):
        with self._lock:
            now = time.time()
            cutoff = now - self._rate_period
            self._request_timestamps = [
                t for t in self._request_timestamps if t > cutoff
            ]
            if len(self._request_timestamps) >= self._rate_limit:
                sleep_for = (
                    self._request_timestamps[0] + self._rate_period - now
                )
                if sleep_for > 0:
                    logger.debug(
                        "Rate limit reached, sleeping %.2fs", sleep_for
                    )
                    time.sleep(sleep_for)
            self._request_timestamps.append(time.time())

    # ------------------------------------------------------------------
    # URL building
    # ------------------------------------------------------------------

    def _build_url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: dict = None,
        headers: dict = None,
        timeout: int = None,
        use_cache: bool = True,
    ) -> requests.Response:
        url = self._build_url(endpoint)
        cache_key = None
        if use_cache:
            cache_key = self._cache.make_key(endpoint, params or {})
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._total_requests += 1
                resp = requests.Response()
                resp.status_code = cached.get("status_code", 200)
                resp.headers.update(cached.get("headers", {}))
                content = cached.get("content", "")
                resp._content = (
                    content.encode("utf-8")
                    if isinstance(content, str)
                    else content
                )
                resp.url = url
                resp.encoding = cached.get("encoding", "utf-8")
                resp.request = requests.Request("GET", url).prepare()
                return resp

        self._cache_misses += 1
        resp = self._request(
            "GET", url, params=params, headers=headers, timeout=timeout
        )
        if use_cache and cache_key and resp.status_code < 400:
            self._cache.set(
                cache_key,
                {
                    "status_code": resp.status_code,
                    "headers": dict(resp.headers),
                    "content": resp.text,
                    "encoding": resp.encoding or "utf-8",
                },
            )
        return resp

    def post(
        self,
        endpoint: str,
        json: dict = None,
        headers: dict = None,
        timeout: int = None,
    ) -> requests.Response:
        url = self._build_url(endpoint)
        return self._request(
            "POST", url, json=json, headers=headers, timeout=timeout
        )

    # ------------------------------------------------------------------
    # Core request execution with retry
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        params: dict = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = None,
    ) -> requests.Response:
        timeout = timeout if timeout is not None else self._default_timeout
        merged_headers = dict(self._session.headers)
        if headers:
            merged_headers.update(headers)

        last_exc = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            start = time.perf_counter()
            logger.debug(
                "HTTP %s %s (attempt %d/%d)",
                method, url, attempt + 1, self._max_retries + 1,
            )
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=merged_headers,
                    timeout=timeout,
                )
                duration = (time.perf_counter() - start) * 1000
                with self._lock:
                    self._total_requests += 1
                    self._total_response_time += duration

                log_level = (
                    logging.WARNING
                    if resp.status_code >= 400
                    else logging.INFO
                )
                logger.log(
                    log_level,
                    "HTTP %s %s -> %d (%.1fms, attempt %d/%d)",
                    method,
                    url,
                    resp.status_code,
                    duration,
                    attempt + 1,
                    self._max_retries + 1,
                )

                log_api_call(
                    logger=logger,
                    method=method,
                    url=url,
                    status=resp.status_code,
                    duration_ms=duration,
                    retries=attempt,
                )

                should_retry_status = (
                    resp.status_code == 429 or resp.status_code >= 500
                )
                if should_retry_status and attempt < self._max_retries:
                    delay = (2 ** attempt) * (
                        1 + random.uniform(-0.25, 0.25)
                    )
                    logger.warning(
                        "Retrying %s %s attempt %d after %.1fs due to status %d",
                        method,
                        url,
                        attempt + 1,
                        delay,
                        resp.status_code,
                    )
                    time.sleep(max(delay, 0.1))
                    continue
                return resp

            except requests.Timeout:
                duration = (time.perf_counter() - start) * 1000
                with self._lock:
                    self._total_requests += 1
                    self._total_response_time += duration
                logger.warning(
                    "HTTP %s %s timed out after %.1fms (attempt %d/%d)",
                    method,
                    url,
                    duration,
                    attempt + 1,
                    self._max_retries + 1,
                )
                if attempt < self._max_retries:
                    delay = (2 ** attempt) * (
                        1 + random.uniform(-0.25, 0.25)
                    )
                    time.sleep(max(delay, 0.1))
                    continue
                raise

            except (
                requests.ConnectionError,
                requests.exceptions.ConnectionError,
            ) as exc:
                duration = (time.perf_counter() - start) * 1000
                with self._lock:
                    self._total_requests += 1
                    self._total_response_time += duration
                last_exc = exc
                logger.warning(
                    "HTTP %s %s connection error (attempt %d/%d): %s",
                    method,
                    url,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    delay = (2 ** attempt) * (
                        1 + random.uniform(-0.25, 0.25)
                    )
                    time.sleep(max(delay, 0.1))
                    continue
                raise last_exc

        raise RuntimeError("Unreachable: all retries exhausted")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, older_than_hours: int = 24):
        self._cache.clear(older_than_hours=older_than_hours)

    def stats(self) -> dict:
        with self._lock:
            avg_time = (
                self._total_response_time / self._total_requests
                if self._total_requests > 0
                else 0.0
            )
            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "total_requests": self._total_requests,
                "avg_response_time_ms": round(avg_time, 2),
            }
