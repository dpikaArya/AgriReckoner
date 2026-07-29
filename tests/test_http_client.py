import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_sources.http_client import HttpClient


@pytest.fixture
def client(tmp_path):
    return HttpClient(
        base_url="https://api.example.com",
        rate_limit=100,
        rate_period=60,
        cache_dir=tmp_path / "cache",
    )


class TestHttpClientRateLimiting:
    def test_rate_limiting(self):
        client = HttpClient(
            base_url="https://api.example.com",
            rate_limit=5,
            rate_period=10,
        )
        assert client._rate_limit == 5
        assert client._rate_period == 10

    def test_throttle_respects_limit(self):
        client = HttpClient(
            base_url="https://api.example.com",
            rate_limit=5,
            rate_period=60,
        )
        client._request_timestamps = [time.time() - 5, time.time() - 3, time.time() - 1]
        with patch.object(time, "sleep") as mock_sleep:
            client._throttle()
            mock_sleep.assert_not_called()


class TestCacheKey:
    def test_cache_key_uniqueness(self, client):
        key1 = client._cache.make_key("/data/weather", {"city": "London"})
        key2 = client._cache.make_key("/data/weather", {"city": "Paris"})
        assert key1 != key2

    def test_cache_key_deterministic(self, client):
        key1 = client._cache.make_key("/data/weather", {"city": "London", "units": "metric"})
        key2 = client._cache.make_key("/data/weather", {"units": "metric", "city": "London"})
        assert key1 == key2


class TestRequestCache:
    @pytest.fixture
    def cache(self, tmp_path):
        from src.data_sources.cache import RequestCache
        return RequestCache(cache_dir=tmp_path / "cache", ttl_hours=1)

    def test_set_and_get(self, cache):
        key = "testkey123"
        data = {"status_code": 200, "content": "hello"}
        cache.set(key, data)
        result = cache.get(key)
        assert result is not None
        assert result["status_code"] == 200
        assert result["content"] == "hello"

    def test_cache_miss_returns_none(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_clear_removes_expired(self, cache):
        key = "expired_key"
        cache.set(key, {"data": "old"})
        cache.clear(older_than_hours=0)
        result = cache.get(key)
        assert result is None

    def test_clear_keeps_fresh(self, cache):
        key = "fresh_key"
        cache.set(key, {"data": "fresh"})
        cache.clear(older_than_hours=24)
        result = cache.get(key)
        assert result is not None


class TestHttpClientRequests:
    def test_get_uses_cache_on_hit(self, client):
        cache_key = client._cache.make_key("/test", {})
        client._cache.set(cache_key, {
            "status_code": 200,
            "headers": {"content-type": "text/plain"},
            "content": "cached response",
            "encoding": "utf-8",
        })
        resp = client.get("/test", use_cache=True)
        assert resp.status_code == 200
        assert resp.text == "cached response"

    def test_stats(self, client):
        stats = client.stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "total_requests" in stats
        assert "avg_response_time_ms" in stats

    def test_clear_cache(self, client, tmp_path):
        cache_key = client._cache.make_key("/test", {})
        client._cache.set(cache_key, {"data": "x"})
        client.clear_cache(older_than_hours=0)
        assert client._cache.get(cache_key) is None

    def test_get_without_cache(self, client):
        with patch.object(client._session, "request") as mock_request:
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.text = "live response"
            mock_response.encoding = "utf-8"
            mock_request.return_value = mock_response

            resp = client.get("/live", use_cache=False)
            assert resp.text == "live response"
            mock_request.assert_called_once()

    def test_post_method(self, client):
        with patch.object(client._session, "request") as mock_request:
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 201
            mock_response.headers = {}
            mock_request.return_value = mock_response

            resp = client.post("/create", json={"name": "test"})
            assert resp.status_code == 201
            mock_request.assert_called_once()

    def test_build_url_absolute_preserved(self, client):
        url = client._build_url("https://other.com/api")
        assert url == "https://other.com/api"

    def test_build_url_relative_joined(self, client):
        url = client._build_url("/data/weather")
        assert url == "https://api.example.com/data/weather"
