"""Unit tests for the literature module's SQLite state store."""

from __future__ import annotations

from agri_ai_agent.literature.state import ConnectorStateStore


def test_cursor_round_trip(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.set_cursor("OpenAlex", "last_sync_at", "2026-01-01T00:00:00+00:00")
    store.set_cursor("OpenAlex", "offset", 120)
    store.set_cursor("OpenAlex", "flags", {"a": True})

    assert store.get_cursor("OpenAlex", "last_sync_at") == "2026-01-01T00:00:00+00:00"
    assert store.get_cursor("OpenAlex", "offset") == 120
    assert store.get_cursor("OpenAlex", "flags") == {"a": True}
    assert store.get_cursor("OpenAlex", "missing", "default") == "default"


def test_cursor_overwrite(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.set_cursor("A", "k", 1)
    store.set_cursor("A", "k", 2)
    assert store.get_cursor("A", "k") == 2


def test_list_cursors(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.set_cursor("A", "k1", 1)
    store.set_cursor("A", "k2", "x")
    cursors = store.list_cursors("A")
    assert cursors == {"k1": 1, "k2": "x"}


def test_cache_round_trip(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    assert not store.cache_has("Crossref", "id-1")
    store.cache_put("Crossref", "id-1", {"title": "A study"})
    assert store.cache_has("Crossref", "id-1")
    assert store.cache_get("Crossref", "id-1") == {"title": "A study"}
    assert store.cache_count() == 1
    assert store.cache_count("Crossref") == 1
    assert store.cache_count("Other") == 0


def test_cache_upsert(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.cache_put("A", "x", {"n": 1})
    store.cache_put("A", "x", {"n": 2})
    assert store.cache_get("A", "x") == {"n": 2}
    assert store.cache_count() == 1


def test_iter_cache_and_sources(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.cache_put("A", "a1", {"v": 1})
    store.cache_put("A", "a2", {"v": 2})
    store.cache_put("B", "b1", {"v": 3})

    items = store.iter_cache("A")
    assert sorted(sid for sid, _ in items) == ["a1", "a2"]
    assert any(payload == {"v": 2} for _, payload in items)

    assert sorted(store.iter_sources()) == ["A", "B"]


def test_summary(tmp_path):
    store = ConnectorStateStore(tmp_path / "state.sqlite")
    store.set_cursor("A", "k", 1)
    store.cache_put("A", "x", {})
    summary = store.summary()
    assert summary["cursors_by_connector"] == {"A": 1}
    assert summary["cached_records"] == 1


def test_persists_across_instances(tmp_path):
    path = tmp_path / "state.sqlite"
    ConnectorStateStore(path).set_cursor("A", "k", "value")
    fresh = ConnectorStateStore(path)
    assert fresh.get_cursor("A", "k") == "value"
