"""Tests for the source registry repository, service, and scraper integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import textwrap
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Lightweight in-memory fake for regulation_sources + app_settings tables
# ---------------------------------------------------------------------------

@dataclass
class _FakeResult:
    data: list[dict[str, Any]]
    count: int | None = None


class _FakeTableQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str):
        self._store = store
        self._table = table
        self._filters: dict[str, Any] = {}
        self._order_col: str | None = None
        self._count_mode: str | None = None

    def select(self, cols: str, **kwargs: Any) -> "_FakeTableQuery":
        if "count" in kwargs:
            self._count_mode = kwargs["count"]
        return self

    def eq(self, col: str, val: Any) -> "_FakeTableQuery":
        self._filters[col] = val
        return self

    def in_(self, col: str, vals: list) -> "_FakeTableQuery":
        self._filters[f"__in__{col}"] = vals
        return self

    def order(self, col: str) -> "_FakeTableQuery":
        self._order_col = col
        return self

    def limit(self, _n: int) -> "_FakeTableQuery":
        return self

    def execute(self) -> _FakeResult:
        rows = list(self._store.get(self._table, []))
        for col, val in self._filters.items():
            if col.startswith("__in__"):
                real_col = col[len("__in__"):]
                rows = [r for r in rows if r.get(real_col) in val]
            else:
                rows = [r for r in rows if r.get(col) == val]
        if self._order_col:
            rows.sort(key=lambda r: str(r.get(self._order_col, "")))
        return _FakeResult(data=rows, count=len(rows) if self._count_mode else None)


class _FakeInsertQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str, payloads: list[dict]):
        self._store = store
        self._table = table
        self._payloads = payloads

    def execute(self) -> _FakeResult:
        rows = self._store.setdefault(self._table, [])
        inserted = []
        for p in self._payloads:
            new_row = dict(p)
            new_row.setdefault("id", len(rows) + 1)
            rows.append(new_row)
            inserted.append(new_row)
        return _FakeResult(data=inserted)


class _FakeUpdateQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str, payload: dict):
        self._store = store
        self._table = table
        self._payload = payload
        self._filters: dict[str, Any] = {}

    def eq(self, col: str, val: Any) -> "_FakeUpdateQuery":
        self._filters[col] = val
        return self

    def execute(self) -> _FakeResult:
        updated = []
        for row in self._store.get(self._table, []):
            if all(row.get(k) == v for k, v in self._filters.items()):
                row.update(self._payload)
                updated.append(row)
        return _FakeResult(data=updated)


class _FakeUpsertQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str, payload: dict, conflict: str):
        self._store = store
        self._table = table
        self._payload = payload
        self._conflict = conflict

    def execute(self) -> _FakeResult:
        rows = self._store.setdefault(self._table, [])
        conflict_val = self._payload.get(self._conflict)
        for row in rows:
            if row.get(self._conflict) == conflict_val:
                row.update(self._payload)
                return _FakeResult(data=[row])
        new_row = dict(self._payload)
        new_row.setdefault("id", len(rows) + 1)
        rows.append(new_row)
        return _FakeResult(data=[new_row])


class _FakeDeleteQuery:
    def __init__(self, store: dict[str, list[dict[str, Any]]], table: str):
        self._store = store
        self._table = table
        self._filters: dict[str, Any] = {}

    def eq(self, col: str, val: Any) -> "_FakeDeleteQuery":
        self._filters[col] = val
        return self

    def execute(self) -> _FakeResult:
        rows = self._store.get(self._table, [])
        removed = [r for r in rows if all(r.get(k) == v for k, v in self._filters.items())]
        self._store[self._table] = [r for r in rows if r not in removed]
        return _FakeResult(data=removed)


class FakeDB:
    """In-memory fake Supabase client for source-registry tests."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {}

    def table(self, name: str) -> Any:
        return _FakeTable(self._data, name)

    def seed(self, table: str, rows: list[dict[str, Any]]) -> None:
        self._data[table] = list(rows)


class _FakeTable:
    def __init__(self, store: dict[str, list[dict[str, Any]]], name: str):
        self._store = store
        self._name = name

    def select(self, cols: str = "*", **kwargs: Any) -> _FakeTableQuery:
        q = _FakeTableQuery(self._store, self._name)
        return q.select(cols, **kwargs)

    def insert(self, payloads: list[dict]) -> _FakeInsertQuery:
        return _FakeInsertQuery(self._store, self._name, payloads)

    def update(self, payload: dict) -> _FakeUpdateQuery:
        return _FakeUpdateQuery(self._store, self._name, payload)

    def upsert(self, payload: dict, *, on_conflict: str = "") -> _FakeUpsertQuery:
        return _FakeUpsertQuery(self._store, self._name, payload, on_conflict)

    def delete(self) -> _FakeDeleteQuery:
        return _FakeDeleteQuery(self._store, self._name)

    def eq(self, col: str, val: Any) -> _FakeTableQuery:
        q = _FakeTableQuery(self._store, self._name)
        return q.eq(col, val)

    def order(self, col: str) -> _FakeTableQuery:
        q = _FakeTableQuery(self._store, self._name)
        return q.order(col)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_db() -> FakeDB:
    return FakeDB()


@pytest.fixture()
def app_settings_repo(fake_db: FakeDB):
    from core.regulations.source_registry import AppSettingsRepo
    return AppSettingsRepo(db_getter=lambda: fake_db)


@pytest.fixture()
def source_repo(fake_db: FakeDB):
    from core.regulations.source_registry import SourceRepository
    return SourceRepository(db_getter=lambda: fake_db)


@pytest.fixture()
def registry_service(app_settings_repo, source_repo):
    from core.regulations.source_registry import SourceRegistryService
    return SourceRegistryService(settings_repo=app_settings_repo, source_repo=source_repo)


# ---------------------------------------------------------------------------
# AppSettingsRepo tests
# ---------------------------------------------------------------------------

class TestAppSettingsRepo:
    def test_get_missing_key_returns_default(self, app_settings_repo):
        assert app_settings_repo.get("nonexistent") is None
        assert app_settings_repo.get("nonexistent", "fallback") == "fallback"

    def test_set_and_get(self, app_settings_repo):
        app_settings_repo.set("my_flag", "true")
        assert app_settings_repo.get("my_flag") == "true"

    def test_get_bool(self, app_settings_repo):
        app_settings_repo.set("flag_on", "true")
        app_settings_repo.set("flag_off", "false")
        assert app_settings_repo.get_bool("flag_on") is True
        assert app_settings_repo.get_bool("flag_off") is False
        assert app_settings_repo.get_bool("missing", default=True) is True

    def test_upsert_overwrites(self, app_settings_repo):
        app_settings_repo.set("key", "v1")
        app_settings_repo.set("key", "v2")
        assert app_settings_repo.get("key") == "v2"


# ---------------------------------------------------------------------------
# SourceRepository tests
# ---------------------------------------------------------------------------

class TestSourceRepository:
    def test_insert_and_list(self, source_repo):
        source_repo.insert({
            "jurisdiction_id": 1,
            "source_name": "Test Law",
            "url": "https://example.com/law",
            "domain": "housing",
            "category": "General",
            "is_active": True,
        })
        rows = source_repo.list_all()
        assert len(rows) == 1
        assert rows[0]["source_name"] == "Test Law"

    def test_get_by_url(self, source_repo):
        source_repo.insert({
            "jurisdiction_id": 1,
            "source_name": "A",
            "url": "https://example.com/a",
            "is_active": True,
        })
        assert source_repo.get_by_url("https://example.com/a") is not None
        assert source_repo.get_by_url("https://example.com/missing") is None

    def test_active_only_filter(self, source_repo):
        source_repo.insert({"jurisdiction_id": 1, "source_name": "Active", "url": "https://a.com", "is_active": True})
        source_repo.insert({"jurisdiction_id": 1, "source_name": "Inactive", "url": "https://b.com", "is_active": False})
        assert len(source_repo.list_all(active_only=True)) == 1
        assert len(source_repo.list_all(active_only=False)) == 2

    def test_delete(self, source_repo):
        source_repo.insert({"jurisdiction_id": 1, "source_name": "Del", "url": "https://d.com", "is_active": True})
        rows = source_repo.list_all()
        assert len(rows) == 1
        source_repo.delete(rows[0]["id"])
        assert len(source_repo.list_all()) == 0

    def test_update(self, source_repo):
        source_repo.insert({"jurisdiction_id": 1, "source_name": "Old", "url": "https://u.com", "is_active": True})
        rows = source_repo.list_all()
        source_repo.update(rows[0]["id"], {"source_name": "New"})
        updated = source_repo.list_all()
        assert updated[0]["source_name"] == "New"

    def test_table_exists(self, source_repo):
        assert source_repo.table_exists() is True


# ---------------------------------------------------------------------------
# SourceRegistryService tests
# ---------------------------------------------------------------------------

class TestSourceRegistryService:
    def test_toggle_default_off(self, registry_service):
        assert registry_service.is_db_registry_enabled() is False

    def test_toggle_on_off(self, registry_service):
        registry_service.set_db_registry_enabled(True)
        assert registry_service.is_db_registry_enabled() is True
        registry_service.set_db_registry_enabled(False)
        assert registry_service.is_db_registry_enabled() is False

    def test_backfill_from_csv(self, registry_service, fake_db, monkeypatch):
        fake_db.seed("jurisdictions", [
            {"id": 1, "type": "federal", "name": "United States", "parent_id": None, "state_code": None},
            {"id": 10, "type": "state", "name": "Texas", "parent_id": 1, "state_code": "TX"},
        ])

        csv_content = textwrap.dedent("""\
            category,city_name,law_name,hyperlink,state_code
            State,Texas-Statewide,Texas Property Code,https://example.com/tx-prop,TX
            State,Texas-Statewide,Texas Penal Code,https://example.com/tx-penal,TX
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        import db.client as _db_mod
        monkeypatch.setattr(_db_mod, "get_db", lambda: fake_db)
        import core.regulations.scraper as _scraper_mod
        monkeypatch.setattr(_scraper_mod, "get_db", lambda: fake_db)

        result = registry_service.backfill_from_csv(csv_path)
        assert result["imported"] == 2
        assert result["skipped"] == 0

        result2 = registry_service.backfill_from_csv(csv_path)
        assert result2["imported"] == 0
        assert result2["skipped"] == 2

    def test_test_source_unreachable(self, registry_service, monkeypatch):
        import requests
        import core.regulations.source_registry as _sr_mod

        def _raise(*a, **kw):
            raise requests.ConnectionError("nope")

        monkeypatch.setattr(_sr_mod.requests, "get", _raise)
        result = registry_service.test_source("https://unreachable.invalid")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Scraper provider-switch tests
# ---------------------------------------------------------------------------

class TestScraperProviderSwitch:
    def test_use_db_registry_flag(self, monkeypatch):
        from core.regulations.scraper import RegulationScraper

        scraper = RegulationScraper()

        import core.regulations.source_registry as _sr
        monkeypatch.setattr(_sr.source_registry, "is_db_registry_enabled", lambda: True)
        assert scraper._use_db_registry() is True

        monkeypatch.setattr(_sr.source_registry, "is_db_registry_enabled", lambda: False)
        assert scraper._use_db_registry() is False
