"""
Source registry: manages regulation_sources and the CSV-vs-DB toggle.

Layers
------
AppSettingsRepo   – read/write app_settings rows (feature flags).
SourceRepository  – CRUD for regulation_sources table.
SourceRegistryService – orchestration: backfill, toggle, test-source.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from config import settings
from db.client import get_db
from db.models import RegulationSource

logger = logging.getLogger(__name__)

# Re-use the canonical maps from scraper to avoid duplication.
from core.regulations.scraper import (
    _resolve_jurisdiction_id,
    _infer_state_code,
)

_TOGGLE_KEY = "use_db_source_registry"


# ---------------------------------------------------------------------------
# App settings repository
# ---------------------------------------------------------------------------

class AppSettingsRepo:
    """Thin wrapper around the app_settings table."""

    def __init__(self, db_getter=get_db):
        self._db = db_getter

    def get(self, key: str, default: str | None = None) -> str | None:
        try:
            db = self._db()
            res = db.table("app_settings").select("value").eq("key", key).limit(1).execute()
            if res.data:
                return str(res.data[0]["value"])
        except Exception:
            logger.debug("app_settings table may not exist yet; returning default for '%s'", key)
        return default

    def set(self, key: str, value: str) -> None:
        db = self._db()
        db.table("app_settings").upsert(
            {"key": key, "value": value, "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="key",
        ).execute()

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Source repository (regulation_sources CRUD)
# ---------------------------------------------------------------------------

class SourceRepository:
    """Direct DB operations for the regulation_sources table."""

    def __init__(self, db_getter=get_db):
        self._db = db_getter

    def list_all(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        db = self._db()
        q = db.table("regulation_sources").select("*").order("source_name")
        if active_only:
            q = q.eq("is_active", True)
        return q.execute().data or []

    def get_by_id(self, source_id: int) -> dict[str, Any] | None:
        db = self._db()
        res = db.table("regulation_sources").select("*").eq("id", source_id).limit(1).execute()
        return res.data[0] if res.data else None

    def get_by_url(self, url: str) -> dict[str, Any] | None:
        db = self._db()
        res = db.table("regulation_sources").select("*").eq("url", url.strip()).limit(1).execute()
        return res.data[0] if res.data else None

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert or update by URL (unique index)."""
        db = self._db()
        res = db.table("regulation_sources").upsert(payload, on_conflict="url").execute()
        if res.data:
            return res.data[0]
        existing = self.get_by_url(payload["url"])
        if existing:
            return existing
        raise RuntimeError(f"Failed to upsert regulation_source for url={payload.get('url')}")

    def insert(self, payload: dict[str, Any]) -> dict[str, Any]:
        db = self._db()
        res = db.table("regulation_sources").insert([payload]).execute()
        if res.data:
            return res.data[0]
        raise RuntimeError("Insert into regulation_sources returned no data")

    def update(self, source_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
        db = self._db()
        res = db.table("regulation_sources").update(payload).eq("id", source_id).execute()
        return res.data[0] if res.data else None

    def delete(self, source_id: int) -> bool:
        db = self._db()
        res = db.table("regulation_sources").delete().eq("id", source_id).execute()
        return bool(res.data)

    def update_scrape_status(
        self,
        source_id: int,
        *,
        last_scraped_at: datetime | None = None,
        last_error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if last_scraped_at is not None:
            payload["last_scraped_at"] = last_scraped_at.isoformat()
        payload["last_error"] = last_error
        if payload:
            db = self._db()
            db.table("regulation_sources").update(payload).eq("id", source_id).execute()

    def list_paginated(
        self, *, offset: int = 0, limit: int = 25, active_only: bool = False,
    ) -> list[dict[str, Any]]:
        db = self._db()
        q = db.table("regulation_sources").select("*").order("source_name")
        if active_only:
            q = q.eq("is_active", True)
        q = q.range(offset, offset + limit - 1)
        return q.execute().data or []

    def count(self, *, active_only: bool = False) -> int:
        db = self._db()
        q = db.table("regulation_sources").select("id", count="exact")
        if active_only:
            q = q.eq("is_active", True)
        res = q.execute()
        return res.count if hasattr(res, "count") and res.count is not None else len(res.data or [])

    def table_exists(self) -> bool:
        try:
            db = self._db()
            res = db.table("regulation_sources").select("id").limit(1).execute()
            # Supabase PostgREST returns 200 with empty data for an empty table,
            # but raises or returns an error object for missing/forbidden tables.
            # res.data being a list (even empty) means the table is accessible.
            return isinstance(res.data, list)
        except Exception as exc:
            logger.warning("regulation_sources table check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------

class SourceRegistryService:
    """Orchestrates source registry operations."""

    def __init__(
        self,
        settings_repo: AppSettingsRepo | None = None,
        source_repo: SourceRepository | None = None,
    ):
        self._settings = settings_repo or AppSettingsRepo()
        self._sources = source_repo or SourceRepository()

    # -- toggle management --

    def is_db_registry_enabled(self) -> bool:
        """Check if the DB source registry is the active provider."""
        return self._settings.get_bool(_TOGGLE_KEY, default=settings.USE_DB_SOURCE_REGISTRY)

    def set_db_registry_enabled(self, enabled: bool) -> None:
        self._settings.set(_TOGGLE_KEY, "true" if enabled else "false")

    def registry_table_exists(self) -> bool:
        return self._sources.table_exists()

    # -- source CRUD (delegates) --

    def list_sources(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        return self._sources.list_all(active_only=active_only)

    def get_source(self, source_id: int) -> dict[str, Any] | None:
        return self._sources.get_by_id(source_id)

    def add_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._sources.insert(payload)

    def update_source(self, source_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
        return self._sources.update(source_id, payload)

    def delete_source(self, source_id: int) -> bool:
        return self._sources.delete(source_id)

    def toggle_source_active(self, source_id: int, is_active: bool) -> dict[str, Any] | None:
        return self._sources.update(source_id, {"is_active": is_active})

    # -- CSV -> DB backfill --

    def backfill_from_csv(self, csv_path: Path) -> dict[str, int]:
        """
        Idempotent import: reads sources.csv and upserts into regulation_sources.
        Returns {"imported": N, "skipped": N, "errors": N}.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing seed file: {csv_path}")

        db = get_db()
        imported = 0
        skipped = 0
        errors = 0

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get("hyperlink") or "").strip()
                if not url:
                    continue

                category = (row.get("category") or "").strip()
                city_name = (row.get("city_name") or "").strip()
                law_name = (row.get("law_name") or "").strip()
                state_code = (row.get("state_code") or "").strip().upper()
                if not state_code:
                    state_code = _infer_state_code(city_name) or ""

                if self._sources.get_by_url(url) is not None:
                    skipped += 1
                    continue

                try:
                    jurisdiction_id = _resolve_jurisdiction_id(db, category, city_name, state_code)
                except Exception as exc:
                    logger.warning("Skipping CSV row url=%s: %s", url, exc)
                    errors += 1
                    continue

                payload = {
                    "jurisdiction_id": jurisdiction_id,
                    "source_name": law_name or "Unknown",
                    "url": url,
                    "domain": "housing",
                    "category": category or "General",
                    "state_code": state_code or None,
                    "is_active": True,
                }

                try:
                    self._sources.insert(payload)
                    imported += 1
                except Exception as exc:
                    logger.warning("Failed to insert source url=%s: %s", url, exc)
                    errors += 1

        return {"imported": imported, "skipped": skipped, "errors": errors}

    # -- pagination helpers --

    def list_sources_paginated(
        self, *, offset: int = 0, limit: int = 25, active_only: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (page_rows, total_count)."""
        rows = self._sources.list_paginated(offset=offset, limit=limit, active_only=active_only)
        total = self._sources.count(active_only=active_only)
        return rows, total

    # -- scrape history --

    def scrape_history_for_url(self, url: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """
        Return version history from the `regulations` table for a given URL.
        Each row represents a scraped version (version, content_hash, created_at, is_current).
        """
        try:
            db = get_db()
            res = (
                db.table("regulations")
                .select("id,version,content_hash,is_current,created_at")
                .eq("url", url)
                .order("version", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception:
            return []

    # -- CSV export --

    def export_sources_csv(self) -> str:
        """Return all sources as a CSV string."""
        rows = self._sources.list_all()
        if not rows:
            return ""
        import io as _io
        buf = _io.StringIO()
        fieldnames = ["id", "source_name", "url", "category", "domain", "state_code",
                       "is_active", "last_scraped_at", "last_error", "jurisdiction_id"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    # -- test source connectivity --

    def test_source(self, url: str, *, timeout: int = 15) -> dict[str, Any]:
        """
        Probe a URL and return reachability + content-type + content length.
        Does NOT scrape or store anything.
        """
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            content_type = (resp.headers.get("content-type") or "unknown").split(";")[0].strip()
            content_length = len(resp.content)
            return {
                "ok": resp.status_code < 400,
                "status_code": resp.status_code,
                "content_type": content_type,
                "content_length": content_length,
            }
        except requests.RequestException as exc:
            return {"ok": False, "status_code": None, "error": str(exc)}


# Module-level singletons for convenient import.
app_settings_repo = AppSettingsRepo()
source_repo = SourceRepository()
source_registry = SourceRegistryService()
