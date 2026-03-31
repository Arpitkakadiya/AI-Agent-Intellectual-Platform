from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from core.rag.vector_store import RegulationVectorStore
from db.client import get_db
from db.models import Regulation

logger = logging.getLogger(__name__)

STATE_NAME_TO_CODE: dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

CITY_TO_STATE_CODE: dict[str, str] = {
    "Los Angeles": "CA",
    "San Francisco": "CA",
    "San Diego": "CA",
    "New York City": "NY",
    "Dallas": "TX",
    "Houston": "TX",
    "Austin": "TX",
    "San Antonio": "TX",
    "Fort Worth": "TX",
}

FEDERAL_NAME_ALIASES: dict[str, str] = {
    "Federal Government": "United States",
    "United States": "United States",
}
CODE_TO_STATE_NAME: dict[str, str] = {code: name for name, code in STATE_NAME_TO_CODE.items()}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _infer_state_code(city_name: str) -> str | None:
    """
    Best-effort state_code inference for backward compatibility.
    Prefer using the CSV `state_code` column when present.
    """
    label = (city_name or "").strip()
    if not label:
        return None

    if label.lower().endswith("-statewide"):
        # Example: "California-Statewide" -> "California"
        state_part = label.split("-", 1)[0].strip()
        state_part = state_part.replace("NewYork", "New York")
        return STATE_NAME_TO_CODE.get(state_part)

    if label in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[label]

    # City name fallback.
    return CITY_TO_STATE_CODE.get(label)


def _get_state_id_by_code(db: Any, state_code: str) -> int:
    res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "state")
        .eq("state_code", state_code)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(
            f"State with state_code='{state_code}' not found. Run scripts/seed_jurisdictions.py first."
        )
    return int(res.data[0]["id"])


def _get_federal_id(db: Any) -> int:
    # Some earlier seeds used "Federal Government" vs "United States".
    res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "federal")
        .in_("name", list(FEDERAL_NAME_ALIASES.keys()))
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(
            "Federal jurisdiction row missing. Run scripts/seed_jurisdictions.py first."
        )
    return int(res.data[0]["id"])


def _resolve_jurisdiction_id(
    db: Any, category: str, city_name: str, state_code: str
) -> int:
    kind = (category or "").strip().lower()
    label = (city_name or "").strip()
    code = (state_code or "").strip().upper()

    if kind == "federal":
        return _get_federal_id(db)

    if kind == "state":
        if not code:
            raise RuntimeError("Missing state_code for state jurisdiction row.")
        return _get_state_id_by_code(db, code)

    if kind == "city":
        if not code:
            raise RuntimeError("Missing state_code for city jurisdiction row.")
        city_res = (
            db.table("jurisdictions")
            .select("id")
            .eq("type", "city")
            .eq("name", label)
            .eq("state_code", code)
            .limit(1)
            .execute()
        )
        if city_res.data:
            return int(city_res.data[0]["id"])
        # Fallback: if city jurisdictions aren't seeded for every state, use the state scope.
        return _get_state_id_by_code(db, code)

    # Regulation-category rows (e.g. "Renters", "Pet Policy", "ESA") use `city_name` + `state_code`
    # to decide whether the target jurisdiction is a city or a state.
    if not code:
        raise RuntimeError(
            f"Missing state_code for regulation row category='{category}', city_name='{label}'."
        )

    expected_state_name = CODE_TO_STATE_NAME.get(code)
    if expected_state_name and label.lower() == expected_state_name.lower():
        return _get_state_id_by_code(db, code)

    city_res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "city")
        .eq("name", label)
        .eq("state_code", code)
        .limit(1)
        .execute()
    )
    if city_res.data:
        return int(city_res.data[0]["id"])
    # Fallback: if city jurisdictions aren't seeded for this state, fall back to the state scope.
    return _get_state_id_by_code(db, code)


# ---------------------------------------------------------------------------
# CSV loader (used by settings page "Load regulations from CSV")
# ---------------------------------------------------------------------------


def load_regulations_from_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing seed file: {csv_path}")

    db = get_db()

    loaded = 0
    skipped = 0

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
                state_code = _infer_state_code(city_name)
            if not state_code:
                raise RuntimeError(
                    f"Missing/unknown state_code for row category='{category}', city_name='{city_name}'."
                )

            jurisdiction_id = _resolve_jurisdiction_id(
                db, category, city_name, state_code
            )

            # Your requirement: since we don't have a separate content column, store
            # only the law name in `content` and hash based on `content` only.
            content = law_name.strip()
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Idempotency: skip if *any* regulation already has this content_hash.
            existing_hash_res = (
                db.table("regulations")
                .select("id")
                .eq("content_hash", content_hash)
                .limit(1)
                .execute()
            )
            if existing_hash_res.data:
                skipped += 1
                continue

            payload: dict[str, Any] = {
                "jurisdiction_id": jurisdiction_id,
                "domain": "housing",
                "category": category or "General",
                "source_name": law_name or "Unknown",
                "url": url,
                "content": content,
                "content_hash": content_hash,
                "version": 1,
                "is_current": True,
                "effective_date": None,
            }

            db.table("regulations").insert([payload]).execute()
            loaded += 1

    return {"loaded": loaded, "skipped": skipped}


# ---------------------------------------------------------------------------
# Vector-index helpers (used by settings page)
# ---------------------------------------------------------------------------


def get_unindexed_regulations() -> list[dict[str, Any]]:
    db = get_db()

    regs_res = (
        db.table("regulations")
        .select("id,content,source_name,url,domain,category,jurisdiction_id")
        .eq("is_current", True)
        .execute()
    )
    reg_rows: list[dict[str, Any]] = regs_res.data or []
    if not reg_rows:
        return []

    reg_ids: list[int] = [int(r["id"]) for r in reg_rows if r.get("id") is not None]

    # Performance: only fetch embeddings for regulation IDs we're currently considering.
    embeddings_res = (
        db.table("regulation_embeddings")
        .select("regulation_id")
        .in_("regulation_id", reg_ids)
        .execute()
    )
    embedded_ids: set[int] = {
        int(row["regulation_id"]) for row in (embeddings_res.data or [])
    }

    docs: list[dict[str, Any]] = []
    for row in reg_rows:
        rid = int(row["id"])
        if rid in embedded_ids:
            continue
        docs.append(
            {
                "text": row.get("content") or "",
                "regulation_id": rid,
                "metadata": {
                    "source_name": row.get("source_name"),
                    "url": row.get("url"),
                    "domain": row.get("domain"),
                    "category": row.get("category"),
                    "jurisdiction_id": row.get("jurisdiction_id"),
                },
            }
        )
    return docs


def initialize_vector_index() -> dict[str, Any]:
    docs = get_unindexed_regulations()
    store = RegulationVectorStore()
    if docs:
        store.add_documents(docs)
    return {"indexed_docs": len(docs)}


def get_indexing_status() -> list[dict[str, Any]]:
    db = get_db()
    states_res = (
        db.table("jurisdictions")
        .select("id,name")
        .eq("type", "state")
        .order("name")
        .execute()
    )
    state_rows = states_res.data or []

    regs_res = (
        db.table("regulations")
        .select("id,jurisdiction_id")
        .eq("is_current", True)
        .execute()
    )
    regs = regs_res.data or []
    regs_by_state: dict[int, list[int]] = {}
    for r in regs:
        jid = int(r["jurisdiction_id"])
        regs_by_state.setdefault(jid, []).append(int(r["id"]))

    # Avoid scanning the whole embeddings table.
    all_reg_ids: list[int] = []
    for ids in regs_by_state.values():
        all_reg_ids.extend([int(i) for i in ids])

    if all_reg_ids:
        embeddings_res = (
            db.table("regulation_embeddings")
            .select("regulation_id")
            .in_("regulation_id", all_reg_ids)
            .execute()
        )
        indexed_ids: set[int] = {
            int(r["regulation_id"]) for r in (embeddings_res.data or [])
        }
    else:
        indexed_ids = set()

    status: list[dict[str, Any]] = []
    for row in state_rows:
        state_id = int(row["id"])
        regulation_ids = regs_by_state.get(state_id, [])
        reg_count = len(regulation_ids)
        indexed_count = sum(1 for rid in regulation_ids if rid in indexed_ids)
        status.append(
            {
                "jurisdiction": row.get("name"),
                "jurisdiction_id": state_id,
                "regulation_count": reg_count,
                "indexed_count": indexed_count,
                "last_indexed_date": None,
            }
        )
    return status


def is_supabase_connected() -> bool:
    try:
        db = get_db()
        db.table("jurisdictions").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# RegulationScraper — web scraping + DB upsert + vector indexing
# ---------------------------------------------------------------------------


class RegulationScraper:
    def scrape_source(
        self,
        url: str,
        source_name: str,
        jurisdiction_id: int,
        domain: str,
        category: str,
    ) -> Regulation | None:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code >= 400:
                logger.warning("HTTP %s for %s — skipping", resp.status_code, url)
                return None
        except Exception as exc:
            logger.warning("Unreachable URL %s: %s", url, exc)
            return None

        content_type = (resp.headers.get("content-type") or "").lower()
        is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type

        if is_pdf:
            try:
                reader = PdfReader(io.BytesIO(resp.content))
                pages = [(page.extract_text() or "").strip() for page in reader.pages]
                text = "\n".join(p for p in pages if p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed PDF extraction for %s: %s", url, exc)
                return None
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
        if len(text) < 120:
            logger.warning("Low-content page for %s (len=%s) — skipping", url, len(text))
            return None

        content_hash = _sha256(text)

        return Regulation(
            jurisdiction_id=int(jurisdiction_id),
            domain=domain or "housing",
            category=category or "General",
            source_name=source_name or "Unknown",
            url=url,
            content=text,
            content_hash=content_hash,
            version=1,
            is_current=True,
        )

    # -- provider-aware source loading --

    def _use_db_registry(self) -> bool:
        """Check if the DB source registry toggle is on (graceful fallback to False)."""
        try:
            from core.regulations.source_registry import source_registry
            return source_registry.is_db_registry_enabled()
        except Exception:
            return False

    def _get_source_rows_from_db_registry(
        self, jurisdiction_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load active sources from regulation_sources table."""
        try:
            from core.regulations.source_registry import source_repo
        except Exception:
            return []

        rows = source_repo.list_all(active_only=True)
        if jurisdiction_id is not None:
            rows = [r for r in rows if int(r.get("jurisdiction_id") or 0) == jurisdiction_id]
        return rows

    def _get_source_rows_from_regulations(
        self, jurisdiction_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Legacy path: load sources from the regulations table (is_current rows)."""
        db = get_db()
        query = (
            db.table("regulations")
            .select("id,url,source_name,jurisdiction_id,domain,category,content_hash,version")
            .eq("is_current", True)
        )
        if jurisdiction_id is not None:
            query = query.eq("jurisdiction_id", int(jurisdiction_id))
        return query.execute().data or []

    def _update_source_scrape_status(
        self, url: str, *, error: str | None = None,
    ) -> None:
        """Best-effort: update regulation_sources.last_scraped_at / last_error."""
        try:
            from core.regulations.source_registry import source_repo
            from datetime import datetime, timezone

            row = source_repo.get_by_url(url)
            if row:
                source_repo.update_scrape_status(
                    int(row["id"]),
                    last_scraped_at=datetime.now(timezone.utc),
                    last_error=error,
                )
        except Exception:
            pass

    def scrape_all_sources(self) -> list[Regulation]:
        if self._use_db_registry():
            rows = self._get_source_rows_from_db_registry()
        else:
            db = get_db()
            res = (
                db.table("regulations")
                .select("url,source_name,jurisdiction_id,domain,category")
                .eq("is_current", True)
                .execute()
            )
            rows = res.data or []

        results: list[Regulation] = []
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not url:
                continue
            reg = self.scrape_source(
                url=url,
                source_name=str(row.get("source_name") or ""),
                jurisdiction_id=int(row.get("jurisdiction_id") or 0),
                domain=str(row.get("domain") or "housing"),
                category=str(row.get("category") or "General"),
            )
            if reg is not None:
                results.append(reg)
        return results

    def scrape_and_index(
        self, jurisdiction_id: int | None = None
    ) -> dict[str, Any]:
        db = get_db()
        use_registry = self._use_db_registry()

        if use_registry:
            source_rows = self._get_source_rows_from_db_registry(jurisdiction_id)
        else:
            source_rows = self._get_source_rows_from_regulations(jurisdiction_id)

        scraped = 0
        indexed = 0
        errors: list[str] = []
        new_docs: list[dict[str, Any]] = []

        for row in source_rows:
            url = str(row.get("url") or "").strip()
            if not url:
                continue

            reg = self.scrape_source(
                url=url,
                source_name=str(row.get("source_name") or ""),
                jurisdiction_id=int(row.get("jurisdiction_id") or 0),
                domain=str(row.get("domain") or "housing"),
                category=str(row.get("category") or "General"),
            )
            if reg is None:
                errors.append(f"Failed to scrape {url}")
                self._update_source_scrape_status(url, error=f"Scrape failed")
                continue

            scraped += 1
            self._update_source_scrape_status(url, error=None)

            # Find the current regulation row for this URL to detect content changes.
            existing = (
                db.table("regulations")
                .select("id,content_hash,version")
                .eq("url", url)
                .eq("is_current", True)
                .limit(1)
                .execute()
            )
            existing_row = (existing.data or [None])[0]

            old_hash = str(existing_row.get("content_hash") or "") if existing_row else ""
            if reg.content_hash == old_hash:
                continue

            old_id = int(existing_row["id"]) if existing_row else None
            old_version = int(existing_row.get("version") or 0) if existing_row else 0

            if old_id is not None:
                db.table("regulations").update({"is_current": False}).eq(
                    "id", old_id
                ).execute()

            new_payload: dict[str, Any] = {
                "jurisdiction_id": reg.jurisdiction_id,
                "domain": reg.domain,
                "category": reg.category,
                "source_name": reg.source_name,
                "url": reg.url,
                "content": reg.content,
                "content_hash": reg.content_hash,
                "version": old_version + 1,
                "is_current": True,
                "effective_date": None,
            }
            ins = db.table("regulations").insert([new_payload]).execute()
            new_id: int | None = None
            if ins.data and ins.data[0].get("id") is not None:
                new_id = int(ins.data[0]["id"])
            else:
                lookup = (
                    db.table("regulations")
                    .select("id")
                    .eq("url", reg.url)
                    .eq("content_hash", reg.content_hash)
                    .eq("is_current", True)
                    .limit(1)
                    .execute()
                )
                if lookup.data:
                    new_id = int(lookup.data[0]["id"])

            if new_id is not None:
                new_docs.append(
                    {
                        "text": reg.content,
                        "regulation_id": new_id,
                        "metadata": {
                            "source_name": reg.source_name,
                            "url": reg.url,
                            "domain": reg.domain,
                            "category": reg.category,
                            "jurisdiction_id": reg.jurisdiction_id,
                        },
                    }
                )

        if new_docs:
            store = RegulationVectorStore()
            store.add_documents(new_docs)
            indexed = len(new_docs)

        return {"scraped": scraped, "indexed": indexed, "errors": errors}


# ---------------------------------------------------------------------------
# Backward-compatible ScraperService used by pages/5_settings.py
# ---------------------------------------------------------------------------


class ScraperService:
    def __init__(self) -> None:
        self._regulation_scraper = RegulationScraper()

    def load_regulations_from_csv(self, csv_path: Path) -> dict[str, Any]:
        return load_regulations_from_csv(csv_path)

    def initialize_vector_index(self) -> dict[str, Any]:
        return initialize_vector_index()

    def get_indexing_status(self) -> list[dict[str, Any]]:
        return get_indexing_status()

    def run_manual_scraper(self) -> dict[str, Any]:
        return self._regulation_scraper.scrape_and_index()

    def scrape_and_index(self, jurisdiction_id: int | None = None) -> dict[str, Any]:
        """Run scraping for all active sources (or a filtered jurisdiction) and index results."""
        return self._regulation_scraper.scrape_and_index(jurisdiction_id=jurisdiction_id)


scraper = ScraperService()
regulation_scraper = RegulationScraper()
