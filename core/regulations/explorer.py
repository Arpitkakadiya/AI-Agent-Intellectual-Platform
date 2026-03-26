from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.llm.client import llm
from core.rag.vector_store import RegulationVectorStore
from db.client import get_db


def get_state_jurisdiction_options() -> list[dict[str, Any]]:
    db = get_db()
    res = (
        db.table("jurisdictions")
        .select("id,name")
        .eq("type", "state")
        .order("name")
        .execute()
    )
    return [
        {"id": int(row["id"]), "name": str(row.get("name") or row["id"])}
        for row in (res.data or [])
    ]


def get_distinct_categories() -> list[str]:
    db = get_db()
    res = db.table("regulations").select("category").execute()
    cats = {str(row.get("category") or "").strip() for row in (res.data or [])}
    cats.discard("")
    return sorted(cats)


def get_explorer_metrics() -> dict[str, Any]:
    db = get_db()

    # Total regulations
    reg_res = db.table("regulations").select("id", count="exact").execute()
    total_regs = int(reg_res.count or 0)

    # States covered (distinct jurisdictions of type="state" that have at least one regulation)
    state_ids = [
        int(row["id"])
        for row in (
            db.table("jurisdictions")
            .select("id")
            .eq("type", "state")
            .execute()
            .data
            or []
        )
    ]
    if not state_ids:
        total_states_covered = 0
    else:
        regs_state_res = (
            db.table("regulations")
            .select("jurisdiction_id")
            .in_("jurisdiction_id", state_ids)
            .execute()
        )
        covered = {int(r["jurisdiction_id"]) for r in (regs_state_res.data or [])}
        total_states_covered = len(covered)

    # Last updated (use most recent regulations.created_at)
    latest_res = (
        db.table("regulations")
        .select("created_at")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    last_updated = None
    if latest_res.data:
        last_updated = latest_res.data[0].get("created_at")

    return {
        "total_regulations": total_regs,
        "total_states_covered": total_states_covered,
        "last_updated": last_updated,
    }


def search_regulations(
    query: str,
    jurisdiction_id: Optional[int],
    category: Optional[str],
    n_results: int = 10,
) -> list[dict[str, Any]]:
    if not llm.is_ai_available():
        return []

    store = RegulationVectorStore()
    raw_results = store.search(
        query=query,
        n_results=n_results,
        jurisdiction_id=jurisdiction_id,
        category_filter=category,
    )

    # `category_filter` is already applied inside the RPC for efficiency,
    # but we keep the safety filter here as well.
    filtered: list[dict[str, Any]] = []
    for r in raw_results:
        meta = r.metadata or {}
        res_category = meta.get("category")
        if category and str(res_category or "").strip() != str(category).strip():
            continue

        filtered.append(
            {
                "source_name": meta.get("source_name") or "",
                "domain": meta.get("domain") or "",
                "category": meta.get("category") or "",
                "url": meta.get("url") or "",
                "last_checked": meta.get("created_at"),
            }
        )

    return filtered


def to_results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=["source_name", "domain", "category", "url", "last_checked"]
    )
    return df

