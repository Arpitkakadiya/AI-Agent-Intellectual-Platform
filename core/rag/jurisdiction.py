"""Jurisdiction hierarchy helpers for strict retrieval scoping.

Provides deterministic resolution of which jurisdiction IDs should be
searched for a given query, respecting the hierarchy:

    federal → state → county → city

Key behaviours:
- Exact jurisdiction match is always preferred.
- Parent fallback (city → state → federal) is explicit and labelled.
- Cross-jurisdiction retrieval is opt-in (comparison questions).
- Results carry a ``scope_label`` so the grounding layer can tell the user
  where each chunk came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from db.client import get_db

_HIERARCHY_ORDER = {"federal": 0, "state": 1, "county": 2, "city": 3}


@dataclass
class ScopedJurisdiction:
    """A jurisdiction ID annotated with its role in the retrieval plan."""
    jurisdiction_id: int
    name: str
    type: str  # federal | state | county | city
    role: str  # "exact" | "parent_fallback" | "cross_jurisdiction"
    hierarchy_depth: int = 0

    @property
    def scope_label(self) -> str:
        if self.role == "exact":
            return f"{self.name} ({self.type})"
        if self.role == "parent_fallback":
            return f"{self.name} ({self.type}, fallback)"
        return f"{self.name} ({self.type}, comparison)"


def _lookup_jurisdiction(db: Any, jid: int) -> dict[str, Any] | None:
    res = (
        db.table("jurisdictions")
        .select("id,type,name,parent_id,state_code")
        .eq("id", jid)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def _get_federal_id(db: Any) -> int | None:
    res = (
        db.table("jurisdictions")
        .select("id,name")
        .eq("type", "federal")
        .limit(1)
        .execute()
    )
    return int(res.data[0]["id"]) if res.data else None


def resolve_hierarchy(
    jurisdiction_id: int | None,
    *,
    include_parents: bool = True,
    include_federal: bool = True,
) -> list[ScopedJurisdiction]:
    """Build an ordered list of jurisdictions to search, from most specific
    to broadest.  Each entry is labelled with its ``role``."""
    if jurisdiction_id is None:
        return []

    db = get_db()
    result: list[ScopedJurisdiction] = []
    seen: set[int] = set()

    row = _lookup_jurisdiction(db, jurisdiction_id)
    if not row:
        return []

    result.append(
        ScopedJurisdiction(
            jurisdiction_id=int(row["id"]),
            name=row["name"],
            type=row["type"],
            role="exact",
            hierarchy_depth=_HIERARCHY_ORDER.get(row["type"], 99),
        )
    )
    seen.add(int(row["id"]))

    if include_parents:
        current = row
        while current and current.get("parent_id"):
            parent = _lookup_jurisdiction(db, int(current["parent_id"]))
            if not parent or int(parent["id"]) in seen:
                break
            seen.add(int(parent["id"]))
            result.append(
                ScopedJurisdiction(
                    jurisdiction_id=int(parent["id"]),
                    name=parent["name"],
                    type=parent["type"],
                    role="parent_fallback",
                    hierarchy_depth=_HIERARCHY_ORDER.get(parent["type"], 99),
                )
            )
            current = parent

    if include_federal:
        fed_id = _get_federal_id(db)
        if fed_id is not None and fed_id not in seen:
            result.append(
                ScopedJurisdiction(
                    jurisdiction_id=fed_id,
                    name="Federal",
                    type="federal",
                    role="parent_fallback",
                    hierarchy_depth=0,
                )
            )

    return result


def build_retrieval_plan(
    question: str,
    sidebar_jurisdiction_id: int | None,
    mentioned_jurisdiction_ids: list[int],
    *,
    is_cross_jurisdiction: bool = False,
) -> list[ScopedJurisdiction]:
    """High-level retrieval planner.

    For single-jurisdiction queries: returns exact + parent chain.
    For cross-jurisdiction queries: returns all mentioned jurisdictions
    labelled ``cross_jurisdiction``, plus federal.
    """
    if is_cross_jurisdiction:
        db = get_db()
        plan: list[ScopedJurisdiction] = []
        seen: set[int] = set()

        for jid in mentioned_jurisdiction_ids:
            if jid in seen:
                continue
            seen.add(jid)
            row = _lookup_jurisdiction(db, jid)
            if row:
                plan.append(
                    ScopedJurisdiction(
                        jurisdiction_id=int(row["id"]),
                        name=row["name"],
                        type=row["type"],
                        role="cross_jurisdiction",
                        hierarchy_depth=_HIERARCHY_ORDER.get(row["type"], 99),
                    )
                )

        if sidebar_jurisdiction_id and sidebar_jurisdiction_id not in seen:
            seen.add(sidebar_jurisdiction_id)
            row = _lookup_jurisdiction(db, sidebar_jurisdiction_id)
            if row:
                plan.append(
                    ScopedJurisdiction(
                        jurisdiction_id=int(row["id"]),
                        name=row["name"],
                        type=row["type"],
                        role="cross_jurisdiction",
                        hierarchy_depth=_HIERARCHY_ORDER.get(row["type"], 99),
                    )
                )

        fed_id = _get_federal_id(db)
        if fed_id and fed_id not in seen:
            plan.append(
                ScopedJurisdiction(
                    jurisdiction_id=fed_id,
                    name="Federal",
                    type="federal",
                    role="parent_fallback",
                    hierarchy_depth=0,
                )
            )

        return plan

    primary_jid = sidebar_jurisdiction_id
    if not primary_jid and mentioned_jurisdiction_ids:
        primary_jid = mentioned_jurisdiction_ids[0]

    return resolve_hierarchy(primary_jid, include_parents=True, include_federal=True)


def detect_jurisdiction_conflicts(
    results: list[dict[str, Any]],
) -> list[str]:
    """Return human-readable conflict notices when chunks from different
    jurisdictions contain contradictory signals (simple heuristic)."""
    by_jurisdiction: dict[str, list[str]] = {}
    for r in results:
        meta = r.get("metadata") or {}
        jname = meta.get("jurisdiction_name") or meta.get("source_name") or "Unknown"
        text = (r.get("document") or "")[:500].lower()
        by_jurisdiction.setdefault(jname, []).append(text)

    conflict_phrases = [
        ("prohibited", "permitted"),
        ("shall not", "may"),
        ("no fee", "fee required"),
        ("exempt", "subject to"),
    ]
    notices: list[str] = []
    jurisdictions = list(by_jurisdiction.keys())
    for i, j1 in enumerate(jurisdictions):
        for j2 in jurisdictions[i + 1 :]:
            texts_1 = " ".join(by_jurisdiction[j1])
            texts_2 = " ".join(by_jurisdiction[j2])
            for a, b in conflict_phrases:
                if (a in texts_1 and b in texts_2) or (b in texts_1 and a in texts_2):
                    notices.append(
                        f"Potential conflict between {j1} and {j2}: "
                        f"one source uses '{a}' while the other uses '{b}'."
                    )
                    break
    return notices
