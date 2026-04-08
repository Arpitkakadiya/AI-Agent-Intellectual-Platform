"""Hybrid retrieval: vector + lexical search with configurable fusion.

Provides three public functions:
- ``vector_search(...)``  — existing pgvector cosine similarity
- ``keyword_search(...)`` — Postgres full-text search (ts_rank) via RPC,
                            with a Python fallback when the DB function is
                            unavailable
- ``hybrid_search(...)``  — merges both candidate pools using Reciprocal
                            Rank Fusion (RRF)

The weighting between vector and keyword results is controlled by
``RAG_HYBRID_VECTOR_WEIGHT`` (default 0.6) in config.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from config import settings

from core.rag.vector_store import RegulationVectorStore, SearchResult

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


# ---------------------------------------------------------------------------
# Vector search (thin wrapper around existing store)
# ---------------------------------------------------------------------------

def vector_search(
    store: RegulationVectorStore,
    query: str,
    n_results: int,
    jurisdiction_ids: list[int] | None = None,
    category_filter: str | None = None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Run vector search across one or more jurisdictions and merge."""
    from core.llm.client import llm

    qemb = query_embedding or llm.embed(query)
    all_hits: list[SearchResult] = []

    if jurisdiction_ids:
        for jid in jurisdiction_ids:
            hits = store.search(
                query=query,
                n_results=n_results,
                jurisdiction_id=jid,
                query_embedding=qemb,
                category_filter=category_filter,
            )
            all_hits.extend(hits)
    else:
        all_hits = store.search(
            query=query,
            n_results=n_results,
            jurisdiction_id=None,
            query_embedding=qemb,
            category_filter=category_filter,
        )

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for h in sorted(all_hits, key=lambda x: x.score, reverse=True):
        fp = f"{h.row_id}|{(h.document or '')[:200]}"
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(
            {"document": h.document, "metadata": h.metadata, "score": h.score}
        )

    logger.debug("Vector search returned %d results (deduped from %d)", len(deduped), len(all_hits))
    return deduped


# ---------------------------------------------------------------------------
# Keyword / lexical search
# ---------------------------------------------------------------------------

def _build_tsquery(query: str) -> str:
    """Convert a natural-language query into a Postgres tsquery string.
    Keeps quoted phrases as phrase queries, ORs remaining tokens."""
    tokens = re.findall(r'"[^"]+"|\S+', query)
    parts: list[str] = []
    for tok in tokens:
        tok = tok.strip('"').strip()
        if not tok:
            continue
        cleaned = re.sub(r"[^\w\s]", "", tok)
        if not cleaned:
            continue
        words = cleaned.split()
        if len(words) > 1:
            parts.append(" <-> ".join(words))
        else:
            parts.append(cleaned)
    return " | ".join(parts) if parts else query


def keyword_search(
    query: str,
    n_results: int = 10,
    jurisdiction_ids: list[int] | None = None,
    category_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Full-text keyword search via Postgres ``match_regulations_lexical``
    RPC.  Falls back to a Python-side substring scorer if the RPC is not
    deployed yet."""
    from db.client import get_db

    db = get_db()
    tsq = _build_tsquery(query)

    jids = jurisdiction_ids or []

    try:
        payload: dict[str, Any] = {
            "search_query": tsq,
            "match_count": n_results,
            "filter_jurisdictions": jids if jids else None,
            "category_filter": category_filter,
        }
        res = db.rpc("match_regulations_lexical", payload).execute()
        rows = res.data or []
        logger.debug("Lexical RPC returned %d results", len(rows))
        return [
            {
                "document": r.get("chunk_text") or "",
                "metadata": r.get("metadata") or {},
                "score": float(r.get("rank") or 0.0),
            }
            for r in rows
        ]
    except Exception:
        logger.debug("Lexical RPC unavailable, using Python fallback")
        return _python_keyword_fallback(query, n_results, jids, category_filter)


def _python_keyword_fallback(
    query: str,
    n_results: int,
    jurisdiction_ids: list[int],
    category_filter: str | None,
) -> list[dict[str, Any]]:
    """Lightweight Python-side keyword scoring when the Postgres lexical
    RPC hasn't been deployed.  Pulls candidate chunks from vector table
    and scores them by token overlap."""
    from db.client import get_db

    db = get_db()

    q = db.table("regulation_embeddings").select(
        "id,regulation_id,chunk_text"
    )
    res = q.limit(500).execute()
    rows = res.data or []

    if not rows:
        return []

    reg_ids = list({int(r["regulation_id"]) for r in rows})
    reg_res = (
        db.table("regulations")
        .select("id,jurisdiction_id,category,source_name,url,domain")
        .in_("id", reg_ids[:500])
        .execute()
    )
    reg_map: dict[int, dict[str, Any]] = {
        int(r["id"]): r for r in (reg_res.data or [])
    }

    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored: list[dict[str, Any]] = []

    for row in rows:
        rid = int(row["regulation_id"])
        reg = reg_map.get(rid, {})

        if jurisdiction_ids:
            if int(reg.get("jurisdiction_id") or 0) not in jurisdiction_ids:
                continue
        if category_filter and reg.get("category") != category_filter:
            continue

        chunk = row.get("chunk_text") or ""
        chunk_tokens = set(re.findall(r"\w+", chunk.lower()))
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue

        score = overlap / max(len(query_tokens), 1)
        scored.append(
            {
                "document": chunk,
                "metadata": {
                    "source_name": reg.get("source_name", ""),
                    "url": reg.get("url", ""),
                    "category": reg.get("category", ""),
                    "domain": reg.get("domain", ""),
                    "jurisdiction_id": reg.get("jurisdiction_id"),
                },
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("Python keyword fallback returned %d results", min(n_results, len(scored)))
    return scored[:n_results]


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------

def hybrid_search(
    store: RegulationVectorStore,
    query: str,
    n_results: int = 10,
    jurisdiction_ids: list[int] | None = None,
    category_filter: str | None = None,
    query_embedding: list[float] | None = None,
    vector_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of vector + keyword results.

    ``vector_weight`` controls the balance (0.0 = keyword only, 1.0 = vector
    only).  Defaults to ``RAG_HYBRID_VECTOR_WEIGHT`` from config.
    """
    vw = vector_weight if vector_weight is not None else getattr(
        settings, "RAG_HYBRID_VECTOR_WEIGHT", 0.6
    )
    kw = 1.0 - vw

    fetch_n = max(n_results * 2, 15)

    vec_results = vector_search(
        store, query, fetch_n, jurisdiction_ids, category_filter, query_embedding
    )
    kw_results = keyword_search(query, fetch_n, jurisdiction_ids, category_filter)

    logger.debug(
        "Hybrid fusion: %d vector + %d keyword candidates (vw=%.2f, kw=%.2f)",
        len(vec_results), len(kw_results), vw, kw,
    )

    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    def _fingerprint(r: dict[str, Any]) -> str:
        meta = r.get("metadata") or {}
        return f"{meta.get('source_name', '')}|{(r.get('document') or '')[:200]}"

    for rank, r in enumerate(vec_results):
        fp = _fingerprint(r)
        scores[fp] = scores.get(fp, 0.0) + vw * _rrf_score(rank + 1)
        if fp not in docs:
            docs[fp] = r

    for rank, r in enumerate(kw_results):
        fp = _fingerprint(r)
        scores[fp] = scores.get(fp, 0.0) + kw * _rrf_score(rank + 1)
        if fp not in docs:
            docs[fp] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[dict[str, Any]] = []
    for fp, score in ranked[:n_results]:
        entry = dict(docs[fp])
        entry["hybrid_score"] = score
        entry["score"] = score
        out.append(entry)

    logger.debug("Hybrid search returning %d fused results", len(out))
    return out
