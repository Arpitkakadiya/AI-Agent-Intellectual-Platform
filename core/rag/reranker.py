"""Deterministic reranker for compliance RAG results.

Runs after initial retrieval (high-recall) and before final context
assembly (high-precision).  Scoring is a weighted sum of:

- **jurisdiction_match**: exact match to the requested jurisdiction
- **recency**: chunks with effective dates or newer version numbers
- **topic_relevance**: keyword overlap with the query
- **source_quality**: primary/official sources score higher
- **citation_density**: chunks that cite statutes, section numbers, etc.

An optional LLM-assisted reranker can be enabled via
``RAG_LLM_RERANK_ENABLED=true`` in the environment.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Scoring weights (deterministic reranker)
_W_JURISDICTION = 0.30
_W_TOPIC = 0.25
_W_CITATION = 0.20
_W_SOURCE_QUALITY = 0.15
_W_RECENCY = 0.10

_OFFICIAL_DOMAIN_PATTERNS = (
    r"\.gov\b",
    r"\.us\b",
    r"hud\.gov",
    r"legislature",
    r"leginfo",
    r"statutes",
    r"code\.org",
)

_CITATION_RE = re.compile(
    r"§\s*[\d.]+|"
    r"\b(?:Sec(?:tion)?|SEC(?:TION)?)\s*\.?\s*[\d.\-]+|"
    r"\b\d+\s+(?:U\.?S\.?C\.?|C\.?F\.?R\.?)\s*§?\s*\d+|"
    r"\b(?:Public Law|P\.?L\.?)\s+\d+[\-–]\d+",
    re.IGNORECASE,
)


def _jurisdiction_score(
    result: dict[str, Any],
    target_jurisdiction_ids: list[int],
    exact_jurisdiction_id: int | None,
) -> float:
    meta = result.get("metadata") or {}
    jid = meta.get("jurisdiction_id")
    if jid is None:
        return 0.0
    jid = int(jid)
    if exact_jurisdiction_id is not None and jid == exact_jurisdiction_id:
        return 1.0
    if jid in target_jurisdiction_ids:
        return 0.6
    return 0.0


def _topic_score(result: dict[str, Any], query_tokens: set[str]) -> float:
    doc = (result.get("document") or "").lower()
    meta = result.get("metadata") or {}
    source = (meta.get("source_name") or "").lower()
    combined = f"{doc} {source}"
    combined_tokens = set(re.findall(r"\w+", combined))
    if not query_tokens:
        return 0.0
    return len(query_tokens & combined_tokens) / len(query_tokens)


def _citation_score(result: dict[str, Any]) -> float:
    doc = result.get("document") or ""
    citations = _CITATION_RE.findall(doc)
    if len(citations) >= 3:
        return 1.0
    if len(citations) >= 1:
        return 0.6
    return 0.0


def _source_quality_score(result: dict[str, Any]) -> float:
    meta = result.get("metadata") or {}
    url = (meta.get("url") or "").lower()
    for pat in _OFFICIAL_DOMAIN_PATTERNS:
        if re.search(pat, url):
            return 1.0
    return 0.3


def _recency_score(result: dict[str, Any]) -> float:
    doc = (result.get("document") or "").lower()
    year_matches = re.findall(r"\b(20[12]\d)\b", doc)
    if not year_matches:
        return 0.0
    latest = max(int(y) for y in year_matches)
    if latest >= 2024:
        return 1.0
    if latest >= 2020:
        return 0.6
    return 0.3


def rerank_deterministic(
    results: list[dict[str, Any]],
    query: str,
    target_jurisdiction_ids: list[int] | None = None,
    exact_jurisdiction_id: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score and reorder results using deterministic heuristics.

    Returns the top ``top_k`` results (or all if ``top_k`` is None),
    each annotated with a ``rerank_score`` field.
    """
    target_jids = target_jurisdiction_ids or []
    query_tokens = set(re.findall(r"\w+", query.lower()))

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, r in enumerate(results):
        s = (
            _W_JURISDICTION * _jurisdiction_score(r, target_jids, exact_jurisdiction_id)
            + _W_TOPIC * _topic_score(r, query_tokens)
            + _W_CITATION * _citation_score(r)
            + _W_SOURCE_QUALITY * _source_quality_score(r)
            + _W_RECENCY * _recency_score(r)
        )
        scored.append((s, idx, r))

    scored.sort(key=lambda x: (-x[0], x[1]))

    k = top_k or len(scored)
    out: list[dict[str, Any]] = []
    for score, _idx, r in scored[:k]:
        entry = dict(r)
        entry["rerank_score"] = round(score, 4)
        out.append(entry)
    return out


def rerank_llm(
    results: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """LLM-assisted reranker (opt-in via RAG_LLM_RERANK_ENABLED).

    Asks the LLM to pick the most relevant chunks for the query.
    Falls back to deterministic reranking on any failure.
    """
    from core.llm.client import llm

    if not llm.is_ai_available():
        return rerank_deterministic(results, query, top_k=top_k)

    numbered = []
    for i, r in enumerate(results[:20]):
        snippet = (r.get("document") or "")[:300]
        meta = r.get("metadata") or {}
        src = meta.get("source_name") or ""
        numbered.append(f"[{i}] ({src}) {snippet}")

    prompt = (
        "You are a legal compliance retrieval assistant.\n"
        "Given the user query and numbered document snippets below, "
        f"return the indices of the {top_k} most relevant snippets "
        "for answering the query, in order of relevance.\n"
        "Return ONLY a JSON array of integers, e.g. [3, 0, 7, 1, 5].\n\n"
        f"Query: {query}\n\n"
        "Snippets:\n" + "\n".join(numbered)
    )

    try:
        raw = llm.ask(system="You are a retrieval reranker.", user=prompt, max_tokens=200)
        indices = _parse_index_list(raw, len(results))
        reranked = [results[i] for i in indices if i < len(results)]
        if not reranked:
            raise ValueError("Empty rerank result")
        return reranked[:top_k]
    except Exception:
        logger.warning("LLM rerank failed, falling back to deterministic")
        return rerank_deterministic(results, query, top_k=top_k)


def _parse_index_list(raw: str, max_idx: int) -> list[int]:
    import json
    text = raw.strip()
    match = re.search(r"\[[\d,\s]+\]", text)
    if match:
        arr = json.loads(match.group())
        return [int(x) for x in arr if 0 <= int(x) < max_idx]
    nums = re.findall(r"\d+", text)
    return [int(n) for n in nums if int(n) < max_idx]


def rerank(
    results: list[dict[str, Any]],
    query: str,
    target_jurisdiction_ids: list[int] | None = None,
    exact_jurisdiction_id: int | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Main entry point: uses LLM reranker if enabled, else deterministic."""
    use_llm = getattr(settings, "RAG_LLM_RERANK_ENABLED", False)
    k = top_k or getattr(settings, "RAG_RERANK_TOP_K", 5)

    if use_llm:
        return rerank_llm(results, query, top_k=k)

    return rerank_deterministic(
        results,
        query,
        target_jurisdiction_ids=target_jurisdiction_ids,
        exact_jurisdiction_id=exact_jurisdiction_id,
        top_k=k,
    )
