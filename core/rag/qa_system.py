from __future__ import annotations

import re
from typing import Any

from core.llm.client import llm
from core.llm.prompts import QA_SYSTEM_PROMPT
from core.rag.utils import deduplicate_sources
from core.rag.vector_store import RegulationVectorStore, SearchResult
from db.client import get_db

_MAX_CONTEXT_RESULTS = 5
_MAX_CONTEXT_CROSS_JURISDICTION = 8
_MAX_HISTORY_ITEMS = 6
_SEARCH_CANDIDATES = 5
_MIN_INFORMATIVE_CHARS = 220
_MAX_CHUNKS_PER_SOURCE = 2

# US states + DC for multi-state / comparison detection (lowercase tokens).
_US_STATE_NAMES: frozenset[str] = frozenset(
    {
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new york",
        "north carolina",
        "north dakota",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "rhode island",
        "south carolina",
        "south dakota",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "west virginia",
        "wisconsin",
        "wyoming",
        "district of columbia",
    }
)
_US_STATE_ABBREVS: frozenset[str] = frozenset(
    {
        "al",
        "ak",
        "az",
        "ar",
        "ca",
        "co",
        "ct",
        "de",
        "fl",
        "ga",
        "hi",
        "id",
        "il",
        "in",
        "ia",
        "ks",
        "ky",
        "la",
        "me",
        "md",
        "ma",
        "mi",
        "mn",
        "ms",
        "mo",
        "mt",
        "ne",
        "nv",
        "nh",
        "nj",
        "nm",
        "ny",
        "nc",
        "nd",
        "oh",
        "ok",
        "or",
        "pa",
        "ri",
        "sc",
        "sd",
        "tn",
        "tx",
        "ut",
        "vt",
        "va",
        "wa",
        "wv",
        "wi",
        "wy",
        "dc",
    }
)

# Improves embedding match: corpora say "assistance animal" more often than "ESA".
_RETRIEVAL_ESA_HINT = (
    "emotional support animal assistance animal reasonable accommodation "
    "Fair Housing Act HUD"
)

_CONTEXT_LEAK_RE = re.compile(r"^From\s+.*\(.*\).*$", re.MULTILINE)
_NOTE_SUFFIX_RE = re.compile(r"\[Note:.*$", re.DOTALL)

_SCOPE_KEYWORDS: frozenset[str] = frozenset(
    {
        "lease",
        "rent",
        "renter",
        "tenant",
        "landlord",
        "housing",
        "evict",
        "eviction",
        "security deposit",
        "deposit",
        "repairs",
        "habitability",
        "fair housing",
        "hud",
        "esa",
        "emotional support",
        "assistance animal",
        "service animal",
        "pet",
        "rent control",
        "rent stabilization",
        "insurance",
        "homeowners",
    }
)


def _is_in_scope_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(k in q for k in _SCOPE_KEYWORDS)


def _out_of_scope_answer() -> str:
    return (
        "I'm sorry, but your question doesn't seem to be related to housing regulations, "
        "leasing, compliance, or tenant/landlord law. I'm specialized in helping with:\n\n"
        "- Housing and leasing regulations\n"
        "- Tenant rights and landlord obligations (repairs, deposits, habitability)\n"
        "- ESA / service animal rules and accommodations\n"
        "- Rent control and renters protections\n"
        "- City/state-specific regulations and compliance checks\n\n"
        "Please ask a question related to these topics and I can assist you."
    )


def _states_mentioned(question: str) -> list[str]:
    """Return canonical state tokens found in the question (names or abbreviations)."""
    ql = (question or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for name in _US_STATE_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", ql):
            key = name
            if key not in seen:
                seen.add(key)
                found.append(name.title())
    for ab in _US_STATE_ABBREVS:
        if re.search(rf"\b{re.escape(ab)}\b", ql):
            key = f"abbr:{ab}"
            if key not in seen:
                seen.add(key)
                found.append(ab.upper())
    return found


def _needs_cross_jurisdiction_retrieval(question: str) -> bool:
    """
    True when the sidebar jurisdiction filter would hide relevant materials
    (e.g. comparing CA vs TX while only Texas is selected).
    """
    ql = (question or "").lower()
    states = _states_mentioned(question)
    unique_states = {s.lower() for s in states}
    if len(unique_states) >= 2:
        return True

    broad_phrases = (
        "all states",
        "every state",
        "nationwide",
        "state by state",
        "cross state",
        "between states",
        "different states",
        "multiple states",
        "which state",
        "which states",
    )
    if any(p in ql for p in broad_phrases):
        return True

    compare_markers = (
        "compare",
        "comparison",
        "versus",
        " vs ",
        " vs.",
        "stricter",
        "strictest",
        "more strict",
        "more lenient",
        "tougher",
        "harsher",
        "better for tenants",
        "worse for tenants",
        "difference between",
        "differences between",
    )
    if any(m in ql for m in compare_markers):
        return True

    return False


def _retrieval_jurisdiction_ids(
    question: str, sidebar_jurisdiction_id: int | None
) -> list[int]:
    """
    Jurisdiction DB ids to search separately (OR semantics via merge).
    Includes federal, every state mentioned in the question, and the sidebar selection.
    """
    db = get_db()
    ids: list[int] = []

    fed = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "federal")
        .eq("name", "Federal Government")
        .limit(1)
        .execute()
    )
    if fed.data:
        ids.append(int(fed.data[0]["id"]))

    for token in _states_mentioned(question):
        if len(token) == 2:
            res = (
                db.table("jurisdictions")
                .select("id")
                .eq("type", "state")
                .eq("state_code", token)
                .limit(1)
                .execute()
            )
        else:
            res = (
                db.table("jurisdictions")
                .select("id")
                .eq("type", "state")
                .eq("name", token)
                .limit(1)
                .execute()
            )
        if res.data:
            ids.append(int(res.data[0]["id"]))

    if sidebar_jurisdiction_id is not None:
        ids.append(int(sidebar_jurisdiction_id))

    out: list[int] = []
    seen: set[int] = set()
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _merge_vector_hits(hits: list[SearchResult]) -> list[dict[str, Any]]:
    by_row: dict[int, SearchResult] = {}
    seen_fp: set[str] = set()
    extras: list[SearchResult] = []

    for r in hits:
        rid = r.row_id
        if rid is not None:
            prev = by_row.get(rid)
            if prev is None or r.score > prev.score:
                by_row[rid] = r
        else:
            fp = f"{(r.metadata or {}).get('source_name', '')}|{(r.document or '')[:200]}"
            if fp not in seen_fp:
                seen_fp.add(fp)
                extras.append(r)

    merged = list(by_row.values()) + extras
    merged.sort(key=lambda x: x.score, reverse=True)
    return [
        {"document": r.document, "metadata": r.metadata, "score": r.score}
        for r in merged
    ]


def _diversify_by_source(
    results: list[dict[str, Any]], max_items: int
) -> list[dict[str, Any]]:
    """Prefer multiple sources so comparisons are not five chunks from one URL."""
    by_score = sorted(
        results, key=lambda r: float(r.get("score") or 0.0), reverse=True
    )
    picked: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    for r in by_score:
        meta = r.get("metadata") or {}
        src = str(meta.get("source_name") or meta.get("url") or "")
        n = per_source.get(src, 0)
        if n >= _MAX_CHUNKS_PER_SOURCE:
            continue
        per_source[src] = n + 1
        picked.append(r)
        if len(picked) >= max_items:
            break
    if len(picked) < max_items:
        for r in by_score:
            if r in picked:
                continue
            picked.append(r)
            if len(picked) >= max_items:
                break
    return picked


def _build_context(results: list[dict[str, Any]], max_blocks: int) -> str:
    blocks: list[str] = []
    for r in results[:max_blocks]:
        meta = r.get("metadata") or {}
        header = meta.get("source_name") or meta.get("url") or "Source"
        blocks.append(f"[{header}]\n{r['document']}")
    return "\n---\n".join(blocks)


def _build_history(chat_history: list[dict[str, Any]]) -> str:
    recent = chat_history[-_MAX_HISTORY_ITEMS:]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _clean_answer(text: str) -> str:
    cleaned = _CONTEXT_LEAK_RE.sub("", text)
    cleaned = _NOTE_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip()


def _retrieval_query(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()
    parts = [q]
    if "esa" in ql or "emotional support" in ql:
        parts.append(_RETRIEVAL_ESA_HINT)
    if _needs_cross_jurisdiction_retrieval(q):
        mentioned = _states_mentioned(q)
        if mentioned:
            parts.append(" ".join(mentioned))
        parts.append("state law federal law comparison")
    return "\n".join(parts)


def _extract_sources(
    results: list[dict[str, Any]], max_items: int
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for r in results[:max_items]:
        meta = r.get("metadata") or {}
        raw.append(
            {
                "source": meta.get("source_name", ""),
                "url": meta.get("url", ""),
                "category": meta.get("category", ""),
                "domain": meta.get("domain", ""),
            }
        )
    return deduplicate_sources(raw)


def _is_informative_chunk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < _MIN_INFORMATIVE_CHARS:
        return False
    # Skip noisy title/url-only chunks that can outrank useful legal text.
    if t.count("http") >= 1 and "\n" not in t and "." not in t[:120]:
        return False
    return True


class QASystem:
    def __init__(self) -> None:
        self._store = RegulationVectorStore()

    def answer_question(
        self,
        question: str,
        chat_history: list[dict[str, Any]],
        jurisdiction_id: int | None = None,
    ) -> dict[str, Any]:
        if not _is_in_scope_question(question):
            return {"answer": _out_of_scope_answer(), "sources": []}

        # In `rule_based` mode we should still return a helpful message even if
        # vector search is unavailable (e.g., missing/invalid Supabase config).
        cross = _needs_cross_jurisdiction_retrieval(question)
        max_context = _MAX_CONTEXT_CROSS_JURISDICTION if cross else _MAX_CONTEXT_RESULTS
        q_text = _retrieval_query(question)
        n_per_jurisdiction = 5

        try:
            if cross:
                jids = _retrieval_jurisdiction_ids(question, jurisdiction_id)
                q_emb = llm.embed(q_text)
                raw_hits: list[SearchResult] = []
                for jid in jids:
                    raw_hits.extend(
                        self._store.search(
                            q_text,
                            n_results=n_per_jurisdiction,
                            jurisdiction_id=jid,
                            query_embedding=q_emb,
                        )
                    )
                result_dicts = _merge_vector_hits(raw_hits)
            else:
                search_results = self._store.search(
                    query=q_text,
                    n_results=_SEARCH_CANDIDATES,
                    jurisdiction_id=jurisdiction_id,
                )
                result_dicts = [
                    {"document": r.document, "metadata": r.metadata, "score": r.score}
                    for r in search_results
                ]
        except Exception:
            if llm.is_ai_available():
                raise
        informative_results = [
            r for r in result_dicts if _is_informative_chunk(str(r.get("document") or ""))
        ]
        pool = informative_results or result_dicts
        selected_results = (
            _diversify_by_source(pool, max_context)
            if cross
            else pool[:max_context]
        )
        sources = _extract_sources(selected_results, max_context)

        if not llm.is_ai_available():
            return {
                "answer": (
                    "AI-powered answers are currently unavailable because no LLM API key "
                    "is configured. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
                    "GOOGLE_API_KEY in your environment to enable AI responses.\n\n"
                    "In the meantime, here are the most relevant regulation sources I found."
                ),
                "sources": sources,
            }

        context = _build_context(selected_results, max_context)
        history = _build_history(chat_history)

        compare_note = ""
        if cross:
            compare_note = (
                "\nNote: This question spans multiple jurisdictions or asks for a "
                "comparison. Use all relevant excerpts below; if only federal rules "
                "appear, explain how they apply broadly and what state-specific text "
                "is or is not present.\n"
            )

        user_message = (
            f"Regulation context:\n{context}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"{compare_note}"
            f"Question: {question}"
        )

        raw_answer = llm.ask(system=QA_SYSTEM_PROMPT, user=user_message)
        answer = _clean_answer(raw_answer)

        return {"answer": answer, "sources": sources}


qa = QASystem()
