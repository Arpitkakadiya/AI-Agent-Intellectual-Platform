"""End-to-end RAG QA orchestrator.

Follows the upgraded pipeline:
1. Resolve effective question (follow-up handling)
2. Validate domain scope
3. Build jurisdiction-aware retrieval plan
4. Run hybrid retrieval (vector + lexical + RRF) when enabled
5. Fall back to vector-only when hybrid is disabled or returns too little
6. Merge and normalize candidate results
7. Run deterministic reranking
8. Build grounded context from top reranked chunks
9. Assess confidence and detect evidence weakness/conflicts
10. Generate final grounded answer with clear source attribution
11. Return structured result
"""
from __future__ import annotations

import logging
import re
from typing import Any

from config import settings

from core.llm.client import llm
from core.llm.prompts import QA_SYSTEM_PROMPT
from core.rag.grounding import (
    assess_confidence,
    build_grounded_answer,
    build_grounded_context,
    extract_sources,
)
from core.rag.hybrid import hybrid_search, vector_search
from core.rag.jurisdiction import (
    ScopedJurisdiction,
    build_retrieval_plan,
)
from core.rag.reranker import rerank
from core.rag.vector_store import RegulationVectorStore
from db.client import get_db

logger = logging.getLogger(__name__)

_MAX_HISTORY_ITEMS = 6

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

# ---------------------------------------------------------------------------
# Helper functions (domain scope, follow-ups, state detection)
# ---------------------------------------------------------------------------


def _is_in_scope_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    return any(k in q for k in _SCOPE_KEYWORDS)


def _is_followup_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    followup_markers = (
        "what about",
        "how about",
        "same for",
        "and for",
        "for ",
        "in ",
        "there",
        "that one",
        "this one",
        "those",
        "these",
        "it",
        "them",
    )
    if len(q.split()) <= 7:
        return True
    return any(m in q for m in followup_markers)


def _latest_user_turn(chat_history: list[dict[str, Any]], current_q: str) -> str:
    ql = (current_q or "").strip().lower()
    for msg in reversed(chat_history or []):
        if str(msg.get("role") or "").lower() != "user":
            continue
        prev = str(msg.get("content") or "").strip()
        if prev and prev.lower() != ql:
            return prev
    return ""


def _effective_question(question: str, chat_history: list[dict[str, Any]]) -> str:
    """Resolve follow-ups by borrowing the previous user intent from chat memory."""
    q = (question or "").strip()
    if not q:
        return q
    if _is_in_scope_question(q):
        return q
    if not _is_followup_question(q):
        return q
    last_user = _latest_user_turn(chat_history, q)
    if not last_user:
        return q
    return f"{last_user}\nFollow-up constraint: {q}"


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
    """Return canonical state tokens found in the question."""
    ql = (question or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for name in _US_STATE_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", ql):
            if name not in seen:
                seen.add(name)
                found.append(name.title())
    for ab in _US_STATE_ABBREVS:
        if re.search(rf"\b{re.escape(ab)}\b", ql):
            key = f"abbr:{ab}"
            if key not in seen:
                seen.add(key)
                found.append(ab.upper())
    return found


def _needs_cross_jurisdiction_retrieval(question: str) -> bool:
    ql = (question or "").lower()
    states = _states_mentioned(question)
    if len({s.lower() for s in states}) >= 2:
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
    return any(m in ql for m in compare_markers)


def _retrieval_jurisdiction_ids(
    question: str, sidebar_jurisdiction_id: int | None
) -> list[int]:
    """Jurisdiction DB ids to search (OR semantics via merge)."""
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


def _infer_category_filter(question: str) -> str | None:
    ql = (question or "").lower()
    if "rent control" in ql or "rent stabilization" in ql:
        return "Rent Control"
    if "rental insurance" in ql or "renters insurance" in ql:
        return "Rental insurance"
    if "esa" in ql or "emotional support" in ql or "assistance animal" in ql or "service animal" in ql:
        return "ESA"
    if "pet policy" in ql or ("pet" in ql and "policy" in ql):
        return "Pet Policy"
    if "tenant" in ql or "landlord" in ql or "security deposit" in ql or "habitability" in ql or "evict" in ql or "eviction" in ql:
        return "Renters"
    return None


def _is_informative_chunk(text: str, min_chars: int) -> bool:
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    if t.count("http") >= 1 and "\n" not in t and "." not in t[:120]:
        return False
    return True


def _diversify_by_source(
    results: list[dict[str, Any]], max_items: int, max_per_source: int
) -> list[dict[str, Any]]:
    """Prefer multiple sources so comparisons aren't dominated by one URL."""
    by_score = sorted(
        results, key=lambda r: float(r.get("rerank_score") or r.get("score") or 0.0), reverse=True
    )
    picked: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    for r in by_score:
        meta = r.get("metadata") or {}
        src = str(meta.get("source_name") or meta.get("url") or "")
        n = per_source.get(src, 0)
        if n >= max_per_source:
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


# ---------------------------------------------------------------------------
# Primary retrieval orchestration
# ---------------------------------------------------------------------------


def _run_hybrid_retrieval(
    store: RegulationVectorStore,
    q_text: str,
    top_n: int,
    plan_jids: list[int],
    category_filter: str | None,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval (vector + lexical + RRF)."""
    return hybrid_search(
        store,
        query=q_text,
        n_results=top_n,
        jurisdiction_ids=plan_jids or None,
        category_filter=category_filter,
    )


def _run_vector_retrieval(
    store: RegulationVectorStore,
    q_text: str,
    top_n: int,
    plan_jids: list[int],
    jurisdiction_id: int | None,
    category_filter: str | None,
    cross: bool,
) -> list[dict[str, Any]]:
    """Run vector-only retrieval (fallback or when hybrid is disabled)."""
    if cross or plan_jids:
        return vector_search(
            store,
            query=q_text,
            n_results=top_n,
            jurisdiction_ids=plan_jids or None,
            category_filter=category_filter,
        )
    search_results = store.search(
        query=q_text,
        n_results=top_n,
        jurisdiction_id=jurisdiction_id,
        category_filter=category_filter,
    )
    return [
        {"document": r.document, "metadata": r.metadata, "score": r.score}
        for r in search_results
    ]


def _run_fallback_broadening(
    store: RegulationVectorStore,
    q_text: str,
    top_n: int,
    category_filter: str | None,
) -> list[dict[str, Any]]:
    """Progressively broaden search: drop jurisdiction, then drop category."""
    try:
        fb1 = store.search(
            query=q_text,
            n_results=top_n,
            jurisdiction_id=None,
            category_filter=category_filter,
        )
        if fb1:
            logger.debug("Fallback: broadened jurisdiction, got %d results", len(fb1))
            return [
                {"document": r.document, "metadata": r.metadata, "score": r.score}
                for r in fb1
            ]
    except Exception:
        pass

    try:
        fb2 = store.search(
            query=q_text,
            n_results=top_n,
            jurisdiction_id=None,
            category_filter=None,
        )
        if fb2:
            logger.debug("Fallback: broadened jurisdiction+category, got %d results", len(fb2))
            return [
                {"document": r.document, "metadata": r.metadata, "score": r.score}
                for r in fb2
            ]
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# QA System class
# ---------------------------------------------------------------------------


class QASystem:
    def __init__(self) -> None:
        self._store = RegulationVectorStore()

    def answer_question(
        self,
        question: str,
        chat_history: list[dict[str, Any]],
        jurisdiction_id: int | None = None,
    ) -> dict[str, Any]:
        """Primary entry point (backward-compatible dict return).

        Orchestrates: question resolution → scope check → jurisdiction plan
        → hybrid retrieval → fallback → rerank → grounding → LLM answer.
        """
        # ---- 1. Resolve effective question ----
        effective_q = _effective_question(question, chat_history)

        # ---- 2. Validate domain scope ----
        if not _is_in_scope_question(effective_q):
            return {"answer": _out_of_scope_answer(), "sources": [], "confidence": "out_of_scope"}

        # ---- 3. Read config-driven RAG settings ----
        cross = _needs_cross_jurisdiction_retrieval(effective_q)
        max_context = settings.RAG_CROSS_JURISDICTION_MAX if cross else settings.RAG_RERANK_TOP_K
        q_text = _retrieval_query(effective_q)
        category_filter = _infer_category_filter(effective_q)
        top_n = settings.RAG_RETRIEVAL_TOP_N
        top_k = settings.RAG_RERANK_TOP_K
        min_chars = settings.RAG_MIN_INFORMATIVE_CHARS
        max_per_source = settings.RAG_MAX_CHUNKS_PER_SOURCE
        use_hybrid = settings.RAG_HYBRID_ENABLED

        # ---- 4. Build jurisdiction-aware retrieval plan ----
        mentioned_jids = _retrieval_jurisdiction_ids(question, jurisdiction_id)
        scoped: list[ScopedJurisdiction] = []
        try:
            scoped = build_retrieval_plan(
                question,
                sidebar_jurisdiction_id=jurisdiction_id,
                mentioned_jurisdiction_ids=mentioned_jids,
                is_cross_jurisdiction=cross,
            )
        except Exception:
            logger.debug("Jurisdiction plan failed, using legacy id list")

        plan_jids = [sj.jurisdiction_id for sj in scoped] if scoped else mentioned_jids
        exact_jid = jurisdiction_id

        logger.debug(
            "RAG pipeline: hybrid=%s, cross=%s, top_n=%d, top_k=%d, plan_jids=%s",
            use_hybrid, cross, top_n, top_k, plan_jids,
        )

        # ---- 5. Hybrid retrieval first when enabled ----
        result_dicts: list[dict[str, Any]] = []
        fallback_used = False

        try:
            if use_hybrid:
                result_dicts = _run_hybrid_retrieval(
                    self._store, q_text, top_n, plan_jids, category_filter
                )
                logger.debug("Hybrid retrieval returned %d results", len(result_dicts))

            # ---- 5b. Fall back to vector-only when hybrid disabled or empty ----
            if not result_dicts:
                if use_hybrid:
                    logger.debug("Hybrid returned empty, falling back to vector-only")
                result_dicts = _run_vector_retrieval(
                    self._store, q_text, top_n, plan_jids,
                    jurisdiction_id, category_filter, cross,
                )
                if use_hybrid and result_dicts:
                    fallback_used = True
                logger.debug("Vector retrieval returned %d results", len(result_dicts))
        except Exception:
            if llm.is_ai_available():
                raise
            result_dicts = []

        # ---- 6. Fallback: broaden jurisdiction, then remove category ----
        if not result_dicts:
            fallback_used = True
            result_dicts = _run_fallback_broadening(
                self._store, q_text, top_n, category_filter
            )

        # ---- 7. Filter non-informative chunks ----
        informative_results = [
            r for r in result_dicts if _is_informative_chunk(str(r.get("document") or ""), min_chars)
        ]
        pool = informative_results or result_dicts

        # ---- 8. Deterministic reranking ----
        reranked = rerank(
            pool,
            query=q_text,
            target_jurisdiction_ids=plan_jids,
            exact_jurisdiction_id=exact_jid,
            top_k=top_k if not cross else max(top_k, max_context),
        )

        # ---- 9. Diversify by source for cross-jurisdiction ----
        selected_results = (
            _diversify_by_source(reranked, max_context, max_per_source)
            if cross
            else reranked[:max_context]
        )

        # ---- 10. Assess confidence from evidence quality ----
        confidence, conflict_notices = assess_confidence(selected_results, scoped)

        # ---- 11. No-LLM fallback ----
        sources = extract_sources(selected_results, max_context, scoped)

        if not llm.is_ai_available():
            return {
                "answer": (
                    "AI-powered answers are currently unavailable because no LLM API key "
                    "is configured. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
                    "GOOGLE_API_KEY in your environment to enable AI responses.\n\n"
                    "In the meantime, here are the most relevant regulation sources I found."
                ),
                "sources": sources,
                "confidence": "weak_evidence",
            }

        # ---- 12. Build grounded context + prompt ----
        context = build_grounded_context(selected_results, scoped, max_context)
        history = _build_history(chat_history)

        compare_note = ""
        if cross:
            compare_note = (
                "\nNote: This question spans multiple jurisdictions or asks for a "
                "comparison. Use all relevant excerpts below; if only federal rules "
                "appear, explain how they apply broadly and what state-specific text "
                "is or is not present.\n"
            )

        thin_context_note = ""
        if selected_results and not informative_results:
            thin_context_note = (
                "\nNote: Retrieved context is sparse/title-like. Still provide a useful "
                "LLM answer grounded in the listed sources and their titles/categories. "
                "Do not respond with 'no information' if relevant sources are present. "
                "Give a high-level overview, jurisdiction caveats, and practical next steps.\n"
            )

        confidence_instruction = ""
        if confidence == "weak_evidence":
            confidence_instruction = (
                "\nIMPORTANT: Evidence for this query is limited. Clearly state that "
                "your answer is based on partial information and recommend verifying "
                "with official sources. Do NOT present uncertain information as definitive.\n"
            )
        elif confidence == "conflicting":
            conflict_text = " ".join(conflict_notices[:3])
            confidence_instruction = (
                f"\nIMPORTANT: Sources may conflict. {conflict_text} "
                "Present both positions and advise consulting legal counsel.\n"
            )

        jurisdiction_note = ""
        if scoped:
            labels = ", ".join(sj.scope_label for sj in scoped[:5])
            jurisdiction_note = f"\nJurisdictions in scope: {labels}\n"

        user_message = (
            f"Regulation context:\n{context}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"{jurisdiction_note}"
            f"{compare_note}"
            f"{thin_context_note}"
            f"{confidence_instruction}"
            f"Question: {question}\n"
            f"Resolved intent for retrieval: {effective_q}"
        )

        raw_answer = llm.ask(system=QA_SYSTEM_PROMPT, user=user_message)
        answer = _clean_answer(raw_answer)

        # ---- 13. Build structured grounded answer ----
        grounded = build_grounded_answer(
            answer_text=answer,
            results=selected_results,
            confidence=confidence,
            conflict_notices=conflict_notices,
            scoped_jurisdictions=scoped,
            fallback_used=fallback_used,
            max_sources=max_context,
        )

        logger.debug(
            "QA complete: confidence=%s, sources=%d, fallback=%s",
            grounded.confidence, len(grounded.sources), grounded.fallback_used,
        )

        return grounded.to_dict()


qa = QASystem()
