from __future__ import annotations

import logging
import re
from typing import Any

from config import settings

from core.llm.client import llm
from core.llm.prompts import QA_SYSTEM_PROMPT
from core.rag.grounding import (
    GroundedAnswer,
    assess_confidence,
    build_grounded_answer,
    build_grounded_context,
    extract_sources,
)
from core.rag.hybrid import hybrid_search, vector_search
from core.rag.jurisdiction import (
    ScopedJurisdiction,
    build_retrieval_plan,
    detect_jurisdiction_conflicts,
)
from core.rag.reranker import rerank
from core.rag.utils import deduplicate_sources
from core.rag.vector_store import RegulationVectorStore, SearchResult
from db.client import get_db

logger = logging.getLogger(__name__)

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
    """
    Resolve follow-ups by borrowing the previous user intent from chat memory.
    This is intentionally general and not tied to any specific category or state.
    """
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


def _infer_category_filter(question: str) -> str | None:
    ql = (question or "").lower()
    if "rent control" in ql or "rent stabilization" in ql:
        return "Rent Control"
    if "rental insurance" in ql or "renters insurance" in ql:
        return "Rental insurance"
    if (
        "esa" in ql
        or "emotional support" in ql
        or "assistance animal" in ql
        or "service animal" in ql
    ):
        return "ESA"
    if "pet policy" in ql or ("pet" in ql and "policy" in ql):
        return "Pet Policy"
    if (
        "tenant" in ql
        or "landlord" in ql
        or "security deposit" in ql
        or "habitability" in ql
        or "evict" in ql
        or "eviction" in ql
    ):
        return "Renters"
    return None


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


def _build_source_based_overview(
    question: str, sources: list[dict[str, Any]], category_filter: str | None
) -> str | None:
    if not sources:
        return None

    labels = [str(s.get("source") or "").strip() for s in sources if s.get("source")]
    labels = [x for x in labels if x]
    if not labels:
        return None

    topic = category_filter or "the requested topic"
    top = labels[:5]
    bullets = "\n".join(f"- {name}" for name in top)
    return (
        f"Here is a high-level overview of {topic} based on the regulation sources in this system.\n\n"
        f"- This topic is covered by multiple official/legal references.\n"
        f"- Rules vary by jurisdiction, so exact requirements can differ by state/city.\n"
        f"- For compliance decisions, use the linked primary sources below.\n\n"
        f"Most relevant sources found:\n{bullets}\n\n"
        f"If you want, I can summarize this specifically for a state/city (for example: "
        f"\"{question} in Colorado\")."
    )


class QASystem:
    def __init__(self) -> None:
        self._store = RegulationVectorStore()

    # ------------------------------------------------------------------
    # Primary entry point (backward-compatible dict return)
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        chat_history: list[dict[str, Any]],
        jurisdiction_id: int | None = None,
    ) -> dict[str, Any]:
        effective_question = _effective_question(question, chat_history)
        if not _is_in_scope_question(effective_question):
            return {"answer": _out_of_scope_answer(), "sources": [], "confidence": "out_of_scope"}

        cross = _needs_cross_jurisdiction_retrieval(effective_question)
        max_context = _MAX_CONTEXT_CROSS_JURISDICTION if cross else _MAX_CONTEXT_RESULTS
        q_text = _retrieval_query(effective_question)
        category_filter = _infer_category_filter(effective_question)
        top_n = getattr(settings, "RAG_RETRIEVAL_TOP_N", 15)
        top_k = getattr(settings, "RAG_RERANK_TOP_K", max_context)

        # ----- 1. Build jurisdiction retrieval plan -----
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

        # ----- 2. Hybrid retrieval -----
        use_hybrid = getattr(settings, "RAG_HYBRID_ENABLED", True)
        fallback_used = False

        try:
            if use_hybrid:
                result_dicts = hybrid_search(
                    self._store,
                    query=q_text,
                    n_results=top_n,
                    jurisdiction_ids=plan_jids or None,
                    category_filter=category_filter,
                )
            elif cross:
                result_dicts = vector_search(
                    self._store,
                    query=q_text,
                    n_results=top_n,
                    jurisdiction_ids=plan_jids or None,
                    category_filter=category_filter,
                )
            else:
                search_results = self._store.search(
                    query=q_text,
                    n_results=top_n,
                    jurisdiction_id=jurisdiction_id,
                    category_filter=category_filter,
                )
                result_dicts = [
                    {"document": r.document, "metadata": r.metadata, "score": r.score}
                    for r in search_results
                ]
        except Exception:
            if llm.is_ai_available():
                raise
            result_dicts = []

        # ----- 2b. Fallback: broaden jurisdiction, then remove category -----
        if not result_dicts:
            fallback_used = True
            try:
                fb1 = self._store.search(
                    query=q_text,
                    n_results=top_n,
                    jurisdiction_id=None,
                    category_filter=category_filter,
                )
                result_dicts = [
                    {"document": r.document, "metadata": r.metadata, "score": r.score}
                    for r in fb1
                ]
            except Exception:
                pass

        if not result_dicts:
            fallback_used = True
            try:
                fb2 = self._store.search(
                    query=q_text,
                    n_results=top_n,
                    jurisdiction_id=None,
                    category_filter=None,
                )
                result_dicts = [
                    {"document": r.document, "metadata": r.metadata, "score": r.score}
                    for r in fb2
                ]
            except Exception:
                pass

        # ----- 3. Filter non-informative chunks -----
        informative_results = [
            r for r in result_dicts if _is_informative_chunk(str(r.get("document") or ""))
        ]
        pool = informative_results or result_dicts

        # ----- 4. Rerank -----
        reranked = rerank(
            pool,
            query=q_text,
            target_jurisdiction_ids=plan_jids,
            exact_jurisdiction_id=exact_jid,
            top_k=top_k,
        )

        # Diversify by source for cross-jurisdiction comparisons
        selected_results = (
            _diversify_by_source(reranked, max_context)
            if cross
            else reranked[:max_context]
        )

        # ----- 5. Assess confidence -----
        confidence, conflict_notices = assess_confidence(selected_results, scoped)

        # ----- 6. No-LLM fallback -----
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

        # ----- 7. Build grounded context + prompt -----
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
            f"Resolved intent for retrieval: {effective_question}"
        )

        raw_answer = llm.ask(system=QA_SYSTEM_PROMPT, user=user_message)
        answer = _clean_answer(raw_answer)

        # ----- 8. Build structured grounded answer -----
        grounded = build_grounded_answer(
            answer_text=answer,
            results=selected_results,
            confidence=confidence,
            conflict_notices=conflict_notices,
            scoped_jurisdictions=scoped,
            fallback_used=fallback_used,
            max_sources=max_context,
        )

        return grounded.to_dict()


qa = QASystem()
