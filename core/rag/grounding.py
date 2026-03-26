"""Answer grounding, source transparency, and uncertainty handling.

Responsible for:
- Building the final context block with clear source attribution
- Classifying answer confidence (grounded / weak_evidence / conflicting / out_of_scope)
- Formatting structured answer responses
- Adding uncertainty language when evidence is thin
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.rag.jurisdiction import ScopedJurisdiction, detect_jurisdiction_conflicts
from core.rag.utils import deduplicate_sources

ConfidenceLevel = Literal["grounded", "weak_evidence", "conflicting", "out_of_scope"]

_MIN_GROUNDED_CHUNKS = 2
_MIN_INFORMATIVE_CHARS = 220


@dataclass
class GroundedAnswer:
    """Structured answer with provenance and confidence metadata."""
    answer: str
    confidence: ConfidenceLevel
    sources: list[dict[str, Any]]
    jurisdiction_labels: list[str] = field(default_factory=list)
    conflict_notices: list[str] = field(default_factory=list)
    fallback_used: bool = False
    uncertainty_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
        }
        if self.jurisdiction_labels:
            d["jurisdiction_labels"] = self.jurisdiction_labels
        if self.conflict_notices:
            d["conflict_notices"] = self.conflict_notices
        if self.fallback_used:
            d["fallback_used"] = True
        if self.uncertainty_note:
            d["uncertainty_note"] = self.uncertainty_note
        return d


def assess_confidence(
    results: list[dict[str, Any]],
    scoped_jurisdictions: list[ScopedJurisdiction] | None = None,
) -> tuple[ConfidenceLevel, list[str]]:
    """Determine confidence level from the retrieved result set."""
    if not results:
        return "out_of_scope", []

    informative = [
        r for r in results
        if len((r.get("document") or "").strip()) >= _MIN_INFORMATIVE_CHARS
    ]

    conflicts = detect_jurisdiction_conflicts(results) if len(results) >= 2 else []

    if conflicts:
        return "conflicting", conflicts

    if len(informative) >= _MIN_GROUNDED_CHUNKS:
        return "grounded", []

    return "weak_evidence", []


def build_grounded_context(
    results: list[dict[str, Any]],
    scoped_jurisdictions: list[ScopedJurisdiction] | None = None,
    max_blocks: int = 5,
) -> str:
    """Build the context string with clear source + jurisdiction labels."""
    blocks: list[str] = []
    scope_map: dict[int, ScopedJurisdiction] = {}
    if scoped_jurisdictions:
        scope_map = {sj.jurisdiction_id: sj for sj in scoped_jurisdictions}

    for r in results[:max_blocks]:
        meta = r.get("metadata") or {}
        header = meta.get("source_name") or meta.get("url") or "Source"

        jid = meta.get("jurisdiction_id")
        scope_label = ""
        if jid is not None and int(jid) in scope_map:
            scope_label = f" [{scope_map[int(jid)].scope_label}]"
        elif jid is not None:
            scope_label = f" [jurisdiction_id={jid}]"

        rerank_score = r.get("rerank_score")
        score_note = f" (relevance: {rerank_score:.2f})" if rerank_score else ""

        blocks.append(
            f"[{header}]{scope_label}{score_note}\n{r['document']}"
        )

    return "\n---\n".join(blocks)


def format_uncertainty_prefix(confidence: ConfidenceLevel) -> str:
    """Return a prefix the LLM should prepend when evidence is weak."""
    if confidence == "weak_evidence":
        return (
            "**Note:** The available regulatory sources for this specific question "
            "are limited. The following answer is based on partial evidence and "
            "may not cover all applicable rules. Verify with official sources.\n\n"
        )
    if confidence == "conflicting":
        return (
            "**Note:** The retrieved sources contain potentially conflicting "
            "provisions across jurisdictions. Review the cited sources carefully "
            "and consult legal counsel for definitive guidance.\n\n"
        )
    return ""


def extract_sources(
    results: list[dict[str, Any]],
    max_items: int,
    scoped_jurisdictions: list[ScopedJurisdiction] | None = None,
) -> list[dict[str, Any]]:
    """Extract and deduplicate source metadata, enriched with jurisdiction labels."""
    scope_map: dict[int, ScopedJurisdiction] = {}
    if scoped_jurisdictions:
        scope_map = {sj.jurisdiction_id: sj for sj in scoped_jurisdictions}

    raw: list[dict[str, Any]] = []
    for r in results[:max_items]:
        meta = r.get("metadata") or {}
        entry: dict[str, Any] = {
            "source": meta.get("source_name", ""),
            "url": meta.get("url", ""),
            "category": meta.get("category", ""),
            "domain": meta.get("domain", ""),
        }
        jid = meta.get("jurisdiction_id")
        if jid is not None and int(jid) in scope_map:
            entry["jurisdiction"] = scope_map[int(jid)].scope_label
        raw.append(entry)

    return deduplicate_sources(raw)


def build_grounded_answer(
    answer_text: str,
    results: list[dict[str, Any]],
    confidence: ConfidenceLevel,
    conflict_notices: list[str],
    scoped_jurisdictions: list[ScopedJurisdiction] | None = None,
    fallback_used: bool = False,
    max_sources: int = 8,
) -> GroundedAnswer:
    """Assemble the final structured answer."""
    sources = extract_sources(results, max_sources, scoped_jurisdictions)
    jurisdiction_labels = []
    if scoped_jurisdictions:
        jurisdiction_labels = [sj.scope_label for sj in scoped_jurisdictions]

    uncertainty = format_uncertainty_prefix(confidence)
    final_answer = f"{uncertainty}{answer_text}" if uncertainty else answer_text

    return GroundedAnswer(
        answer=final_answer,
        confidence=confidence,
        sources=sources,
        jurisdiction_labels=jurisdiction_labels,
        conflict_notices=conflict_notices,
        fallback_used=fallback_used,
        uncertainty_note=uncertainty,
    )
