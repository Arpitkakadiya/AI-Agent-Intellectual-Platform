"""Unified data models for the RAG pipeline.

All retrieval stages (vector, lexical, hybrid, reranked) normalize
results into ``RetrievalCandidate`` so that downstream components
(reranker, grounding, answer generation) receive a consistent shape.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    """Single chunk flowing through the retrieval → rerank → grounding pipeline."""

    chunk_id: Optional[int] = None
    content: str = ""
    source_id: Optional[int] = None
    source_name: str = ""
    jurisdiction_id: Optional[int] = None
    jurisdiction_label: str = ""
    citation: str = ""
    section_title: str = ""
    url: str = ""
    category: str = ""
    domain: str = ""

    score_vector: float = 0.0
    score_lexical: float = 0.0
    score_fused: float = 0.0
    score_rerank: float = 0.0

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def best_score(self) -> float:
        return max(self.score_rerank, self.score_fused, self.score_vector, self.score_lexical)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert back to the dict shape used by existing code paths."""
        meta = dict(self.metadata)
        meta.update({
            "source_name": self.source_name,
            "url": self.url,
            "category": self.category,
            "domain": self.domain,
            "jurisdiction_id": self.jurisdiction_id,
        })
        if self.section_title:
            meta["section_title"] = self.section_title
        if self.citation:
            meta["citation"] = self.citation

        d: dict[str, Any] = {
            "document": self.content,
            "metadata": meta,
            "score": self.best_score,
        }
        if self.score_rerank > 0:
            d["rerank_score"] = self.score_rerank
        if self.score_fused > 0:
            d["hybrid_score"] = self.score_fused
        return d

    @classmethod
    def from_legacy_dict(cls, d: dict[str, Any], *, origin: str = "vector") -> "RetrievalCandidate":
        """Build a candidate from the existing dict shape used throughout the codebase."""
        meta = d.get("metadata") or {}
        score = float(d.get("score") or 0.0)

        kwargs: dict[str, Any] = {
            "content": d.get("document") or "",
            "source_name": meta.get("source_name") or "",
            "url": meta.get("url") or "",
            "category": meta.get("category") or "",
            "domain": meta.get("domain") or "",
            "jurisdiction_id": _safe_int(meta.get("jurisdiction_id")),
            "section_title": meta.get("section_title") or "",
            "citation": meta.get("citation") or "",
            "metadata": meta,
        }

        if origin == "vector":
            kwargs["score_vector"] = score
        elif origin == "lexical":
            kwargs["score_lexical"] = score
        elif origin == "hybrid":
            kwargs["score_fused"] = float(d.get("hybrid_score") or score)
            kwargs["score_vector"] = score
        elif origin == "rerank":
            kwargs["score_rerank"] = float(d.get("rerank_score") or score)

        return cls(**kwargs)


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
