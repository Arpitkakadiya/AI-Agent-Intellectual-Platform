from __future__ import annotations

import logging
from typing import Any, Optional

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CONTEXT_CHARS, settings

from pydantic import BaseModel

from core.llm.client import llm
from db.client import get_db

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    document: str
    metadata: dict[str, Any]
    score: float
    row_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Backward-compatible chunking entry point
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Chunk text using legal-aware splitter when enabled, else sliding window."""
    if settings.RAG_USE_LEGAL_CHUNKING:
        from core.rag.chunking import chunk_legal_text
        pairs = chunk_legal_text(text, chunk_size=chunk_size, overlap=overlap)
        return [chunk for chunk, _meta in pairs]

    return _sliding_window_chunk(text, chunk_size, overlap)


def _sliding_window_chunk(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Original character-level sliding window (preserved for fallback)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    if chunk_size > MAX_CONTEXT_CHARS:
        raise ValueError("chunk_size must be <= MAX_CONTEXT_CHARS")

    t = text or ""
    if not t:
        return []

    if len(t) <= chunk_size:
        cleaned = t.strip()
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(t), step):
        end = min(start + chunk_size, len(t))
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(t):
            break
    return chunks


class RegulationVectorStore:
    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """
        Each doc: {"text": str, "regulation_id": int, "metadata": dict}
        """
        if not docs:
            return

        db = get_db()

        regulation_ids: list[int] = sorted(
            {int(d["regulation_id"]) for d in docs if "regulation_id" in d}
        )
        if regulation_ids:
            delete_batch_size: int = 500
            for i in range(0, len(regulation_ids), delete_batch_size):
                batch_ids = regulation_ids[i : i + delete_batch_size]
                db.table("regulation_embeddings").delete().in_(
                    "regulation_id", batch_ids
                ).execute()

        rows: list[dict[str, Any]] = []
        for doc in docs:
            text = str(doc.get("text") or "")
            regulation_id = int(doc["regulation_id"])
            for chunk in _chunk_text(text):
                embedding = llm.embed(chunk)
                rows.append(
                    {
                        "regulation_id": regulation_id,
                        "embedding": embedding,
                        "chunk_text": chunk,
                    }
                )

        if not rows:
            return

        batch_size = 100
        for i in range(0, len(rows), batch_size):
            db.table("regulation_embeddings").insert(rows[i : i + batch_size]).execute()

    def search(
        self,
        query: str,
        n_results: int = 10,
        jurisdiction_id: int | None = None,
        query_embedding: list[float] | None = None,
        category_filter: str | None = None,
    ) -> list[SearchResult]:
        """Single-jurisdiction vector search (backward compatible, uses v2 RPC)."""
        db = get_db()

        qemb = query_embedding if query_embedding is not None else llm.embed(query)
        payload: dict[str, Any] = {
            "query_embedding": qemb,
            "match_count": int(n_results),
            "filter_jurisdiction": jurisdiction_id,
            "category_filter": category_filter,
        }
        res = db.rpc("match_regulations_v2", payload).execute()

        return self._parse_vector_results(res.data)

    def search_v3(
        self,
        query: str,
        n_results: int = 10,
        jurisdiction_ids: list[int] | None = None,
        query_embedding: list[float] | None = None,
        category_filter: str | None = None,
    ) -> list[SearchResult]:
        """Multi-jurisdiction vector search using v3 RPC with explicit ID array."""
        db = get_db()

        qemb = query_embedding if query_embedding is not None else llm.embed(query)
        payload: dict[str, Any] = {
            "query_embedding": qemb,
            "match_count": int(n_results),
            "filter_jurisdictions": jurisdiction_ids,
            "category_filter": category_filter,
        }
        try:
            res = db.rpc("match_regulations_v3", payload).execute()
            return self._parse_vector_results(res.data)
        except Exception:
            logger.debug("v3 RPC unavailable, falling back to v2 per-jurisdiction")
            all_hits: list[SearchResult] = []
            for jid in (jurisdiction_ids or [None]):
                all_hits.extend(
                    self.search(query, n_results, jid, qemb, category_filter)
                )
            return all_hits

    @staticmethod
    def _parse_vector_results(data: list[dict[str, Any]] | None) -> list[SearchResult]:
        out: list[SearchResult] = []
        for row in data or []:
            rid = row.get("id")
            out.append(
                SearchResult(
                    document=row.get("chunk_text") or "",
                    metadata=row.get("metadata") or {},
                    score=float(row.get("similarity") or 0.0),
                    row_id=int(rid) if rid is not None else None,
                )
            )
        return out

    def delete_by_regulation_id(self, regulation_id: int) -> None:
        db = get_db()
        db.table("regulation_embeddings").delete().eq("regulation_id", int(regulation_id)).execute()

