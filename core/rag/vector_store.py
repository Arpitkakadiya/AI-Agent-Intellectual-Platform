from __future__ import annotations

from typing import Any, Optional

from config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CONTEXT_CHARS

from pydantic import BaseModel

from core.llm.client import llm
from db.client import get_db


class SearchResult(BaseModel):
    document: str
    metadata: dict[str, Any]
    score: float
    row_id: Optional[int] = None


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
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

        # Replace semantics: delete old embeddings for any affected regulation_ids.
        regulation_ids = sorted(
            {int(d["regulation_id"]) for d in docs if "regulation_id" in d}
        )
        for rid in regulation_ids:
            db.table("regulation_embeddings").delete().eq("regulation_id", rid).execute()

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

        # Batch insert (acts like upsert due to delete+insert above).
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            db.table("regulation_embeddings").insert(rows[i : i + batch_size]).execute()

    def search(
        self,
        query: str,
        n_results: int = 10,
        jurisdiction_id: int | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        db = get_db()

        qemb = query_embedding if query_embedding is not None else llm.embed(query)
        payload: dict[str, Any] = {
            "query_embedding": qemb,
            "match_count": int(n_results),
            "filter_jurisdiction": jurisdiction_id,
        }
        res = db.rpc("match_regulations", payload).execute()

        out: list[SearchResult] = []
        for row in res.data or []:
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

