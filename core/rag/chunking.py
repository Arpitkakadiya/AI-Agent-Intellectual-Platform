"""Legal/compliance-aware document chunking.

Splits regulation text by structural markers (section headings, article
boundaries, numbered subsections) before applying size control.  Falls back
to the original sliding-window chunker for unstructured text.

Each chunk carries a ``ChunkMeta`` dict so downstream retrieval and reranking
can use section titles, jurisdiction hints, and positional info.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE

# ---------------------------------------------------------------------------
# Section-boundary patterns common in US legal / regulatory text
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"""
    (?:^|\n)                          # start of text or newline
    (?:                               # one of several heading styles:
        (?:ARTICLE|CHAPTER|PART|TITLE|DIVISION)\s+[IVXLCDM\d]+  # ARTICLE IV
      | §\s*[\d.]+                    # § 42.1
      | (?:Sec(?:tion)?|SEC(?:TION)?)\s*\.?\s*[\d.\-]+           # Section 5.2
      | \d+\.\d+[\.\d]*              # 12.3.1 style numbering
      | [A-Z][A-Z ]{4,}(?:\n|$)      # ALL-CAPS heading line
    )
    """,
    re.VERBOSE | re.MULTILINE,
)

_EFFECTIVE_DATE_RE = re.compile(
    r"(?:effective|eff\.?)\s*(?:date)?\s*[:;]?\s*\w+\s+\d{1,2},?\s*\d{4}",
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r'(?:^|\n)\s*"[^"]{3,60}"\s*(?:means|shall mean|refers to|is defined as)',
    re.IGNORECASE,
)


@dataclass
class ChunkMeta:
    """Lightweight metadata attached to each chunk during indexing."""
    section_title: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    has_definitions: bool = False
    has_effective_date: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _find_section_boundaries(text: str) -> list[int]:
    """Return character offsets where new sections begin."""
    return sorted({m.start() for m in _SECTION_RE.finditer(text)})


def _extract_section_title(block: str) -> str:
    """Best-effort extraction of the first heading-like line from a block."""
    for line in block.strip().splitlines()[:3]:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 120:
            continue
        if _SECTION_RE.match(stripped) or stripped.isupper():
            return stripped
    return ""


def _sliding_window_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Character-level sliding window (original algorithm preserved)."""
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= chunk_size:
        return [t]
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(t), step):
        end = min(start + chunk_size, len(t))
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(t):
            break
    return chunks


def _split_oversized_section(
    section: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split a single section that exceeds chunk_size using paragraph
    boundaries first, then falling back to sliding window."""
    paragraphs = re.split(r"\n{2,}", section)
    if len(paragraphs) <= 1:
        return _sliding_window_chunks(section, chunk_size, overlap)

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                chunks.extend(_sliding_window_chunks(para, chunk_size, overlap))
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_legal_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_metadata: dict[str, Any] | None = None,
) -> list[tuple[str, ChunkMeta]]:
    """Split legal/regulatory text into chunks with metadata.

    Strategy:
    1. Detect section boundaries via regex patterns.
    2. Split at those boundaries to preserve heading+body coherence.
    3. If a section exceeds ``chunk_size``, sub-split by paragraphs then
       sliding window.
    4. If no section markers are found, fall back to sliding window.

    Returns a list of ``(chunk_text, chunk_meta)`` tuples.
    """
    t = (text or "").strip()
    if not t:
        return []

    boundaries = _find_section_boundaries(t)

    if not boundaries:
        raw_chunks = _sliding_window_chunks(t, chunk_size, overlap)
        total = len(raw_chunks)
        return [
            (
                c,
                ChunkMeta(
                    chunk_index=i,
                    total_chunks=total,
                    has_definitions=bool(_DEFINITION_RE.search(c)),
                    has_effective_date=bool(_EFFECTIVE_DATE_RE.search(c)),
                ),
            )
            for i, c in enumerate(raw_chunks)
        ]

    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    sections: list[str] = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(t)
        section = t[start:end].strip()
        if section:
            sections.append(section)

    result: list[tuple[str, ChunkMeta]] = []
    for section in sections:
        title = _extract_section_title(section)
        if len(section) <= chunk_size:
            sub_chunks = [section]
        else:
            sub_chunks = _split_oversized_section(section, chunk_size, overlap)
        for sc in sub_chunks:
            meta = ChunkMeta(
                section_title=title,
                has_definitions=bool(_DEFINITION_RE.search(sc)),
                has_effective_date=bool(_EFFECTIVE_DATE_RE.search(sc)),
            )
            result.append((sc, meta))

    total = len(result)
    for i, (_, meta) in enumerate(result):
        meta.chunk_index = i
        meta.total_chunks = total

    return result
