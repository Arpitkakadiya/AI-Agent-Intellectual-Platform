"""Tests for the legal/compliance-aware chunking module."""
from __future__ import annotations

import pytest

from core.rag.chunking import (
    ChunkMeta,
    chunk_legal_text,
    _find_section_boundaries,
    _sliding_window_chunks,
)


class TestSlidingWindowFallback:
    def test_empty_text_returns_empty(self) -> None:
        assert _sliding_window_chunks("") == []
        assert _sliding_window_chunks("   ") == []

    def test_short_text_single_chunk(self) -> None:
        result = _sliding_window_chunks("Hello world", chunk_size=100)
        assert result == ["Hello world"]

    def test_long_text_splits(self) -> None:
        text = "A" * 1000
        chunks = _sliding_window_chunks(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)


class TestSectionBoundaryDetection:
    def test_detects_section_headings(self) -> None:
        text = "Preamble text.\n\nSection 5.2 Definitions\nSome content.\n\nSection 5.3 Scope\nMore content."
        boundaries = _find_section_boundaries(text)
        assert len(boundaries) >= 2

    def test_detects_article_headings(self) -> None:
        text = "ARTICLE I\nGeneral provisions.\n\nARTICLE II\nSpecific rules."
        boundaries = _find_section_boundaries(text)
        assert len(boundaries) >= 2

    def test_detects_section_symbol(self) -> None:
        text = "Introduction.\n\n§ 42.1 Fair Housing\nContent here.\n\n§ 42.2 Definitions\nMore."
        boundaries = _find_section_boundaries(text)
        assert len(boundaries) >= 2

    def test_no_boundaries_in_plain_text(self) -> None:
        text = "This is just a plain paragraph with no legal structure at all."
        boundaries = _find_section_boundaries(text)
        assert len(boundaries) == 0


class TestLegalChunking:
    def test_empty_returns_empty(self) -> None:
        assert chunk_legal_text("") == []

    def test_short_text_one_chunk(self) -> None:
        result = chunk_legal_text("Short legal text.", chunk_size=500)
        assert len(result) == 1
        text, meta = result[0]
        assert text == "Short legal text."
        assert isinstance(meta, ChunkMeta)
        assert meta.chunk_index == 0
        assert meta.total_chunks == 1

    def test_structured_text_splits_by_section(self) -> None:
        text = (
            "ARTICLE I\n"
            "This is the first article with some content about general provisions.\n\n"
            "ARTICLE II\n"
            "This is the second article with specific rules and regulations.\n\n"
            "ARTICLE III\n"
            "This is the third article about enforcement and penalties."
        )
        result = chunk_legal_text(text, chunk_size=2000)
        assert len(result) >= 3
        titles = [meta.section_title for _, meta in result]
        assert any("ARTICLE I" in t for t in titles)

    def test_definitions_detected(self) -> None:
        text = (
            '"Assistance animal" means any animal that works, provides assistance, '
            "or performs tasks for the benefit of a person with a disability. "
            "This definition includes emotional support animals under HUD guidance."
        )
        result = chunk_legal_text(text, chunk_size=2000)
        assert len(result) >= 1
        _, meta = result[0]
        assert meta.has_definitions is True

    def test_effective_date_detected(self) -> None:
        text = (
            "Section 5.1 Rent Control\n"
            "Effective Date: January 1, 2024\n"
            "All rental units in the city shall be subject to the following rent "
            "stabilization provisions as outlined in this ordinance."
        )
        result = chunk_legal_text(text, chunk_size=2000)
        assert len(result) >= 1
        _, meta = result[0]
        assert meta.has_effective_date is True

    def test_oversized_section_gets_subsplit(self) -> None:
        long_section = "Section 1.0 Overview\n" + ("Legal content. " * 200)
        result = chunk_legal_text(long_section, chunk_size=300, overlap=50)
        assert len(result) > 1

    def test_chunk_indices_are_sequential(self) -> None:
        text = (
            "ARTICLE I\nContent one.\n\n"
            "ARTICLE II\nContent two.\n\n"
            "ARTICLE III\nContent three."
        )
        result = chunk_legal_text(text, chunk_size=2000)
        indices = [meta.chunk_index for _, meta in result]
        assert indices == list(range(len(result)))
        totals = [meta.total_chunks for _, meta in result]
        assert all(t == len(result) for t in totals)


class TestChunkMetaSerialization:
    def test_to_dict_basic(self) -> None:
        meta = ChunkMeta(
            section_title="§ 42.1 Fair Housing",
            chunk_index=0,
            total_chunks=3,
            has_definitions=True,
            has_effective_date=False,
            citation_hint="§ 42.1",
        )
        d = meta.to_dict()
        assert d["section_title"] == "§ 42.1 Fair Housing"
        assert d["citation_hint"] == "§ 42.1"
        assert d["has_definitions"] is True

    def test_source_metadata_propagated(self) -> None:
        result = chunk_legal_text(
            "Section 1.0 Overview\nSome content about housing.",
            chunk_size=2000,
            source_metadata={"source_name": "HUD", "jurisdiction_id": 42},
        )
        assert len(result) >= 1
        _, meta = result[0]
        assert meta.extra.get("source_name") == "HUD"
        assert meta.extra.get("jurisdiction_id") == 42

    def test_citation_hint_extracted(self) -> None:
        text = "Under § 3604 of the Fair Housing Act, discrimination is prohibited."
        result = chunk_legal_text(text, chunk_size=2000)
        assert len(result) >= 1
        _, meta = result[0]
        assert "3604" in meta.citation_hint


class TestRerankerScoring:
    """Basic smoke tests for the deterministic reranker."""

    def test_rerank_returns_scored_results(self) -> None:
        from core.rag.reranker import rerank_deterministic

        results = [
            {
                "document": "ESA rules under Fair Housing Act § 3604",
                "metadata": {
                    "source_name": "HUD Guidance",
                    "url": "https://hud.gov/esa",
                    "jurisdiction_id": 1,
                },
                "score": 0.9,
            },
            {
                "document": "General pet policy information",
                "metadata": {
                    "source_name": "Pet Blog",
                    "url": "https://example.com/pets",
                    "jurisdiction_id": 2,
                },
                "score": 0.8,
            },
        ]
        reranked = rerank_deterministic(results, "ESA rules", target_jurisdiction_ids=[1])
        assert len(reranked) == 2
        assert "rerank_score" in reranked[0]
        assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]

    def test_rerank_section_relevance(self) -> None:
        from core.rag.reranker import rerank_deterministic

        results = [
            {
                "document": "Content about pets",
                "metadata": {
                    "source_name": "Source A",
                    "url": "https://example.com/a",
                    "section_title": "ESA Rules and Regulations",
                },
                "score": 0.7,
            },
            {
                "document": "Content about pets",
                "metadata": {
                    "source_name": "Source B",
                    "url": "https://example.com/b",
                    "section_title": "Parking Requirements",
                },
                "score": 0.7,
            },
        ]
        reranked = rerank_deterministic(results, "ESA rules")
        assert reranked[0]["metadata"]["section_title"] == "ESA Rules and Regulations"


class TestGroundingConfidence:
    """Tests for the grounding module's confidence assessment."""

    def test_empty_results_out_of_scope(self) -> None:
        from core.rag.grounding import assess_confidence
        conf, notices = assess_confidence([])
        assert conf == "out_of_scope"

    def test_informative_chunks_grounded(self) -> None:
        from core.rag.grounding import assess_confidence
        results = [
            {"document": "A" * 300 + " § 3604 Fair Housing Act provision", "metadata": {"url": "https://hud.gov/esa"}},
            {"document": "B" * 300 + " Section 504 regulation text here", "metadata": {"url": "https://hud.gov/504"}},
        ]
        conf, notices = assess_confidence(results)
        assert conf == "grounded"

    def test_short_chunks_weak(self) -> None:
        from core.rag.grounding import assess_confidence
        results = [
            {"document": "short", "metadata": {}},
        ]
        conf, notices = assess_confidence(results)
        assert conf == "weak_evidence"
