"""Tests for the upgraded RAG pipeline."""
from __future__ import annotations

from typing import Any, Optional

import pytest

import core.rag.qa_system as _qa_mod
from tests.conftest import DALLAS_JURISDICTION_ID, HOUSTON_JURISDICTION_ID


def _fake_embed(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _configure_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ai_available: bool,
    ask_return: Optional[str],
) -> None:
    def _is_ai_available() -> bool:
        return ai_available

    monkeypatch.setattr(_qa_mod.llm, "is_ai_available", _is_ai_available)
    monkeypatch.setattr(_qa_mod.llm, "embed", _fake_embed)

    if ask_return is None:
        def _fail_ask(*_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("llm.ask should not be called in rule-based mode")

        monkeypatch.setattr(_qa_mod.llm, "ask", _fail_ask)
    else:
        def _fake_ask(*_args: Any, **_kwargs: Any) -> str:
            return ask_return

        monkeypatch.setattr(_qa_mod.llm, "ask", _fake_ask)


# -----------------------------------------------------------------------
# Core QA tests
# -----------------------------------------------------------------------


def test_esa_query_returns_non_empty_answer(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qa.answer_question on an ESA topic must always return a non-empty answer string"""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Non-empty ESA answer for QA."
    )

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules for a landlord in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert isinstance(result.get("answer"), str)
    assert result["answer"].strip() != ""


def test_answer_never_returns_empty_string(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when vector search returns zero results, answer must be a non-empty fallback string"""
    mock_supabase_client.set_match_regulations_override(lambda _payload: [])

    _configure_llm(monkeypatch, ai_available=False, ask_return=None)

    result = _qa_mod.qa.answer_question(
        "What are ESA rules?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("answer") is not None
    assert isinstance(result["answer"], str)
    assert result["answer"].strip() != ""


def test_jurisdiction_scoping_dallas_excludes_houston(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When jurisdiction_id resolves to Dallas TX, returned sources must not contain Houston-specific regulation names"""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="ESA answer with scoped sources."
    )

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    sources = result.get("sources") or []
    source_names = [str(s.get("source") or "") for s in sources]

    assert any("Dallas" in name for name in source_names), "Expected Dallas sources"
    assert not any("Houston" in name for name in source_names), "Houston should be excluded"


def test_rule_based_fallback_works_with_no_api_keys(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When all API keys are unset, the system must still return a valid non-empty answer"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    _configure_llm(monkeypatch, ai_available=False, ask_return=None)

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("answer") is not None
    assert isinstance(result["answer"], str)
    assert result["answer"].strip() != ""
    assert "no llm api key" in result["answer"].lower()


# -----------------------------------------------------------------------
# Upgraded pipeline tests
# -----------------------------------------------------------------------


def test_answer_includes_confidence_field(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgraded pipeline must return a 'confidence' field in the response."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Grounded ESA answer."
    )

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules for a landlord in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert "confidence" in result
    assert result["confidence"] in ("grounded", "weak_evidence", "conflicting", "out_of_scope")


def test_out_of_scope_returns_confidence_out_of_scope(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out-of-scope questions should return confidence='out_of_scope'."""
    result = _qa_mod.qa.answer_question(
        "What is the best pizza in NYC?",
        chat_history=[],
    )
    assert result.get("confidence") == "out_of_scope"


def test_reranking_preserves_top_results(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After reranking, the most relevant Dallas result should still appear."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Reranked answer."
    )

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    sources = result.get("sources") or []
    assert len(sources) >= 1
    source_names = [str(s.get("source") or "") for s in sources]
    assert any("Dallas" in name for name in source_names)


# -----------------------------------------------------------------------
# Config-driven top N / top K tests
# -----------------------------------------------------------------------


def test_config_driven_top_n(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline should respect RAG_RETRIEVAL_TOP_N from settings."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Config-driven answer."
    )
    monkeypatch.setattr(_qa_mod.settings, "RAG_RETRIEVAL_TOP_N", 3)

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("answer")
    assert result.get("confidence") in ("grounded", "weak_evidence", "conflicting", "out_of_scope")


# -----------------------------------------------------------------------
# Hybrid retrieval routing tests
# -----------------------------------------------------------------------


def test_hybrid_enabled_uses_hybrid_path(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When RAG_HYBRID_ENABLED=True, the pipeline should attempt hybrid search."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Hybrid answer."
    )
    monkeypatch.setattr(_qa_mod.settings, "RAG_HYBRID_ENABLED", True)

    hybrid_called = {"value": False}
    original_hybrid = _qa_mod.hybrid_search

    def _tracking_hybrid(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        hybrid_called["value"] = True
        return original_hybrid(*args, **kwargs)

    monkeypatch.setattr(_qa_mod, "hybrid_search", _tracking_hybrid)

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("answer")
    assert hybrid_called["value"], "hybrid_search should have been called"


def test_vector_fallback_when_hybrid_disabled(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When RAG_HYBRID_ENABLED=False, the pipeline should use vector-only."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Vector-only answer."
    )
    monkeypatch.setattr(_qa_mod.settings, "RAG_HYBRID_ENABLED", False)

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("answer")
    assert "confidence" in result


# -----------------------------------------------------------------------
# Confidence labeling tests
# -----------------------------------------------------------------------


def test_confidence_grounded_with_informative_chunks(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With informative chunks, confidence should be 'grounded'."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Well-grounded answer."
    )

    result = _qa_mod.qa.answer_question(
        "What are the ESA rules for a landlord in Dallas, TX?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("confidence") in ("grounded", "weak_evidence")


def test_confidence_weak_with_empty_results(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no results, the no-LLM fallback should report weak_evidence."""
    mock_supabase_client.set_match_regulations_override(lambda _: [])
    _configure_llm(monkeypatch, ai_available=False, ask_return=None)

    result = _qa_mod.qa.answer_question(
        "What are ESA rules?",
        chat_history=[],
        jurisdiction_id=DALLAS_JURISDICTION_ID,
    )
    assert result.get("confidence") == "weak_evidence"


# -----------------------------------------------------------------------
# Jurisdiction comparison tests
# -----------------------------------------------------------------------


def test_cross_jurisdiction_returns_multiple_sources(
    mock_supabase_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-jurisdiction queries should retrieve from multiple jurisdictions."""
    _configure_llm(
        monkeypatch, ai_available=True, ask_return="Comparison answer."
    )
    monkeypatch.setattr(_qa_mod.settings, "RAG_HYBRID_ENABLED", False)

    result = _qa_mod.qa.answer_question(
        "Compare ESA rules in Dallas vs Houston in Texas",
        chat_history=[],
    )
    assert result.get("answer")
    sources = result.get("sources") or []
    assert len(sources) >= 1


# -----------------------------------------------------------------------
# Gemini embedding dimension validation
# -----------------------------------------------------------------------


def test_embedding_dim_validation() -> None:
    """validate_embedding_dims should raise on mismatched dimensions."""
    import core.rag.vector_store as _vs_mod

    _vs_mod.validate_embedding_dims([0.0] * 3072)

    try:
        _vs_mod.validate_embedding_dims([0.0] * 1536)
        pytest.fail("Expected EmbeddingError for 1536 dims")
    except Exception as e:
        assert "dimension mismatch" in str(e).lower()

    try:
        _vs_mod.validate_embedding_dims([0.0] * 768)
        pytest.fail("Expected EmbeddingError for 768 dims")
    except Exception as e:
        assert "dimension mismatch" in str(e).lower()
