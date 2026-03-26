from __future__ import annotations

from typing import Any, Optional

import pytest

import core.rag.qa_system as _qa_mod
from tests.conftest import DALLAS_JURISDICTION_ID


def _fake_embed(_text: str) -> list[float]:
    # Deterministic dummy embedding vector.
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
    """Even when vector search returns zero results, answer must be a non-empty fallback string, not '' or None"""
    mock_supabase_client.set_match_regulations_override(lambda _payload: [])

    # Force the QA system to use its rule-based / no-LLM fallback path.
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
    """When ANTHROPIC_API_KEY, OPENAI_API_KEY, and GOOGLE_API_KEY are all unset,
    the system must still return a valid non-empty answer using the rule-based engine"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Ensure the LLM path is not used at all.
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

