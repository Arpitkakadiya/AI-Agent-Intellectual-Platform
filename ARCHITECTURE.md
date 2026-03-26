# Compliance Agent — Architecture

## Stack
Python 3.11
Streamlit (multi-page via pages/ folder)
Supabase (Postgres + pgvector for vector search)
Anthropic Claude claude-sonnet-4-20250514 — primary LLM (ANTHROPIC_API_KEY)
OpenAI GPT-4o — fallback LLM if no Anthropic key (OPENAI_API_KEY)
Google Gemini — second fallback LLM + default embedding provider (GOOGLE_API_KEY)
Embeddings: Google Gemini (default) or OpenAI text-embedding-3-small — switchable via EMBED_PROVIDER in .env
Rule-based engine (core/compliance/rules.py) — final fallback, works with no API key
Pydantic v2 for all data models
GitHub Actions for scheduled scraping

## LLM fallback order

Check settings.has_anthropic_key → use Anthropic Claude
Else check settings.has_openai_key → use OpenAI GPT-4o
Else check settings.has_google_key → use Google Gemini
Else → rule-based engine in core/compliance/rules.py (no API key required)

Embeddings use EMBED_PROVIDER from .env:

"gemini" (default) → Google text-embedding via google-genai SDK
"openai" → text-embedding-3-small via openai SDK

## Folder structure
compliance-agent/
├── .cursorrules
├── ARCHITECTURE.md
├── FEATURES.md
├── DATA_MODELS.md
├── PROMPTS.md
├── app.py               # Entry point only — max 30 lines, no logic
├── config.py            # Pydantic Settings from env vars
├── requirements.txt
├── .env.example
├── pages/
│   ├── 1_agent.py       # Chat + compliance checker
│   ├── 2_explorer.py    # Regulation explorer
│   ├── 3_update_log.py  # Update log
│   ├── 4_email_alerts.py
│   └── 5_settings.py
├── core/
│   ├── llm/
│   │   ├── client.py    # Provider-agnostic LLM wrapper
│   │   └── prompts.py   # All system prompts
│   ├── compliance/
│   │   ├── checker.py   # Orchestrator
│   │   ├── rules.py     # Rule-based engine
│   │   └── parser.py    # PDF/DOCX parser
│   ├── regulations/
│   │   ├── scraper.py
│   │   └── update_checker.py
│   └── rag/
│       ├── vector_store.py   # Embedding storage + vector search (v2/v3 RPCs)
│       ├── qa_system.py      # Main QA orchestrator (hybrid → rerank → grounded answer)
│       ├── chunking.py       # Legal/compliance-aware document chunking
│       ├── hybrid.py         # Hybrid retrieval (vector + lexical + RRF fusion)
│       ├── jurisdiction.py   # Jurisdiction hierarchy resolution + retrieval planning
│       ├── reranker.py       # Deterministic + optional LLM reranking
│       ├── grounding.py      # Answer confidence, source attribution, uncertainty
│       └── utils.py
├── db/
│   ├── client.py        # Supabase singleton
│   ├── models.py        # Pydantic DB models
│   └── migrations/      # SQL files
├── notifications/
│   └── email_alerts.py
├── data/
│   ├── seeds/sources.csv
│   ├── eval/
│   │   └── eval_dataset.json  # RAG evaluation seed dataset
│   └── guardrails.py
├── scripts/
│   ├── seed_db.py
│   ├── seed_jurisdictions.py
│   ├── index_regulations.py
│   └── rag_eval.py            # RAG evaluation harness
└── tests/

## RAG pipeline (upgraded)
1. **Chunking**: Legal-aware splitting (section/article boundaries) via `core/rag/chunking.py`, fallback to sliding window
2. **Retrieval**: Hybrid search (vector + Postgres full-text) via `core/rag/hybrid.py`, fused with Reciprocal Rank Fusion
3. **Jurisdiction scoping**: Explicit hierarchy resolution (city→state→federal) via `core/rag/jurisdiction.py`
4. **Reranking**: Deterministic scoring (jurisdiction match, topic relevance, citation density, source quality, recency) via `core/rag/reranker.py`
5. **Grounding**: Confidence assessment, source attribution, uncertainty handling via `core/rag/grounding.py`
6. **Answer generation**: Grounded LLM prompt with jurisdiction labels, conflict notices, and uncertainty instructions

## RAG config (env vars / config.py)
- `RAG_HYBRID_ENABLED` (bool, default true) — enable hybrid retrieval
- `RAG_HYBRID_VECTOR_WEIGHT` (float, default 0.6) — vector vs keyword weight in RRF
- `RAG_RETRIEVAL_TOP_N` (int, default 15) — first-stage recall candidates
- `RAG_RERANK_TOP_K` (int, default 5) — final context chunks after reranking
- `RAG_LLM_RERANK_ENABLED` (bool, default false) — use LLM-assisted reranking
- `RAG_USE_LEGAL_CHUNKING` (bool, default true) — use legal-aware chunking

## Key rules
- pages/ imports from core/ and db/ only — no business logic in pages
- core/ never imports streamlit
- All DB access via db/client.py only
- All LLM calls via core/llm/client.py only
- Zero hardcoded city/state/jurisdiction names in logic files
- All jurisdiction resolution via DB lookup by jurisdiction_id (int)
- Legal disclaimer appended to every compliance result
- Rule-based fallback always works without any API key