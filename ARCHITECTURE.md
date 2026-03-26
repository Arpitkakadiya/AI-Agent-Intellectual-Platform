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
│       ├── vector_store.py
│       ├── qa_system.py
│       └── utils.py
├── db/
│   ├── client.py        # Supabase singleton
│   ├── models.py        # Pydantic DB models
│   └── migrations/      # SQL files
├── notifications/
│   └── email_alerts.py
├── data/
│   ├── seeds/sources.csv
│   └── guardrails.py
├── scripts/
│   ├── seed_db.py
│   ├── seed_jurisdictions.py
│   └── index_regulations.py
└── tests/

## Key rules
- pages/ imports from core/ and db/ only — no business logic in pages
- core/ never imports streamlit
- All DB access via db/client.py only
- All LLM calls via core/llm/client.py only
- Zero hardcoded city/state/jurisdiction names in logic files
- All jurisdiction resolution via DB lookup by jurisdiction_id (int)
- Legal disclaimer appended to every compliance result
- Rule-based fallback always works without any API key