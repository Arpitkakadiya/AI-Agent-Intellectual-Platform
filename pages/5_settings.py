from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import config
from core.llm.client import llm
from core.regulations.scraper import is_supabase_connected, scraper


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "seeds" / "sources.csv"


# ---------------------------------------------------------------------------
# Section: Data import & vector index
# ---------------------------------------------------------------------------

def _section_data() -> None:
    st.subheader("Data")
    if st.button("Load regulations from CSV", key="btn_csv"):
        try:
            result = scraper.load_regulations_from_csv(csv_path=_csv_path())
            st.success(
                f"Loaded {result.get('loaded', 0)} regulations, "
                f"skipped {result.get('skipped', 0)} duplicates."
            )
        except Exception as exc:
            st.error(f"Failed to load regulations: {exc}")

    st.subheader("Vector Index")
    if st.button("Initialize vector index", key="btn_index"):
        try:
            result = scraper.initialize_vector_index()
            st.success(
                f"Vector index initialized for {result.get('indexed_docs', 0)} "
                f"unindexed regulations (doc count)."
            )
        except Exception as exc:
            st.error(f"Failed to initialize vector index: {exc}")

    st.subheader("Indexing status")
    try:
        status_rows = scraper.get_indexing_status()
        if status_rows:
            st.dataframe(status_rows, use_container_width=True)
        else:
            st.info("No indexing status available yet.")
    except Exception as exc:
        st.error(f"Failed to load indexing status: {exc}")


# ---------------------------------------------------------------------------
# Section: Source Registry link card
# ---------------------------------------------------------------------------

def _section_source_registry_link() -> None:
    st.divider()
    st.subheader("Source Registry")
    st.markdown(
        "Manage the regulation source URLs that the scraper monitors — "
        "add, edit, test, activate/deactivate, and import from CSV."
    )
    st.page_link("pages/6_source_registry.py", label="Open Source Registry", icon="📋")


# ---------------------------------------------------------------------------
# Section: API status
# ---------------------------------------------------------------------------

def _section_api_status() -> None:
    st.divider()
    st.subheader("API status")
    supabase_ok = is_supabase_connected()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anthropic", "set" if config.settings.has_anthropic_key else "not set")
    c2.metric("OpenAI", "set" if config.settings.has_openai_key else "not set")
    c3.metric("Gemini", "set" if config.settings.has_google_key else "not set")
    c4.metric("Supabase", "connected" if supabase_ok else "not connected")


# ---------------------------------------------------------------------------
# Section: LLM provider
# ---------------------------------------------------------------------------

def _section_llm_provider() -> None:
    st.subheader("LLM provider selection")
    chat_options = ["auto", "anthropic", "openai", "gemini"]
    current_chat = st.session_state.get("chat_provider", config.settings.chat_provider)
    if current_chat not in chat_options:
        current_chat = "auto"
    selected_chat = st.selectbox(
        "Chat provider",
        options=chat_options,
        index=chat_options.index(current_chat),
        help="Choose which provider to use for chat responses. 'auto' uses priority order.",
    )
    st.session_state["chat_provider"] = selected_chat

    embed_options = ["gemini", "openai"]
    current_embed = st.session_state.get("embed_provider", config.settings.embed_provider)
    if current_embed not in embed_options:
        current_embed = "gemini"
    selected_embed = st.selectbox(
        "Embedding provider (single provider recommended)",
        options=embed_options,
        index=embed_options.index(current_embed),
        help="Keep one embedding provider for consistent vector quality.",
    )
    st.session_state["embed_provider"] = selected_embed

    if st.button("Apply LLM provider settings", key="btn_llm"):
        llm.set_chat_provider(selected_chat)
        llm.set_embed_provider(selected_embed)
        st.success(
            f"Applied chat provider '{selected_chat}' and embedding provider "
            f"'{selected_embed}'. Current chat mode: {llm.mode}."
        )


# ---------------------------------------------------------------------------
# Section: Scraper
# ---------------------------------------------------------------------------

def _section_scraper() -> None:
    st.divider()
    st.subheader("Scraper")
    if st.button("Manual scraper trigger", key="btn_scrape"):
        try:
            res = scraper.run_manual_scraper()
            scraped = res.get("scraped", 0)
            indexed = res.get("indexed", 0)
            errs = res.get("errors", [])
            st.success(f"Scraped {scraped} sources, indexed {indexed} new versions.")
            if errs:
                st.warning(f"{len(errs)} error(s): {'; '.join(errs[:5])}")
        except Exception as exc:
            st.error(f"Manual scraper trigger failed: {exc}")


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def show_page() -> None:
    st.title("Settings")
    _section_data()
    _section_source_registry_link()
    _section_api_status()
    _section_llm_provider()
    _section_scraper()


show_page()
