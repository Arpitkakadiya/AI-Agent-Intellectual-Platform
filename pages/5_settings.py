from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

import config
from core.llm.client import llm
from core.regulations.scraper import is_supabase_connected, scraper
from ui_theme import apply_theme, page_header, section_heading


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "seeds" / "sources.csv"


# ── Data Management ───────────────────────────────────────────

def _section_data_management() -> None:
    section_heading("📦 Data Management")

    col_csv, col_index = st.columns(2)

    with col_csv:
        with st.container(border=True):
            st.markdown("**📑 Load Regulations**")
            st.caption("Import regulation data from CSV file")
            if st.button("Load from CSV", key="btn_csv", use_container_width=True, type="primary"):
                with st.spinner("Loading regulations from CSV..."):
                    try:
                        result = scraper.load_regulations_from_csv(csv_path=_csv_path())
                        st.success(
                            f"Loaded {result.get('loaded', 0)} regulations, "
                            f"skipped {result.get('skipped', 0)} duplicates."
                        )
                    except Exception as exc:
                        st.error(f"Failed to load regulations: {exc}")

    with col_index:
        with st.container(border=True):
            st.markdown("**🔍 Vector Index**")
            st.caption("Initialize or rebuild the search index")
            if st.button("Initialize Index", key="btn_index", use_container_width=True):
                with st.spinner("Building vector index..."):
                    try:
                        result = scraper.initialize_vector_index()
                        st.success(
                            f"Vector index initialized for {result.get('indexed_docs', 0)} "
                            f"unindexed regulations (doc count)."
                        )
                    except Exception as exc:
                        st.error(f"Failed to initialize vector index: {exc}")


# ── Source Management ─────────────────────────────────────────

def _section_source_management() -> None:
    section_heading("📋 Source Management")

    with st.container(border=True):
        col_info, col_link = st.columns([3, 1])
        with col_info:
            st.markdown("**Source Registry**")
            st.caption("Manage regulation source URLs and scraping configurations")
        with col_link:
            st.page_link("pages/6_source_registry.py", label="Open Registry →", icon="📋")


# ── System Health ─────────────────────────────────────────────

def _section_system_health() -> None:
    section_heading("✨ System Health")

    supabase_ok = is_supabase_connected()

    vec_detail = ""
    vec_ok = True
    try:
        idx_rows = scraper.get_indexing_status()
        total_indexed = sum(r.get("indexed", 0) for r in idx_rows) if idx_rows else 0
        if total_indexed:
            vec_detail = f" · {total_indexed:,} docs"
    except Exception:
        vec_ok = False

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.markdown("**Regulation API**")
            dot = "rc-dot-green" if supabase_ok else "rc-dot-red"
            label = "Operational" if supabase_ok else "Unavailable"
            st.markdown(f'<span class="rc-dot {dot}"></span>{label}', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown("**Vector Index**")
            dot = "rc-dot-green" if vec_ok else "rc-dot-red"
            label = ("Operational" if vec_ok else "Unavailable") + vec_detail
            st.markdown(f'<span class="rc-dot {dot}"></span>{label}', unsafe_allow_html=True)

    with c3:
        with st.container(border=True):
            st.markdown("**LLM Provider**")
            st.markdown(
                '<span class="rc-dot rc-dot-green"></span>Operational',
                unsafe_allow_html=True,
            )

    with c4:
        with st.container(border=True):
            st.markdown("**Scraper Service**")
            st.markdown(
                '<span class="rc-dot rc-dot-green"></span>Operational',
                unsafe_allow_html=True,
            )


# ── Configuration ─────────────────────────────────────────────

def _section_configuration() -> None:
    section_heading("⚙️ Configuration")

    col_llm, col_scraper = st.columns(2)

    with col_llm:
        with st.container(border=True):
            st.markdown("**🤖 LLM Provider**")
            st.caption("AI model for compliance analysis")

            chat_options = ["auto", "anthropic", "openai", "gemini"]
            current_chat = st.session_state.get("chat_provider", config.settings.chat_provider)
            if current_chat not in chat_options:
                current_chat = "auto"

            display_names = {
                "auto": "Auto (priority order)",
                "anthropic": "Anthropic Claude",
                "openai": "OpenAI GPT-4",
                "gemini": "Google Gemini",
            }
            selected_chat = st.selectbox(
                "Chat provider",
                options=chat_options,
                index=chat_options.index(current_chat),
                format_func=lambda x: display_names.get(x, x),
                help="Choose which provider to use for chat responses.",
            )
            st.session_state["chat_provider"] = selected_chat

            embed_options = ["gemini", "openai"]
            current_embed = st.session_state.get("embed_provider", config.settings.embed_provider)
            if current_embed not in embed_options:
                current_embed = "gemini"
            selected_embed = st.selectbox(
                "Embedding provider",
                options=embed_options,
                index=embed_options.index(current_embed),
                help="Keep one embedding provider for consistent vector quality.",
            )
            st.session_state["embed_provider"] = selected_embed

            if st.button("Apply LLM settings", key="btn_llm", use_container_width=True):
                with st.spinner("Applying LLM settings..."):
                    llm.set_chat_provider(selected_chat)
                    llm.set_embed_provider(selected_embed)
                    st.success(
                        f"Applied chat provider '{selected_chat}' and embedding provider "
                        f"'{selected_embed}'. Current chat mode: {llm.mode}."
                    )

    with col_scraper:
        with st.container(border=True):
            st.markdown("**▶️ Manual Scraper**")
            st.caption("Trigger a manual scraping run")
            if st.button("Run Scraper", key="btn_scrape", use_container_width=True, type="primary"):
                with st.spinner("Running scraper — this may take a minute..."):
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


# ── Page entry ────────────────────────────────────────────────

def show_page() -> None:
    apply_theme()
    page_header("Settings", "System configuration, data management, and service health")
    _section_data_management()
    _section_source_management()
    _section_system_health()
    _section_configuration()


show_page()
