from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import streamlit as st

from core.regulations import explorer
from db.client import get_db
from db.models import Jurisdiction
from ui_theme import apply_theme, log_activity, metric_card, page_hero, section_heading, skeleton_card

_CST = timezone(timedelta(hours=-6))

_MERGED_INTO_RENTERS_INSURANCE = {"rental insurance", "rent control", "renters"}


def _normalize_category(cat: str) -> str:
    """Merge 'Rental Insurance', 'Rent Control', and 'Renters' under 'Renters Insurance'."""
    if cat.strip().lower() in _MERGED_INTO_RENTERS_INSURANCE:
        return "Renters Insurance"
    return cat


_EXPLORER_STATES: list[Jurisdiction] = [
    Jurisdiction(id=1, type="state", name="California", state_code="CA"),
    Jurisdiction(id=2, type="state", name="Colorado", state_code="CO"),
    Jurisdiction(id=3, type="state", name="Florida", state_code="FL"),
    Jurisdiction(id=4, type="state", name="New York", state_code="NY"),
    Jurisdiction(id=5, type="state", name="Texas", state_code="TX"),
]


def _format_sync_time(raw: Any) -> str:
    """Return a human-readable date + time in CST, e.g. 'Apr 10, 2025 · 2:32 PM CST'."""
    if not raw:
        return "N/A"
    try:
        dt = datetime.fromisoformat(str(raw)).astimezone(_CST)
        return dt.strftime("%b %d, %Y · %I:%M %p") + " CST"
    except (ValueError, TypeError):
        return str(raw)


def _get_sources_for_state(state_code: Optional[str]) -> list[dict[str, Any]]:
    """Fetch regulation sources from the regulation_sources table, optionally filtered by state_code."""
    try:
        db = get_db()
        q = db.table("regulation_sources").select("*").order("source_name")
        if state_code:
            q = q.eq("state_code", state_code)
        return q.execute().data or []
    except Exception:  # noqa: BLE001
        return []


def show_page() -> None:
    apply_theme()
    page_hero("🔍", "Regulation Explorer", "Search and browse indexed housing regulations across all covered jurisdictions.", "blue")

    metrics = explorer.get_explorer_metrics()
    total_regs = metrics["total_regulations"]
    last_updated = _format_sync_time(metrics["last_updated"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card("Indexed Regulations", f"{total_regs:,}", "📑"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("States Covered", "5", "📍"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Most Recent Sync", last_updated, "🕐"), unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    section_heading("Search Regulations")

    popular_searches = ["rent control", "eviction notice", "fair housing", "security deposit", "lease termination"]
    chip_cols = st.columns(len(popular_searches))
    chip_clicked: Optional[str] = None
    for i, term in enumerate(popular_searches):
        with chip_cols[i]:
            if st.button(term, key=f"chip_{term}", use_container_width=True):
                chip_clicked = term

    state_names = ["All States"] + [s.name for s in _EXPLORER_STATES]
    state_map: dict[str, Optional[str]] = {"All States": None}
    for s in _EXPLORER_STATES:
        state_map[s.name] = s.state_code

    col_cat, col_state, col_btn = st.columns([2, 2, 1])

    raw_categories = explorer.get_distinct_categories()
    categories = sorted({_normalize_category(c) for c in raw_categories})
    with col_cat:
        selected_category = st.selectbox(
            "Category",
            options=["All Categories"] + categories,
            index=0,
        )
    category_value: Optional[str] = (
        None if selected_category == "All Categories" else selected_category
    )

    with col_state:
        selected_state_name = st.selectbox(
            "State",
            options=state_names,
            index=0,
        )
    selected_state_code = state_map[selected_state_name]

    with col_btn:
        st.markdown('<div style="height:1.6rem;"></div>', unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    should_search = search_clicked or chip_clicked
    if should_search:
        query_label = chip_clicked or selected_state_name
        results_placeholder = st.empty()
        results_placeholder.markdown(skeleton_card(3), unsafe_allow_html=True)

        sources = _get_sources_for_state(selected_state_code)

        if chip_clicked or category_value:
            keyword = (chip_clicked or "").lower()
            if category_value == "Renters Insurance":
                match_cats = _MERGED_INTO_RENTERS_INSURANCE
            else:
                match_cats = {category_value.lower()} if category_value else set()

            filtered: list[dict[str, Any]] = []
            for s in sources:
                name = (s.get("source_name") or "").lower()
                cat = (s.get("category") or "").lower()
                url = (s.get("url") or "").lower()
                if keyword and keyword not in name and keyword not in cat and keyword not in url:
                    continue
                if match_cats and cat not in match_cats:
                    continue
                filtered.append(s)
            sources = filtered

        results_placeholder.empty()
        log_activity("Searched regulations", str(query_label)[:60])

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        section_heading("Search Results")

        if not sources:
            st.info("No matching regulation sources found. Try a different state or category.")
            return

        for src in sources:
            name = src.get("source_name") or "Unknown"
            url = src.get("url") or ""
            cat = _normalize_category(src.get("category") or "General")
            domain = src.get("domain") or "housing"
            active = src.get("is_active", True)
            last_scraped = src.get("last_scraped_at")
            state = src.get("state_code") or ""

            status_dot = "🟢" if active else "🔴"
            scraped_label = _format_sync_time(last_scraped) if last_scraped else "Never scraped"

            with st.container(border=True):
                col_info, col_meta = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f'<div style="font-weight:600;font-size:0.95rem;color:var(--rc-text);">'
                        f'{status_dot} {name}</div>'
                        f'<div style="font-size:0.8rem;color:var(--rc-text-muted);margin-top:0.15rem;">'
                        f'{cat} · {domain} · {state}</div>'
                        f'<div style="font-size:0.78rem;margin-top:0.35rem;">'
                        f'<a href="{url}" target="_blank" style="color:var(--rc-primary);text-decoration:none;">'
                        f'{url}</a></div>',
                        unsafe_allow_html=True,
                    )
                with col_meta:
                    st.markdown(
                        f'<div style="font-size:0.75rem;color:var(--rc-text-muted);text-align:right;">'
                        f'Last scraped<br/><strong>{scraped_label}</strong></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            f'<div style="font-size:0.8rem;color:var(--rc-text-muted);margin-top:1rem;text-align:center;">'
            f'Showing {len(sources)} source(s)</div>',
            unsafe_allow_html=True,
        )


show_page()
