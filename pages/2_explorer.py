from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from core.regulations import explorer
from ui_theme import apply_theme, page_header, section_heading


def show_page() -> None:
    apply_theme()
    page_header("Regulation Explorer", "Search and browse housing regulations across jurisdictions")

    metrics = explorer.get_explorer_metrics()
    total_regs = metrics["total_regulations"]
    total_states = metrics["total_states_covered"]
    last_updated = str(metrics["last_updated"] or "N/A")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("📑 Indexed in database", f"{total_regs:,}")
    with c2:
        with st.container(border=True):
            st.metric("📍 Jurisdictions", total_states)
    with c3:
        with st.container(border=True):
            st.metric("🕐 Most recent sync", last_updated)

    st.write("")

    query = st.text_input(
        "Search regulations",
        placeholder="Search by title, keyword, or code...",
    )

    col_cat, col_state = st.columns(2)

    categories = explorer.get_distinct_categories()
    with col_cat:
        selected_category = st.selectbox(
            "Category",
            options=["All Categories"] + categories,
            index=0,
        )
    category_value: Optional[str] = (
        None if selected_category == "All Categories" else selected_category
    )

    state_options = explorer.get_state_jurisdiction_options()
    state_names = ["All States"] + [s["name"] for s in state_options]
    with col_state:
        selected_state_name = st.selectbox(
            "State",
            options=state_names,
            index=0,
        )
    selected_state_id: Optional[int] = None
    if selected_state_name != "All States":
        for s in state_options:
            if s["name"] == selected_state_name:
                selected_state_id = int(s["id"])
                break

    if st.button("🔍 Search", type="primary") and query.strip():
        with st.spinner("Searching regulations..."):
            results = explorer.search_regulations(
                query=query.strip(),
                jurisdiction_id=selected_state_id,
                category=category_value,
                n_results=5,
            )

        section_heading("Search Results")

        if not results:
            st.info("No matching regulations found.")
            return

        df = explorer.to_results_dataframe(results)
        st.dataframe(df, use_container_width=True, hide_index=True)


show_page()
