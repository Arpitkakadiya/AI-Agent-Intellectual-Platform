from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from core.regulations import explorer


def show_page() -> None:
    st.title("Regulation Explorer")

    metrics = explorer.get_explorer_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total regulations", metrics["total_regulations"])
    c2.metric("States covered", metrics["total_states_covered"])
    c3.metric("Last updated", str(metrics["last_updated"] or "N/A"))

    query = st.text_input("Search regulations", placeholder="Search by keyword or topic...")

    categories = explorer.get_distinct_categories()
    selected_category = st.selectbox(
        "Category filter",
        options=["(All categories)"] + categories,
        index=0,
    )
    category_value: Optional[str] = (
        None if selected_category == "(All categories)" else selected_category
    )

    state_options = explorer.get_state_jurisdiction_options()
    state_names = ["(All states)"] + [s["name"] for s in state_options]
    state_index_default = 0
    selected_state_name = st.selectbox(
        "State filter", options=state_names, index=state_index_default
    )
    selected_state_id: Optional[int] = None
    if selected_state_name != "(All states)":
        for s in state_options:
            if s["name"] == selected_state_name:
                selected_state_id = int(s["id"])
                break

    if st.button("Search") and query.strip():
        results = explorer.search_regulations(
            query=query.strip(),
            jurisdiction_id=selected_state_id,
            category=category_value,
            # Keep result count small to reduce pgvector query cost and avoid
            # statement timeouts in Supabase's default statement_timeout window.
            n_results=5,
        )

        st.subheader("Results")
        if not results:
            st.info("No matching regulations found.")
            return

        df = explorer.to_results_dataframe(results)
        st.dataframe(
            df,
            use_container_width=True,
        )


show_page()

