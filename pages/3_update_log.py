from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from core.regulations.explorer import get_state_jurisdiction_options
from core.regulations.update_checker import update_checker
from db.client import get_db
from ui_theme import apply_theme, page_header

CATEGORY_BADGE = {
    "rent control": "rc-badge-teal",
    "fair housing": "rc-badge-blue",
    "eviction": "rc-badge-red",
    "tenant rights": "rc-badge-teal",
    "building codes": "rc-badge-amber",
}


def _badge_cls(category: str) -> str:
    return CATEGORY_BADGE.get(category.lower().strip(), "rc-badge-slate")


def show_page() -> None:
    apply_theme()
    page_header("Update Log", "Monitor regulatory changes and newly detected updates")

    state_options = get_state_jurisdiction_options()
    state_names = ["All States"] + [s["name"] for s in state_options]

    col_filter, col_count = st.columns(2)

    with col_filter:
        selected_state_name = st.selectbox("Filter by state", options=state_names, index=0)
    selected_state_id: Optional[int] = None
    if selected_state_name != "All States":
        for s in state_options:
            if s["name"] == selected_state_name:
                selected_state_id = int(s["id"])
                break

    with col_count:
        count = st.slider("Show up to N updates", min_value=1, max_value=50, value=10)

    if st.button("🔄 Check for updates", type="primary"):
        with st.spinner("Scanning for regulatory changes..."):
            updates = update_checker.check_for_updates()
            st.session_state["latest_updates"] = updates
        st.rerun()

    updates: list[Any] = st.session_state.get("latest_updates") or []
    if not updates:
        st.info("No updates found. Click **Check for updates** to scan for changes.")
        return

    filtered = []
    for u in updates:
        affected = getattr(u, "affected_jurisdiction_ids", []) or []
        if selected_state_id is None or int(selected_state_id) in [int(x) for x in affected]:
            filtered.append(u)

    filtered.sort(key=lambda x: getattr(x, "detected_at"), reverse=True)

    deduped: list[Any] = []
    seen: set[str] = set()
    for u in filtered:
        key = f"{getattr(u, 'source_name', '')}|{getattr(u, 'url', '')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)

    deduped = deduped[:count]

    all_affected_ids: set[int] = set()
    for u in deduped:
        for jid in getattr(u, "affected_jurisdiction_ids", []) or []:
            all_affected_ids.add(int(jid))

    jid_to_name: dict[int, str] = {}
    if all_affected_ids:
        db = get_db()
        juris_res = (
            db.table("jurisdictions")
            .select("id,name")
            .in_("id", list(all_affected_ids))
            .execute()
        )
        for row in juris_res.data or []:
            jid_to_name[int(row["id"])] = str(row.get("name") or row["id"])

    st.write("")
    for u in deduped:
        category = str(getattr(u, "category", "") or "")
        url = str(getattr(u, "url", "") or "")
        update_summary = str(getattr(u, "update_summary", "") or "")
        detected_at = str(getattr(u, "detected_at", "") or "")
        source_name = str(getattr(u, "source_name", "") or category)
        affected_ids = getattr(u, "affected_jurisdiction_ids", []) or []
        affected_names = [jid_to_name.get(int(jid), str(jid)) for jid in affected_ids]

        badge_cls = _badge_cls(category)
        location_str = ", ".join(affected_names) if affected_names else "Statewide"
        date_str = detected_at[:10] if len(detected_at) >= 10 else detected_at

        with st.container(border=True):
            st.markdown(f"**{source_name}**")
            st.markdown(
                f'<span class="rc-badge {badge_cls}">{category}</span>'
                f" &nbsp; 📍 {location_str}"
                f" &nbsp; 📅 {date_str}",
                unsafe_allow_html=True,
            )
            if update_summary:
                with st.expander("Details"):
                    st.write(update_summary)
                    if url:
                        st.markdown(f"[View source ↗]({url})")


show_page()
