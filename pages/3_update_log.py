from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import streamlit as st

from core.regulations.explorer import get_state_jurisdiction_options
from core.regulations.update_checker import update_checker
from db.client import get_db
from ui_theme import apply_theme, cross_page_link, log_activity, page_hero, section_heading


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
    page_hero("📄", "Update Log", "Monitor regulatory changes across jurisdictions — scan for new laws, amendments, and policy updates.", "green")

    last_scan_count = st.session_state.pop("update_log_last_scan_count", None)
    if last_scan_count is not None:
        if last_scan_count == 0:
            st.success("Scan complete. No new regulatory changes detected.")
        else:
            st.success(f"Scan complete. {last_scan_count} update(s) recorded.")

    scan_err = st.session_state.pop("update_log_scan_error", None)
    if scan_err:
        st.error(scan_err)

    state_options = get_state_jurisdiction_options()
    state_names = ["All States"] + [s["name"] for s in state_options]

    col_filter, col_count, col_btn = st.columns([2, 2, 1])

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

    with col_btn:
        st.markdown('<div style="height:1.6rem;"></div>', unsafe_allow_html=True)
        if st.button("Check for updates", type="primary", use_container_width=True):
            with st.spinner("Scanning for regulatory changes..."):
                try:
                    updates = update_checker.check_for_updates(send_alerts=True)
                    st.session_state["update_log_last_scan_count"] = len(updates)
                except Exception as exc:
                    raw = exc.args[0] if exc.args else None
                    if isinstance(raw, dict):
                        msg = str(raw.get("message") or raw)
                    else:
                        msg = str(raw if raw is not None else exc)
                    low = msg.lower()
                    if "permission denied" in low or "42501" in msg:
                        st.session_state["update_log_scan_error"] = (
                            "Cannot write to `regulation_updates` (permission denied). "
                            "Run `db/migrations/011_regulation_updates_rls.sql` in the Supabase SQL Editor, then retry."
                        )
                    else:
                        st.session_state["update_log_scan_error"] = f"Scan failed: {msg}"
            log_activity("Checked for updates", f"scan complete")
            st.rerun()

    raw_updates, fetch_err = update_checker.fetch_update_log_from_db(limit=400)
    if fetch_err:
        st.error(fetch_err)
    has_any_in_db = len(raw_updates) > 0

    filtered: list[Any] = []
    for u in raw_updates:
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

    if not deduped:
        if last_scan_count is not None and last_scan_count == 0 and not has_any_in_db:
            return
        if fetch_err:
            return
        if has_any_in_db and selected_state_id is not None:
            st.info("No updates match the selected state filter.")
        elif not has_any_in_db:
            st.info(
                "No updates recorded yet. Click **Check for updates** to scan, "
                "or run the scheduled scraper."
            )
        else:
            st.info("No updates to display. Try increasing how many updates to show.")
        return

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

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
    section_heading(f"{len(deduped)} Update{'s' if len(deduped) != 1 else ''}")

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
            st.markdown(
                f'<div class="rc-update-card-header">'
                f'<span class="rc-update-card-title">{source_name}</span>'
                f'<span class="rc-badge {badge_cls}">{category}</span>'
                f'</div>'
                f'<div class="rc-update-card-meta">'
                f'<span>📍 {location_str}</span>'
                f'<span>📅 {date_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if update_summary or url:
                with st.expander("Details"):
                    if update_summary:
                        st.write(update_summary)
                    if url:
                        st.markdown(f"[View source ↗]({url})")

    cross_page_link("📧", "Want automatic notifications? Set up Email Alerts →", "pages/4_email_alerts.py")


show_page()
