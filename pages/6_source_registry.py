from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import streamlit as st

from db.client import get_db
from ui_theme import apply_theme, page_header

_REGISTRY_AVAILABLE = False
try:
    from core.regulations.source_registry import source_registry

    _REGISTRY_AVAILABLE = True
except Exception:
    source_registry = None  # type: ignore[assignment]

PAGE_SIZE = 20


def _csv_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "seeds" / "sources.csv"


def _load_jurisdictions() -> list[dict[str, Any]]:
    if "sr_jurisdictions" not in st.session_state:
        try:
            db = get_db()
            res = db.table("jurisdictions").select("id,type,name,state_code").order("name").execute()
            st.session_state["sr_jurisdictions"] = res.data or []
        except Exception:
            st.session_state["sr_jurisdictions"] = []
    return st.session_state["sr_jurisdictions"]


def _jurisdiction_label(j: dict) -> str:
    jtype = j.get("type", "")
    name = j.get("name", "")
    code = j.get("state_code") or ""
    if jtype == "federal":
        return "[Federal] " + name
    if jtype == "state":
        return "[State] " + name + " (" + code + ")"
    suffix = " (" + code + ")" if code else ""
    return "[" + jtype.title() + "] " + name + suffix


def _find_jurisdiction_index(jurisdictions: list[dict], jid: int) -> int:
    for i, j in enumerate(jurisdictions):
        if j.get("id") == jid:
            return i
    return 0


# ---------------------------------------------------------------------------
# Unavailable
# ---------------------------------------------------------------------------

def _show_unavailable() -> None:
    st.title("Source Registry")
    st.info(
        "Source registry is not available. "
        "Run migration **007_regulation_sources.sql** and grant permissions "
        "(see `LOCAL_DEVELOPMENT.md` step 6) to enable it."
    )
    if st.button("Retry", key="btn_retry_avail"):
        st.rerun()


# ---------------------------------------------------------------------------
# Header: metrics row + provider toggle + import/export
# ---------------------------------------------------------------------------

def _section_header(all_sources: list[dict[str, Any]]) -> None:
    db_enabled = source_registry.is_db_registry_enabled()

    total = len(all_sources)
    active = sum(1 for s in all_sources if s.get("is_active"))
    errored = sum(1 for s in all_sources if s.get("last_error"))
    never_scraped = sum(1 for s in all_sources if not s.get("last_scraped_at"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total)
    m2.metric("Active", active)
    m3.metric("Errors", errored)
    m4.metric("Never scraped", never_scraped)

    st.markdown("---")

    col_toggle, col_import, col_export = st.columns([3, 1, 1])

    with col_toggle:
        provider_label = "Database" if db_enabled else "CSV file"
        st.markdown(
            "**Source provider:** `" + provider_label + "`"
            " — where the scraper reads source URLs from."
        )
        new_val = st.toggle("Use database source registry", value=db_enabled, key="sr_toggle")
        if new_val != db_enabled:
            source_registry.set_db_registry_enabled(new_val)
            st.rerun()

    with col_import:
        if st.button("Import CSV to DB", key="sr_backfill", use_container_width=True):
            with st.spinner("Importing CSV..."):
                try:
                    res = source_registry.backfill_from_csv(_csv_path())
                    st.success(
                        "Imported " + str(res["imported"])
                        + ", skipped " + str(res["skipped"])
                        + ", errors " + str(res["errors"])
                    )
                except Exception as exc:
                    st.error("Backfill failed: " + str(exc))

    with col_export:
        csv_data = source_registry.export_sources_csv()
        if csv_data:
            st.download_button(
                "Export CSV",
                data=csv_data,
                file_name="regulation_sources.csv",
                mime="text/csv",
                key="sr_export",
                use_container_width=True,
            )
        else:
            st.button("Export CSV", key="sr_export_empty", disabled=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _apply_filters(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    col_search, col_cat, col_status = st.columns([2, 1, 1])

    with col_search:
        query = st.text_input("Search", placeholder="Filter by name or URL...",
                              key="sr_search", label_visibility="collapsed")
    categories = sorted({s.get("category", "General") for s in sources})
    with col_cat:
        cat_filter = st.selectbox("Category", ["All"] + categories,
                                  key="sr_cat_filter", label_visibility="collapsed")
    with col_status:
        status_filter = st.selectbox("Status", ["All", "Active", "Inactive", "Has errors"],
                                     key="sr_status_filter", label_visibility="collapsed")

    filtered = list(sources)
    if query:
        q = query.lower()
        filtered = [s for s in filtered
                    if q in (s.get("source_name") or "").lower()
                    or q in (s.get("url") or "").lower()]
    if cat_filter != "All":
        filtered = [s for s in filtered if s.get("category") == cat_filter]
    if status_filter == "Active":
        filtered = [s for s in filtered if s.get("is_active")]
    elif status_filter == "Inactive":
        filtered = [s for s in filtered if not s.get("is_active")]
    elif status_filter == "Has errors":
        filtered = [s for s in filtered if s.get("last_error")]
    return filtered


# ---------------------------------------------------------------------------
# Pagination controls
# ---------------------------------------------------------------------------

def _pagination_bar(total: int) -> tuple[int, int]:
    """Render pagination and return (page_number_0_indexed, page_size)."""
    total_pages = max(1, math.ceil(total / PAGE_SIZE))

    if "sr_page" not in st.session_state:
        st.session_state["sr_page"] = 1

    current = st.session_state["sr_page"]
    if current > total_pages:
        current = total_pages
        st.session_state["sr_page"] = current

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", key="sr_prev", disabled=(current <= 1), use_container_width=True):
            st.session_state["sr_page"] = current - 1
            st.rerun()
    with col_info:
        st.markdown(
            "<div style='text-align:center; padding-top:6px;'>"
            "Page <b>" + str(current) + "</b> of <b>" + str(total_pages) +
            "</b> &nbsp;·&nbsp; " + str(total) + " source(s)"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next →", key="sr_next", disabled=(current >= total_pages), use_container_width=True):
            st.session_state["sr_page"] = current + 1
            st.rerun()

    return current - 1, PAGE_SIZE


# ---------------------------------------------------------------------------
# Edit dialog (session-state driven)
# ---------------------------------------------------------------------------

def _handle_edit_dialog(jurisdictions: list[dict[str, Any]]) -> None:
    """If sr_editing is set, show the edit form at the top of the sources tab."""
    editing_id = st.session_state.get("sr_editing")
    if not editing_id:
        return

    src = source_registry.get_source(editing_id)
    if not src:
        st.warning("Source not found.")
        st.session_state.pop("sr_editing", None)
        return

    st.info("Editing: **" + str(src.get("source_name", "")) + "**")
    with st.form("sr_edit_form"):
        ec1, ec2 = st.columns(2)
        with ec1:
            ed_name = st.text_input("Source name", value=src.get("source_name", ""))
            ed_url = st.text_input("URL", value=src.get("url", ""))
        with ec2:
            ed_category = st.text_input("Category", value=src.get("category", "General"))
            ed_domain = st.text_input("Domain", value=src.get("domain", "housing"))

        if jurisdictions:
            labels = [_jurisdiction_label(j) for j in jurisdictions]
            default_idx = _find_jurisdiction_index(jurisdictions, src.get("jurisdiction_id", 1))
            ed_jur_idx = st.selectbox("Jurisdiction", range(len(jurisdictions)),
                                       format_func=lambda i: labels[i],
                                       index=default_idx, key="sr_edit_jur")
            ed_jur_id = jurisdictions[ed_jur_idx]["id"]
        else:
            ed_jur_id = src.get("jurisdiction_id", 1)

        bc1, bc2 = st.columns(2)
        with bc1:
            save = st.form_submit_button("Save changes", use_container_width=True)
        with bc2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

    if cancel:
        st.session_state.pop("sr_editing", None)
        st.rerun()

    if save:
        if not ed_name or not ed_url:
            st.warning("Name and URL are required.")
            return
        with st.spinner("Saving changes..."):
            try:
                source_registry.update_source(editing_id, {
                    "source_name": ed_name.strip(),
                    "url": ed_url.strip(),
                    "category": ed_category.strip() or "General",
                    "domain": ed_domain.strip() or "housing",
                    "jurisdiction_id": ed_jur_id,
                })
                st.success("Updated: **" + ed_name.strip() + "**")
                st.session_state.pop("sr_editing", None)
                st.rerun()
            except Exception as exc:
                st.error("Update failed: " + str(exc))

    st.markdown("---")


# ---------------------------------------------------------------------------
# Scrape history for a source
# ---------------------------------------------------------------------------

def _show_scrape_history(url: str, source_id: int) -> None:
    """Show version history rows from the regulations table."""
    history = source_registry.scrape_history_for_url(url, limit=10)
    if not history:
        st.caption("No scrape history found for this URL.")
        return
    for h in history:
        ver = h.get("version", "?")
        ts = h.get("created_at", "?")
        hash_short = (h.get("content_hash") or "")[:12]
        current_badge = " (current)" if h.get("is_current") else ""
        st.caption("v" + str(ver) + " · " + str(ts) + " · hash: " + hash_short + current_badge)


# ---------------------------------------------------------------------------
# Source card
# ---------------------------------------------------------------------------

def _render_source_card(src: dict[str, Any]) -> None:
    sid = src["id"]
    active = src.get("is_active", True)
    name = src.get("source_name", "Unknown")
    url = src.get("url", "")
    category = src.get("category", "General")
    domain = src.get("domain", "housing")
    last_scraped = src.get("last_scraped_at")
    last_err = src.get("last_error")

    icon = "✅" if active else "⏸️"
    err_badge = " ⚠️" if last_err else ""
    header = icon + err_badge + "  " + name

    with st.expander(header, expanded=False):
        info_col, action_col = st.columns([3, 1])

        with info_col:
            st.markdown("**URL:** [" + url + "](" + url + ")")
            st.markdown("**Category:** " + category + "  ·  **Domain:** " + domain)
            if last_scraped:
                st.caption("Last scraped: " + str(last_scraped))
            else:
                st.caption("Never scraped")
            if last_err:
                st.error("Last error: " + last_err)

        with action_col:
            if st.button("Edit", key="sr_edit_" + str(sid), use_container_width=True):
                st.session_state["sr_editing"] = sid
                st.rerun()

            toggle_label = "Deactivate" if active else "Activate"
            if st.button(toggle_label, key="sr_tog_" + str(sid), use_container_width=True):
                with st.spinner("Updating..."):
                    source_registry.toggle_source_active(sid, not active)
                st.rerun()

            if st.button("Test URL", key="sr_test_" + str(sid), use_container_width=True):
                with st.spinner("Testing..."):
                    probe = source_registry.test_source(url)
                if probe.get("ok"):
                    ct = probe.get("content_type", "?")
                    cl = probe.get("content_length", 0)
                    st.success("OK " + str(probe["status_code"]) + " · " + ct + " · " + "{:,}".format(cl) + "B")
                else:
                    err_detail = probe.get("error") or ("HTTP " + str(probe.get("status_code")))
                    st.error("Failed: " + err_detail)

            if st.button("Delete", key="sr_del_" + str(sid), use_container_width=True):
                with st.spinner("Deleting..."):
                    source_registry.delete_source(sid)
                st.rerun()

        # Scrape history (collapsed sub-section)
        with st.expander("Scrape history", expanded=False):
            _show_scrape_history(url, sid)


# ---------------------------------------------------------------------------
# Sources tab
# ---------------------------------------------------------------------------

def _tab_sources(all_sources: list[dict[str, Any]]) -> None:
    jurisdictions = _load_jurisdictions()

    # Edit dialog at top if active
    _handle_edit_dialog(jurisdictions)

    # Filters
    filtered = _apply_filters(all_sources)

    # Bulk actions row
    ba1, ba2, ba3 = st.columns([1, 1, 4])
    with ba1:
        if st.button("Activate all shown", key="sr_bulk_act", use_container_width=True):
            with st.spinner("Activating sources..."):
                for s in filtered:
                    if not s.get("is_active"):
                        source_registry.toggle_source_active(s["id"], True)
            st.rerun()
    with ba2:
        if st.button("Deactivate all shown", key="sr_bulk_deact", use_container_width=True):
            with st.spinner("Deactivating sources..."):
                for s in filtered:
                    if s.get("is_active"):
                        source_registry.toggle_source_active(s["id"], False)
            st.rerun()

    if not filtered:
        st.info("No sources match your filters.")
        return

    # Pagination
    page_idx, page_size = _pagination_bar(len(filtered))
    start = page_idx * page_size
    page_sources = filtered[start : start + page_size]

    # Render cards
    for src in page_sources:
        _render_source_card(src)

    # Bottom pagination for long lists
    if len(filtered) > page_size:
        _pagination_bar_bottom(len(filtered), page_idx, page_size)


def _pagination_bar_bottom(total: int, page_idx: int, page_size: int) -> None:
    total_pages = max(1, math.ceil(total / page_size))
    current = page_idx + 1
    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", key="sr_prev_bot", disabled=(current <= 1), use_container_width=True):
            st.session_state["sr_page"] = current - 1
            st.rerun()
    with col_info:
        st.markdown(
            "<div style='text-align:center; padding-top:6px;'>"
            "Page <b>" + str(current) + "</b> of <b>" + str(total_pages) + "</b>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next →", key="sr_next_bot", disabled=(current >= total_pages), use_container_width=True):
            st.session_state["sr_page"] = current + 1
            st.rerun()


# ---------------------------------------------------------------------------
# Add Source tab
# ---------------------------------------------------------------------------

def _tab_add_source() -> None:
    jurisdictions = _load_jurisdictions()

    st.markdown("Fill in the details below to register a new regulation source URL.")

    with st.form("sr_add_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        with fc1:
            new_name = st.text_input("Source name *")
            new_url = st.text_input("URL *")
        with fc2:
            new_category = st.text_input("Category", value="General")
            new_domain = st.text_input("Domain", value="housing")

        if jurisdictions:
            labels = [_jurisdiction_label(j) for j in jurisdictions]
            sel_idx = st.selectbox("Jurisdiction *", range(len(jurisdictions)),
                                   format_func=lambda i: labels[i], key="sr_add_jur")
            jurisdiction_id = jurisdictions[sel_idx]["id"]
        else:
            st.warning("No jurisdictions loaded. Defaulting to ID 1.")
            jurisdiction_id = 1

        submitted = st.form_submit_button("Add source", use_container_width=True)

    if submitted:
        if not new_name or not new_url:
            st.warning("Source name and URL are required.")
            return
        with st.spinner("Adding source..."):
            try:
                source_registry.add_source({
                    "source_name": new_name.strip(),
                    "url": new_url.strip(),
                    "category": new_category.strip() or "General",
                    "domain": new_domain.strip() or "housing",
                    "jurisdiction_id": jurisdiction_id,
                    "is_active": True,
                })
                st.success("Added: **" + new_name.strip() + "**")
            except Exception as exc:
                st.error("Failed to add source: " + str(exc))


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

def show_page() -> None:
    apply_theme()
    page_header("Source Registry", "Manage regulation source URLs that the scraper monitors")

    if not _REGISTRY_AVAILABLE or source_registry is None:
        _show_unavailable()
        return

    if not source_registry.registry_table_exists():
        _show_unavailable()
        return

    all_sources = source_registry.list_sources()

    _section_header(all_sources)

    tab_sources, tab_add = st.tabs(["Sources", "➕ Add Source"])

    with tab_sources:
        _tab_sources(all_sources)

    with tab_add:
        _tab_add_source()


show_page()
