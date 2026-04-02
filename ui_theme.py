"""Shared UI theme — consistent sidebar, responsive CSS, light/dark mode safe."""
from __future__ import annotations

import streamlit as st


_CSS_INJECTED_KEY = "_rc_css_injected"


def apply_theme() -> None:
    """Inject shared CSS + render the consistent sidebar. Call once at top of every page."""
    _inject_css()
    _render_sidebar()


def page_header(title: str, subtitle: str) -> None:
    """Render a page header with title and muted description."""
    st.markdown(f"### {title}")
    st.caption(subtitle)
    st.divider()


def section_heading(label: str) -> None:
    """Render a small uppercase section heading."""
    st.caption(label.upper())


def _render_sidebar() -> None:
    """Render the shared sidebar that appears identically on every page."""
    with st.sidebar:
        st.markdown("**RegComply**")
        st.caption("Compliance Intelligence")
        st.divider()

        st.caption("PLATFORM")
        st.page_link("pages/1_agent.py", label="Compliance Agent", icon="💬")
        st.page_link("pages/2_explorer.py", label="Explorer", icon="🔍")
        st.page_link("pages/3_update_log.py", label="Update Log", icon="📄")
        st.page_link("pages/4_email_alerts.py", label="Email Alerts", icon="📧")

        st.caption("ADMINISTRATION")
        st.page_link("pages/5_settings.py", label="Settings", icon="⚙️")

        st.divider()
        st.caption("🟢 All systems operational")


def _inject_css() -> None:
    """Inject minimal shared CSS once per render. No hardcoded colors."""
    if st.session_state.get(_CSS_INJECTED_KEY):
        return
    st.session_state[_CSS_INJECTED_KEY] = True
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


_GLOBAL_CSS = """
<style>
/* Hide default auto-generated multi-page nav (covers all Streamlit versions) */
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebarNavItems"] { display: none !important; }
nav[data-testid="stSidebarNav"] { display: none !important; }
ul[data-testid="stSidebarNavItems"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > ul { display: none !important; }
section[data-testid="stSidebar"] nav { display: none !important; }

/* Mobile: stack columns vertically below 640px */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        min-width: 85vw;
    }
}

/* Tighten metric spacing inside bordered containers */
div[data-testid="stMetric"] { padding: 0.25rem 0; }

/* Rounded expanders */
[data-testid="stExpander"] { border-radius: 10px; }

/*
 * Badge utilities — these use their own bg/fg so they look
 * correct in both light and dark mode.
 */
.rc-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.rc-badge-teal   { background: #ccfbf1; color: #0f766e; }
.rc-badge-amber  { background: #fef3c7; color: #92400e; }
.rc-badge-red    { background: #fee2e2; color: #991b1b; }
.rc-badge-blue   { background: #dbeafe; color: #1e40af; }
.rc-badge-slate  { background: #e2e8f0; color: #334155; }
.rc-badge-green  { background: #dcfce7; color: #166534; }
</style>
"""
