"""Shared UI theme — premium SaaS dashboard styling with consistent sidebar."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st


ACTIVITY_LOG_KEY = "rc_activity_log"


def log_activity(action: str, detail: str = "") -> None:
    """Append an entry to the session-wide activity log shown on the dashboard."""
    if ACTIVITY_LOG_KEY not in st.session_state:
        st.session_state[ACTIVITY_LOG_KEY] = []
    st.session_state[ACTIVITY_LOG_KEY].append(
        {"timestamp": datetime.now().strftime("%I:%M %p"), "action": action, "detail": detail},
    )
    if len(st.session_state[ACTIVITY_LOG_KEY]) > 30:
        st.session_state[ACTIVITY_LOG_KEY] = st.session_state[ACTIVITY_LOG_KEY][-30:]


def apply_theme() -> None:
    """Inject shared CSS + render the consistent sidebar. Call once at top of every page."""
    _inject_css()
    _render_sidebar()


def page_header(title: str, subtitle: str) -> None:
    """Render a premium page header with title and muted description."""
    st.markdown(
        f"""<div class="rc-page-header">
            <h1 class="rc-page-title">{title}</h1>
            <p class="rc-page-subtitle">{subtitle}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def page_hero(icon: str, title: str, subtitle: str, color: str = "indigo") -> None:
    """Render a compact gradient hero banner at the top of a page."""
    color_map = {
        "indigo": ("var(--rc-primary)", "rgba(79,70,229,0.12)", "var(--rc-primary-light)"),
        "blue": ("#2563eb", "rgba(37,99,235,0.12)", "#dbeafe"),
        "green": ("#059669", "rgba(5,150,105,0.12)", "#dcfce7"),
        "amber": ("#d97706", "rgba(217,119,6,0.12)", "#fef3c7"),
        "red": ("#dc2626", "rgba(220,38,38,0.12)", "#fee2e2"),
        "slate": ("#475569", "rgba(71,85,105,0.12)", "#f1f5f9"),
    }
    accent, bg_tint, icon_bg = color_map.get(color, color_map["indigo"])
    st.markdown(
        f'<div class="rc-page-hero" style="background:{bg_tint};">'
        f'<div class="rc-page-hero-icon" style="background:{icon_bg};color:{accent};">{icon}</div>'
        f'<div>'
        f'<div class="rc-page-hero-title">{title}</div>'
        f'<div class="rc-page-hero-subtitle">{subtitle}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_heading(label: str) -> None:
    """Render a small uppercase section heading with subtle left border."""
    st.markdown(
        f'<div class="rc-section-heading">{label}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = "") -> str:
    """Return HTML for a styled metric card. Caller wraps in st.markdown(unsafe_allow_html=True)."""
    icon_html = f'<span class="rc-metric-icon">{icon}</span>' if icon else ""
    return (
        f'<div class="rc-metric-card">'
        f'{icon_html}'
        f'<div class="rc-metric-value">{value}</div>'
        f'<div class="rc-metric-label">{label}</div>'
        f'</div>'
    )


def status_dot(ok: bool, label: str) -> str:
    """Return HTML for a status indicator dot + label."""
    cls = "rc-dot-green" if ok else "rc-dot-red"
    return f'<span class="rc-dot {cls}"></span><span class="rc-status-label">{label}</span>'


def skeleton_card(count: int = 4) -> str:
    """Return HTML for shimmer-animated skeleton placeholder cards."""
    cards = ""
    for _ in range(count):
        cards += (
            '<div class="rc-skeleton-card">'
            '<div class="rc-skeleton-line rc-skeleton-icon"></div>'
            '<div class="rc-skeleton-line rc-skeleton-value"></div>'
            '<div class="rc-skeleton-line rc-skeleton-label"></div>'
            '</div>'
        )
    return f'<div class="rc-skeleton-row">{cards}</div>'


def activity_timeline(entries: list[dict[str, Any]], max_items: int = 10) -> str:
    """Return HTML for a compact activity timeline."""
    if not entries:
        return (
            '<div class="rc-timeline-empty">'
            '<span style="font-size:1.25rem;">📋</span>'
            '<span>No activity yet this session. Start by asking a question or running a search.</span>'
            '</div>'
        )
    recent = entries[-max_items:][::-1]
    items = ""
    for e in recent:
        items += (
            f'<div class="rc-timeline-item">'
            f'<span class="rc-timeline-dot"></span>'
            f'<span class="rc-timeline-action">{e["action"]}</span>'
            f'<span class="rc-timeline-detail">{e.get("detail", "")}</span>'
            f'<span class="rc-timeline-time">{e["timestamp"]}</span>'
            f'</div>'
        )
    return f'<div class="rc-timeline">{items}</div>'


def setup_banner(steps: list[tuple[bool, str, str]]) -> None:
    """Render a getting-started checklist banner.

    Each tuple is (completed, label, description).
    Banner hides itself when all steps are complete.
    """
    if all(done for done, _, _ in steps):
        return
    items_html = ""
    for done, label, desc in steps:
        icon = '<span class="rc-setup-check done">&#10003;</span>' if done else '<span class="rc-setup-check">&#9675;</span>'
        cls = "rc-setup-step done" if done else "rc-setup-step"
        items_html += f'<div class="{cls}">{icon}<div><strong>{label}</strong><div class="rc-setup-desc">{desc}</div></div></div>'
    st.markdown(
        f'<div class="rc-setup-banner">'
        f'<div class="rc-setup-header">Getting Started</div>'
        f'{items_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def callout_box(icon: str, message: str, style: str = "info") -> None:
    """Render a styled inline callout (info, warning, or tip)."""
    color_map = {"info": "#e0e7ff;#3730a3", "warning": "#fef3c7;#92400e", "tip": "#dcfce7;#166534"}
    bg, fg = color_map.get(style, color_map["info"]).split(";")
    st.markdown(
        f'<div class="rc-callout" style="background:{bg};color:{fg};">'
        f'<span style="font-size:1.1rem;margin-right:0.5rem;">{icon}</span>{message}</div>',
        unsafe_allow_html=True,
    )


def cross_page_link(icon: str, label: str, page_path: str) -> None:
    """Render a subtle cross-page navigation link."""
    st.markdown('<div style="height:0.25rem;"></div>', unsafe_allow_html=True)
    st.page_link(page_path, label=f"{icon} {label}")


def _render_sidebar() -> None:
    """Render brand header and footer in the sidebar. Navigation links come from st.navigation() in app.py."""
    with st.sidebar:
        st.markdown(
            '<div class="rc-sidebar-brand">'
            '<span class="rc-brand-icon">🏢</span>'
            '<span class="rc-brand-text">MFH Comply</span>'
            '</div>'
            '<div class="rc-brand-tagline">Multi-Family Housing Compliance</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="rc-sidebar-spacer"></div>', unsafe_allow_html=True)


def _inject_css() -> None:
    """Inject premium shared CSS on every page render.

    Streamlit re-renders the full page on each navigation, so CSS must
    be injected every time — session_state flags do NOT survive the DOM swap.
    """
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ─── Design Tokens ─────────────────────────────────── */

:root {
    --rc-primary: #4f46e5;
    --rc-primary-hover: #4338ca;
    --rc-primary-light: #e0e7ff;
    --rc-primary-glow: rgba(79, 70, 229, 0.3);
    --rc-secondary: #0f172a;
    --rc-secondary-light: #1e293b;
    --rc-bg: #f8fafc;
    --rc-surface: #ffffff;
    --rc-accent: #10b981;
    --rc-warning: #f59e0b;
    --rc-danger: #ef4444;
    --rc-text: #0f172a;
    --rc-text-muted: #64748b;
    --rc-text-faint: #94a3b8;
    --rc-border: #e2e8f0;
    --rc-border-light: rgba(0,0,0,0.06);
    --rc-shadow-sm: 0 1px 3px rgba(0,0,0,0.04);
    --rc-shadow-md: 0 4px 12px rgba(0,0,0,0.08);
    --rc-shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
    --rc-radius-sm: 8px;
    --rc-radius-md: 12px;
    --rc-radius-lg: 16px;
    --rc-radius-full: 9999px;
}

/* ─── Keyframe Animations ───────────────────────────── */

@keyframes rc-shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes rc-fadeSlideUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes rc-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.5; }
}

@keyframes rc-scaleIn {
    from { opacity: 0; transform: scale(0.95); }
    to   { opacity: 1; transform: scale(1); }
}

/* ─── Base & Typography ─────────────────────────────── */

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--rc-bg);
}

/* ─── Sidebar ───────────────────────────────────────── */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--rc-secondary) 0%, var(--rc-secondary-light) 100%) !important;
}

section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

section[data-testid="stSidebar"] [data-baseweb="tab"],
section[data-testid="stSidebar"] [data-baseweb="tab-list"],
section[data-testid="stSidebar"] [data-baseweb="tab-border"],
section[data-testid="stSidebar"] [data-baseweb="tab-highlight"] {
    display: none !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] span {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    color: #475569 !important;
    text-transform: uppercase !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
    border-radius: var(--rc-radius-sm) !important;
    padding: 0.5rem 0.75rem !important;
    margin: 2px 0 !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: rgba(79, 70, 229, 0.2) !important;
    border-left: 3px solid #818cf8 !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] * {
    color: var(--rc-primary-light) !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
    border-radius: var(--rc-radius-sm) !important;
    padding: 0.5rem 0.75rem !important;
    margin: 2px 0 !important;
    transition: all 0.15s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {
    background: rgba(255, 255, 255, 0.08) !important;
}

.rc-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0 0.125rem 0;
}

.rc-brand-icon { font-size: 1.5rem; }

.rc-brand-text {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    letter-spacing: -0.02em;
}

.rc-brand-tagline {
    font-size: 0.75rem;
    color: var(--rc-text-muted) !important;
    letter-spacing: 0.02em;
    margin-top: -0.25rem;
    padding-bottom: 0.5rem;
}

.rc-sidebar-spacer {
    height: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 0.75rem;
}

.rc-nav-section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #475569 !important;
    padding: 0.25rem 0.75rem;
    margin-bottom: 0.125rem;
}

/* ─── Page Header ───────────────────────────────────── */

.rc-page-header {
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--rc-border-light);
    margin-bottom: 1.75rem;
}

.rc-page-title {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    color: var(--rc-text);
    margin: 0 0 0.25rem 0 !important;
    line-height: 1.2 !important;
}

.rc-page-subtitle {
    font-size: 0.9rem;
    color: var(--rc-text-muted);
    margin: 0;
    font-weight: 400;
}

/* ─── Page Hero Banner ─────────────────────────────── */

.rc-page-hero {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1.5rem 1.75rem;
    border-radius: var(--rc-radius-lg);
    margin-bottom: 1.75rem;
    animation: rc-fadeSlideUp 0.35s ease-out;
}

.rc-page-hero-icon {
    width: 52px;
    height: 52px;
    border-radius: var(--rc-radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.rc-page-hero-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--rc-text);
    line-height: 1.2;
    margin-bottom: 0.2rem;
}

.rc-page-hero-subtitle {
    font-size: 0.88rem;
    color: var(--rc-text-muted);
    line-height: 1.45;
}

/* ─── Section Heading ───────────────────────────────── */

.rc-section-heading {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--rc-text-muted);
    text-transform: uppercase;
    padding: 0.75rem 0 0.5rem 0;
    margin-top: 0.5rem;
    border-left: 3px solid #818cf8;
    padding-left: 0.75rem;
}

/* ─── Metric Cards ──────────────────────────────────── */

.rc-metric-card {
    background: linear-gradient(135deg, var(--rc-bg) 0%, #f1f5f9 100%);
    border: 1px solid var(--rc-border-light);
    border-radius: var(--rc-radius-md);
    padding: 1.25rem;
    text-align: center;
    transition: all 0.2s ease;
    box-shadow: var(--rc-shadow-sm);
}

.rc-metric-card:hover {
    box-shadow: var(--rc-shadow-md);
    transform: translateY(-2px);
    border-color: var(--rc-primary-light);
}

.rc-metric-icon {
    font-size: 1.5rem;
    display: block;
    margin-bottom: 0.5rem;
}

.rc-metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--rc-text);
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.rc-metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--rc-text-muted);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ─── Cards & Containers ────────────────────────────── */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--rc-radius-md) !important;
    border-color: var(--rc-border-light) !important;
    box-shadow: var(--rc-shadow-sm) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--rc-shadow-md) !important;
    border-color: var(--rc-primary-light) !important;
}

/* ─── Buttons ───────────────────────────────────────── */

.stButton > button {
    border-radius: var(--rc-radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.01em !important;
    border: 1px solid var(--rc-border-light) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--rc-shadow-md) !important;
}

.stButton > button:active {
    transform: scale(0.97) !important;
    box-shadow: var(--rc-shadow-sm) !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1 0%, var(--rc-primary) 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 1px 3px var(--rc-primary-glow) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 16px var(--rc-primary-glow) !important;
    background: linear-gradient(135deg, var(--rc-primary) 0%, var(--rc-primary-hover) 100%) !important;
}

/* ─── Inputs ────────────────────────────────────────── */

.stTextInput > div > div > input,
.stSelectbox > div > div,
[data-testid="stTextInput"] input {
    border-radius: var(--rc-radius-sm) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    transition: all 0.15s ease !important;
}

.stTextInput > div > div > input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
}

/* ─── Chat messages ─────────────────────────────────── */

[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.5rem !important;
    animation: rc-fadeSlideUp 0.3s ease-out;
}

/* ─── Expanders ─────────────────────────────────────── */

[data-testid="stExpander"] {
    border-radius: var(--rc-radius-md) !important;
    border: 1px solid var(--rc-border-light) !important;
    box-shadow: var(--rc-shadow-sm) !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: background 0.15s ease !important;
}

/* ─── Tabs ──────────────────────────────────────────── */

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    border-bottom: 2px solid var(--rc-border-light) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--rc-radius-sm) var(--rc-radius-sm) 0 0 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
}

.stTabs [aria-selected="true"] {
    border-bottom-color: var(--rc-primary) !important;
}

/* ─── Alerts ────────────────────────────────────────── */

.stAlert, [data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ─── Dataframes ────────────────────────────────────── */

[data-testid="stDataFrame"] {
    border-radius: var(--rc-radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--rc-shadow-sm) !important;
}

/* ─── Dividers ──────────────────────────────────────── */

hr {
    border-color: var(--rc-border-light) !important;
    margin: 1.5rem 0 !important;
}

/* ─── File uploader ─────────────────────────────────── */

[data-testid="stFileUploader"] {
    border-radius: var(--rc-radius-md) !important;
}

[data-testid="stFileUploader"] section {
    border-radius: var(--rc-radius-md) !important;
    border: 2px dashed rgba(0,0,0,0.1) !important;
    padding: 1.25rem !important;
}

/* ─── Metrics (Streamlit built-in) ──────────────────── */

div[data-testid="stMetric"] {
    padding: 0.25rem 0;
}

div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: var(--rc-text-muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ─── Badge utilities ───────────────────────────────── */

.rc-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: var(--rc-radius-full);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.rc-badge-teal   { background: #ccfbf1; color: #0f766e; }
.rc-badge-amber  { background: #fef3c7; color: #92400e; }
.rc-badge-red    { background: #fee2e2; color: #991b1b; }
.rc-badge-blue   { background: #dbeafe; color: #1e40af; }
.rc-badge-slate  { background: var(--rc-border); color: #334155; }
.rc-badge-green  { background: #dcfce7; color: #166534; }
.rc-badge-indigo { background: var(--rc-primary-light); color: #3730a3; }

/* ─── Status dots ───────────────────────────────────── */

.rc-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.375rem;
    vertical-align: middle;
}
.rc-dot-green {
    background: var(--rc-accent);
    box-shadow: 0 0 6px rgba(16,185,129,0.4);
    animation: rc-pulse 2s ease-in-out infinite;
}
.rc-dot-red {
    background: var(--rc-danger);
    box-shadow: 0 0 6px rgba(239,68,68,0.4);
}
.rc-dot-amber {
    background: var(--rc-warning);
    box-shadow: 0 0 6px rgba(245,158,11,0.4);
}

.rc-status-label {
    font-size: 0.8rem;
    font-weight: 500;
    vertical-align: middle;
}

/* ─── Agent welcome hero ────────────────────────────── */

.rc-hero {
    text-align: center;
    padding: 2.5rem 1.5rem 1rem 1.5rem;
    animation: rc-fadeSlideUp 0.4s ease-out;
}

.rc-hero-logo {
    width: 56px;
    height: 56px;
    border-radius: var(--rc-radius-lg);
    background: linear-gradient(135deg, #6366f1 0%, var(--rc-primary) 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px var(--rc-primary-glow);
}

.rc-hero-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--rc-text);
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}

.rc-hero-subtitle {
    font-size: 0.95rem;
    color: var(--rc-text-muted);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.55;
}

.rc-hero-capabilities {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}

.rc-hero-cap {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--rc-primary);
}

.rc-hero-cap-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--rc-primary);
    flex-shrink: 0;
}

/* ─── Prompt suggestion cards ───────────────────────── */

.rc-cards-heading {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--rc-text-faint);
    text-transform: uppercase;
    text-align: center;
    margin: 1.25rem 0 0.75rem 0;
}

.rc-prompt-card-v2 {
    position: relative;
    padding: 1.25rem;
    min-height: 140px;
    display: flex;
    flex-direction: column;
}

.rc-prompt-card-v2-icon-wrap {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    margin-bottom: 0.75rem;
    flex-shrink: 0;
}

.rc-prompt-card-v2-icon-wrap.blue   { background: #dbeafe; }
.rc-prompt-card-v2-icon-wrap.green  { background: #dcfce7; }
.rc-prompt-card-v2-icon-wrap.amber  { background: #fef3c7; }

.rc-prompt-card-v2-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--rc-text);
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}

.rc-prompt-card-v2-desc {
    font-size: 0.8rem;
    color: var(--rc-text-muted);
    line-height: 1.45;
    margin-bottom: 0.5rem;
}

.rc-prompt-card-v2-example {
    font-size: 0.75rem;
    color: var(--rc-text-faint);
    font-style: italic;
    margin-top: auto;
    padding-top: 0.5rem;
    border-top: 1px solid var(--rc-border-light);
    line-height: 1.4;
}

/* ─── Quick question pills ──────────────────────────── */

.rc-pills-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
    padding: 0 1rem;
}

/* ─── Chat input bar (ChatGPT-style) ────────────────── */

[data-testid="stChatInput"] {
    border-radius: var(--rc-radius-lg) !important;
    border: 1.5px solid rgba(0,0,0,0.1) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease !important;
    padding: 0.25rem !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.12), 0 0 0 3px rgba(129, 140, 248, 0.1) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

[data-testid="stChatInput"] button {
    border-radius: 10px !important;
    background: linear-gradient(135deg, #6366f1 0%, var(--rc-primary) 100%) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, var(--rc-primary) 0%, var(--rc-primary-hover) 100%) !important;
    box-shadow: 0 2px 8px var(--rc-primary-glow) !important;
}

.rc-input-hint {
    text-align: center;
    font-size: 0.72rem;
    color: var(--rc-text-faint);
    margin-top: 0.5rem;
    line-height: 1.4;
}

/* ─── Empty state (generic) ─────────────────────────── */

.rc-empty-state {
    text-align: center;
    padding: 3rem 1.5rem 2rem 1.5rem;
    animation: rc-fadeSlideUp 0.4s ease-out;
}

.rc-empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.8;
}

.rc-empty-state-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--rc-text);
    margin-bottom: 0.5rem;
}

.rc-empty-state-desc {
    font-size: 0.9rem;
    color: var(--rc-text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ─── Skeleton loading ──────────────────────────────── */

.rc-skeleton-row {
    display: flex;
    gap: 1rem;
}

.rc-skeleton-card {
    flex: 1;
    background: var(--rc-surface);
    border: 1px solid var(--rc-border-light);
    border-radius: var(--rc-radius-md);
    padding: 1.25rem;
    text-align: center;
}

.rc-skeleton-line {
    background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
    background-size: 200% 100%;
    animation: rc-shimmer 1.5s ease-in-out infinite;
    border-radius: 6px;
    margin: 0 auto;
}

.rc-skeleton-icon  { width: 32px; height: 32px; border-radius: 8px; margin-bottom: 0.5rem; }
.rc-skeleton-value { width: 60%; height: 24px; margin-bottom: 0.5rem; }
.rc-skeleton-label { width: 80%; height: 12px; }
.rc-skeleton-text  { width: 100%; height: 14px; margin-bottom: 0.5rem; }

/* ─── Activity Timeline ─────────────────────────────── */

.rc-timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.rc-timeline-item {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--rc-border-light);
    font-size: 0.82rem;
    animation: rc-fadeSlideUp 0.25s ease-out;
}

.rc-timeline-item:last-child { border-bottom: none; }

.rc-timeline-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--rc-primary);
    flex-shrink: 0;
}

.rc-timeline-action {
    font-weight: 600;
    color: var(--rc-text);
    white-space: nowrap;
}

.rc-timeline-detail {
    color: var(--rc-text-muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.rc-timeline-time {
    color: var(--rc-text-faint);
    font-size: 0.72rem;
    white-space: nowrap;
    margin-left: auto;
}

.rc-timeline-empty {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1.25rem;
    color: var(--rc-text-muted);
    font-size: 0.85rem;
}

/* ─── Landing Hero (Home page) ──────────────────────── */

.rc-landing-hero {
    position: relative;
    background: url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1600&q=80') center/cover no-repeat;
    border-radius: var(--rc-radius-lg);
    overflow: hidden;
    margin-bottom: 2rem;
    animation: rc-fadeSlideUp 0.5s ease-out;
}

.rc-landing-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(15,23,42,0.88) 0%, rgba(30,41,59,0.82) 50%, rgba(79,70,229,0.55) 100%);
    z-index: 1;
}

.rc-landing-hero-content {
    position: relative;
    z-index: 2;
    padding: 3.5rem 3rem 3rem 3rem;
    color: #ffffff;
}

.rc-landing-hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(224,231,255,0.8);
    margin-bottom: 0.75rem;
}

.rc-landing-hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 0 0 0.75rem 0;
    color: #ffffff;
}

.rc-landing-hero-subtitle {
    font-size: 1.05rem;
    line-height: 1.6;
    color: rgba(203,213,225,0.95);
    max-width: 640px;
    margin: 0 0 1.5rem 0;
}

.rc-landing-hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}

.rc-landing-stat {
    display: flex;
    flex-direction: column;
}

.rc-landing-stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
}

.rc-landing-stat-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(203,213,225,0.7);
    margin-top: 0.125rem;
}

/* ─── Audience Cards (Home page) ────────────────────── */

.rc-audience-card {
    display: flex;
    align-items: flex-start;
    gap: 0.875rem;
    padding: 1rem;
}

.rc-audience-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
}

.rc-audience-icon.indigo { background: var(--rc-primary-light); }
.rc-audience-icon.green  { background: #dcfce7; }
.rc-audience-icon.blue   { background: #dbeafe; }
.rc-audience-icon.amber  { background: #fef3c7; }

.rc-audience-label {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--rc-text);
    margin-bottom: 0.15rem;
}

.rc-audience-desc {
    font-size: 0.78rem;
    color: var(--rc-text-muted);
    line-height: 1.45;
}

/* ─── Feature Highlights (Home page) ────────────────── */

.rc-feature-item {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    padding: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--rc-text);
}

.rc-feature-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--rc-primary);
    flex-shrink: 0;
}

/* ─── Dashboard greeting ────────────────────────────── */

.rc-greeting {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.25rem 0 1.5rem 0;
}

.rc-greeting-text {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--rc-text);
    letter-spacing: -0.03em;
}

.rc-greeting-date {
    font-size: 0.82rem;
    color: var(--rc-text-muted);
}

/* ─── Dashboard Quick Actions ───────────────────────── */

.rc-action-card {
    text-align: center;
    padding: 1.25rem 0.75rem;
    transition: all 0.2s ease;
}

.rc-action-card:hover {
    transform: scale(1.02);
}

.rc-action-icon {
    width: 44px;
    height: 44px;
    border-radius: var(--rc-radius-md);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    margin-bottom: 0.75rem;
}

.rc-action-icon.indigo { background: var(--rc-primary-light); }
.rc-action-icon.green  { background: #dcfce7; }
.rc-action-icon.blue   { background: #dbeafe; }
.rc-action-icon.amber  { background: #fef3c7; }

.rc-action-label {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--rc-text);
    margin-bottom: 0.25rem;
}

.rc-action-desc {
    font-size: 0.75rem;
    color: var(--rc-text-muted);
    line-height: 1.4;
}

/* ─── System Health Strip ───────────────────────────── */

.rc-health-strip {
    display: flex;
    gap: 1.5rem;
    padding: 0.75rem 1rem;
    background: var(--rc-bg);
    border: 1px solid var(--rc-border-light);
    border-radius: var(--rc-radius-md);
    flex-wrap: wrap;
}

.rc-health-chip {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--rc-text-muted);
}

/* ─── Update log cards ──────────────────────────────── */

.rc-update-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.25rem;
}

.rc-update-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--rc-text);
}

.rc-update-card-meta {
    font-size: 0.78rem;
    color: var(--rc-text-faint);
    display: flex;
    gap: 0.75rem;
    align-items: center;
}

/* ─── Settings status grid ──────────────────────────── */

.rc-status-card {
    text-align: center;
    padding: 1rem 0.5rem;
}

.rc-status-card-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--rc-text);
    margin-bottom: 0.5rem;
}

/* ─── Setup / Onboarding banner ─────────────────────── */

.rc-setup-banner {
    background: linear-gradient(135deg, #eef2ff 0%, var(--rc-primary-light) 100%);
    border: 1px solid #c7d2fe;
    border-radius: var(--rc-radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    animation: rc-scaleIn 0.3s ease-out;
}

.rc-setup-header {
    font-size: 0.9rem;
    font-weight: 700;
    color: #312e81;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}

.rc-setup-step {
    display: flex;
    align-items: flex-start;
    gap: 0.625rem;
    padding: 0.375rem 0;
    font-size: 0.85rem;
    color: #3730a3;
}

.rc-setup-step.done { opacity: 0.5; }

.rc-setup-check {
    font-size: 1rem;
    min-width: 1.25rem;
    text-align: center;
    line-height: 1.3;
    color: var(--rc-primary);
}

.rc-setup-check.done {
    color: var(--rc-accent);
    font-weight: 700;
}

.rc-setup-desc {
    font-size: 0.78rem;
    color: #6366f1;
    margin-top: 0.125rem;
    font-weight: 400;
}

/* ─── Inline callout ────────────────────────────────── */

.rc-callout {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.5rem 0;
    line-height: 1.4;
}

/* ─── Chip / tag buttons (Explorer) ─────────────────── */

.rc-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 0.75rem 0;
}

.rc-chip {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: var(--rc-radius-full);
    font-size: 0.78rem;
    font-weight: 500;
    background: #f1f5f9;
    color: #475569;
    border: 1px solid var(--rc-border);
    cursor: pointer;
    transition: all 0.15s ease;
}

.rc-chip:hover {
    background: var(--rc-primary-light);
    color: #3730a3;
    border-color: #c7d2fe;
}

/* ─── Confirmation danger button ────────────────────── */

.rc-btn-danger > button {
    background: #fef2f2 !important;
    color: #991b1b !important;
    border: 1px solid #fecaca !important;
}

.rc-btn-danger > button:hover {
    background: #fee2e2 !important;
    box-shadow: 0 2px 8px rgba(239,68,68,0.15) !important;
}

/* ─── Mobile responsive ─────────────────────────────── */

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
    .rc-page-title { font-size: 1.375rem !important; }
    .rc-greeting-text { font-size: 1.25rem; }
    .rc-health-strip { flex-direction: column; gap: 0.5rem; }
    .rc-skeleton-row { flex-direction: column; }
    .rc-landing-hero-content { padding: 2rem 1.5rem; }
    .rc-landing-hero-title { font-size: 1.75rem; }
    .rc-landing-hero-stats { gap: 1.5rem; }
    .rc-landing-stat-value { font-size: 1.35rem; }
}

/* ─── Dark mode overrides ───────────────────────────── */

@media (prefers-color-scheme: dark) {
    :root {
        --rc-bg: #0f172a;
        --rc-surface: #1e293b;
        --rc-text: #f1f5f9;
        --rc-text-muted: #94a3b8;
        --rc-text-faint: #64748b;
        --rc-border: #334155;
        --rc-border-light: rgba(255,255,255,0.06);
        --rc-shadow-sm: 0 1px 3px rgba(0,0,0,0.2);
        --rc-shadow-md: 0 4px 12px rgba(0,0,0,0.3);
        --rc-shadow-lg: 0 8px 24px rgba(0,0,0,0.4);
    }
    .rc-metric-card {
        background: linear-gradient(135deg, var(--rc-secondary-light) 0%, var(--rc-secondary) 100%);
    }
    .rc-prompt-card-v2-example { border-top-color: rgba(255,255,255,0.06); }
    .rc-setup-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-color: var(--rc-primary-hover);
    }
    .rc-setup-header { color: #c7d2fe; }
    .rc-setup-step { color: #a5b4fc; }
    .rc-setup-desc { color: #818cf8; }
    .rc-callout { background: var(--rc-secondary-light) !important; }
    .rc-chip { background: var(--rc-secondary-light); color: var(--rc-text-muted); border-color: #334155; }
    .rc-chip:hover { background: #312e81; color: #c7d2fe; border-color: var(--rc-primary-hover); }
    .rc-skeleton-line {
        background: linear-gradient(90deg, #1e293b 25%, #334155 50%, #1e293b 75%);
        background-size: 200% 100%;
    }
}
</style>
"""
