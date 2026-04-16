from __future__ import annotations

from datetime import datetime

import streamlit as st

import config
from core.regulations.scraper import is_supabase_connected, scraper
from ui_theme import (
    ACTIVITY_LOG_KEY,
    activity_timeline,
    apply_theme,
    section_heading,
    status_dot,
)


def _get_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _get_stats() -> dict:
    """Gather platform-wide stats safely."""
    stats: dict = {
        "db_ok": False,
        "llm_ok": False,
        "llm_provider": "None",
        "smtp_ok": False,
        "total_regs": 0,
        "total_indexed": 0,
        "total_states": 0,
    }
    try:
        stats["db_ok"] = is_supabase_connected()
    except Exception:  # noqa: BLE001
        pass

    if config.settings.has_anthropic_key:
        stats["llm_ok"] = True
        stats["llm_provider"] = "Anthropic"
    elif config.settings.has_openai_key:
        stats["llm_ok"] = True
        stats["llm_provider"] = "OpenAI"
    elif config.settings.has_google_key:
        stats["llm_ok"] = True
        stats["llm_provider"] = "Google"

    stats["smtp_ok"] = config.settings.has_smtp

    if stats["db_ok"]:
        try:
            from core.regulations.explorer import get_explorer_metrics
            m = get_explorer_metrics()
            stats["total_regs"] = m.get("total_regulations", 0)
            stats["total_states"] = m.get("total_states_covered", 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            idx_rows = scraper.get_indexing_status()
            stats["total_indexed"] = sum(r.get("indexed", 0) for r in idx_rows) if idx_rows else 0
        except Exception:  # noqa: BLE001
            pass

    return stats


def show_page() -> None:
    apply_theme()

    stats = _get_stats()
    # ── Hero Banner with building background ──
    st.markdown(
        '<div class="rc-landing-hero">'
        '<div class="rc-landing-hero-content">'
        '<div class="rc-landing-hero-eyebrow">AI-Powered Compliance Intelligence</div>'
        '<div class="rc-landing-hero-title">MULTI-FAMILY HOUSING</div>'
        '<div class="rc-landing-hero-subtitle">'
        'The all-in-one platform for multi-family housing regulation compliance. '
        'Search regulations, review lease documents against state and local laws, '
        'track regulatory changes, and get AI-powered answers to complex compliance '
        'questions — so you can manage properties with confidence.'
        '</div>'
        '<div class="rc-landing-hero-stats">'
        '<div class="rc-landing-stat">'
        '<span class="rc-landing-stat-value">5</span>'
        '<span class="rc-landing-stat-label">States Covered</span>'
        '</div>'
        '<div class="rc-landing-stat">'
        '<span class="rc-landing-stat-value">24/7</span>'
        '<span class="rc-landing-stat-label">AI Availability</span>'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Who Is This For ──
    section_heading("Who Is This For")

    audiences = [
        (
            "🏢", "indigo",
            "Property Managers & Landlords",
            "Ensure your leases, notices, and policies comply with the latest housing "
            "regulations across every jurisdiction where you operate.",
        ),
        (
            "⚖️", "green",
            "Real Estate Attorneys",
            "Quickly research multi-state housing laws, verify lease clause compliance, "
            "and stay ahead of regulatory changes that affect your clients.",
        ),
        (
            "🏗️", "blue",
            "Housing Developers",
            "Understand zoning, tenant protection, and affordability requirements "
            "before breaking ground or converting existing properties.",
        ),
        (
            "📋", "amber",
            "Compliance Officers",
            "Monitor regulatory updates across your portfolio's jurisdictions and "
            "receive alerts when laws change that impact your operations.",
        ),
    ]

    audience_cols = st.columns(4)
    for i, (icon, color, label, desc) in enumerate(audiences):
        with audience_cols[i]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="rc-audience-card">'
                    f'<div class="rc-audience-icon {color}">{icon}</div>'
                    f'<div>'
                    f'<div class="rc-audience-label">{label}</div>'
                    f'<div class="rc-audience-desc">{desc}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    # ── Platform Capabilities + Quick Actions side by side ──
    col_features, col_actions = st.columns([1, 1])

    with col_features:
        section_heading("What You Can Do")
        with st.container(border=True):
            features = [
                "Ask AI about rent control, eviction rules, tenant rights, and more",
                "Upload lease documents for automated compliance review",
                "Search indexed regulations by keyword, state, or category",
                "Monitor regulatory changes and get email alerts",
                "Compare requirements across multiple jurisdictions",
                "Get citation-backed answers with source links",
            ]
            html = ""
            for f in features:
                html += (
                    f'<div class="rc-feature-item">'
                    f'<span class="rc-feature-dot"></span>'
                    f'<span>{f}</span>'
                    f'</div>'
                )
            st.markdown(html, unsafe_allow_html=True)

    with col_actions:
        section_heading("Get Started")

        actions = [
            ("💬", "Ask a Question", "Chat with the compliance agent", "pages/1_agent.py"),
            ("📄", "Review a Document", "Check a lease for compliance issues", "pages/1_agent.py"),
            ("🔍", "Search Regulations", "Browse indexed housing regulations", "pages/2_explorer.py"),
            ("📋", "Check for Updates", "Scan for regulatory changes", "pages/3_update_log.py"),
        ]

        for icon, label, desc, path in actions:
            with st.container(border=True):
                c_info, c_link = st.columns([3, 1])
                with c_info:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.25rem 0;">'
                        f'<span style="font-size:1.35rem;">{icon}</span>'
                        f'<div>'
                        f'<div style="font-weight:600;font-size:0.88rem;color:var(--rc-text);">{label}</div>'
                        f'<div style="font-size:0.75rem;color:var(--rc-text-muted);">{desc}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                with c_link:
                    st.page_link(path, label="Open →", use_container_width=True)

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    # ── Platform Status + Recent Activity ──
    col_activity, col_health = st.columns([3, 1])

    now = datetime.now()
    greeting = _get_greeting()
    date_str = now.strftime("%A, %B %d, %Y")

    with col_activity:
        section_heading("Recent Activity")
        st.markdown(
            f'<div style="font-size:0.82rem;color:var(--rc-text-muted);margin-bottom:0.5rem;">'
            f'{greeting} &mdash; {date_str}</div>',
            unsafe_allow_html=True,
        )
        activity_entries = st.session_state.get(ACTIVITY_LOG_KEY, [])
        with st.container(border=True):
            st.markdown(activity_timeline(activity_entries), unsafe_allow_html=True)

    with col_health:
        section_heading("System Health")
        with st.container(border=True):
            st.markdown(
                f'<div style="padding:0.5rem 0.25rem;">'
                f'<div class="rc-health-chip" style="margin-bottom:0.75rem;">'
                f'{status_dot(stats["db_ok"], "Database " + ("connected" if stats["db_ok"] else "offline"))}'
                f'</div>'
                f'<div class="rc-health-chip" style="margin-bottom:0.75rem;">'
                f'{status_dot(stats["llm_ok"], stats["llm_provider"] + (" ready" if stats["llm_ok"] else ""))}'
                f'</div>'
                f'<div class="rc-health-chip">'
                f'{status_dot(stats["smtp_ok"], "SMTP " + ("configured" if stats["smtp_ok"] else "not set"))}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-size:0.72rem;color:var(--rc-text-faint);line-height:1.4;">'
        f'{config.LEGAL_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


show_page()
