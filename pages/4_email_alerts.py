from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from core.regulations.explorer import get_state_jurisdiction_options
from core.notifications.email_alerts import email_alerts
from ui_theme import apply_theme, log_activity, page_hero, section_heading


def show_page() -> None:
    apply_theme()
    page_hero("📧", "Email Alerts", "Subscribe to daily digests and get notified when regulations change in your jurisdictions.", "amber")

    with st.container(border=True):
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.75rem;">'
            '<span style="font-size:1.5rem;">📬</span>'
            '<div>'
            '<div style="font-weight:600;font-size:0.95rem;">Daily Regulatory Digest</div>'
            '<div style="font-size:0.82rem;color:var(--rc-text-muted);">'
            'Subscribers receive a daily email summarizing regulatory changes for their '
            'selected jurisdictions. Delivered at 8:00 AM EST on business days.'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)

    state_options = get_state_jurisdiction_options()
    state_names = [s["name"] for s in state_options]

    col_sub, col_unsub = st.columns(2)

    with col_sub:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.25rem;margin-bottom:0.25rem;">🔔</div>'
                '<div style="font-weight:600;font-size:0.95rem;">Subscribe to Alerts</div>'
                '<div style="font-size:0.78rem;color:var(--rc-text-muted);margin-bottom:0.75rem;">'
                'Get notified about regulatory changes</div>',
                unsafe_allow_html=True,
            )
            sub_email = st.text_input(
                "Email address",
                placeholder="you@company.com",
                key="sub_email",
            )
            sub_state_name = st.selectbox(
                "State",
                options=state_names if state_names else ["(No states)"],
                index=0,
                key="sub_state",
            )
            sub_state_id: Optional[int] = None
            if sub_state_name != "(No states)":
                sub_state_id = next(
                    int(s["id"]) for s in state_options if str(s["name"]) == sub_state_name
                )

            if st.button("Subscribe", use_container_width=True, type="primary"):
                if not sub_email.strip():
                    st.error("Please enter an email.")
                elif sub_state_id is None:
                    st.error("Please select a state.")
                else:
                    try:
                        with st.spinner("Subscribing..."):
                            email_alerts.subscribe(
                                email=sub_email.strip(), jurisdiction_id=sub_state_id
                            )
                        st.toast(f"Subscribed to {sub_state_name} alerts", icon="🔔")
                        log_activity("Subscribed to alerts", f"{sub_email.strip()} — {sub_state_name}")
                        st.success(
                            "Subscription saved. If this was a new or reactivated subscription, "
                            "a welcome email is sent when SMTP is configured (or saved under `emails/`)."
                        )
                    except PermissionError:
                        st.error(
                            "⚠️ **Database permission error.** The Supabase `anon` role cannot write to "
                            "`email_subscriptions`. Fix options:\n\n"
                            "1. Run the RLS policy SQL from **LOCAL_DEVELOPMENT.md § Step 6** in the Supabase SQL Editor.\n"
                            "2. Or set `SUPABASE_KEY` to the **service_role** key in `.env`."
                        )
                    except Exception:
                        st.error("Subscription could not be saved. Please check your database connection and try again.")

    with col_unsub:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.25rem;margin-bottom:0.25rem;">🔕</div>'
                '<div style="font-weight:600;font-size:0.95rem;">Unsubscribe</div>'
                '<div style="font-size:0.78rem;color:var(--rc-text-muted);margin-bottom:0.75rem;">'
                'Remove an existing subscription</div>',
                unsafe_allow_html=True,
            )
            unsub_email = st.text_input(
                "Email address",
                placeholder="you@company.com",
                key="unsub_email",
            )
            unsub_state_name = st.selectbox(
                "State",
                options=state_names if state_names else ["(No states)"],
                index=0,
                key="unsub_state",
            )
            unsub_state_id: Optional[int] = None
            if unsub_state_name != "(No states)":
                unsub_state_id = next(
                    int(s["id"]) for s in state_options if str(s["name"]) == unsub_state_name
                )

            if st.button("Unsubscribe", use_container_width=True):
                if not unsub_email.strip():
                    st.error("Please enter an email.")
                elif unsub_state_id is None:
                    st.error("Please select a state.")
                else:
                    try:
                        with st.spinner("Unsubscribing..."):
                            out = email_alerts.unsubscribe(
                                email=unsub_email.strip(),
                                jurisdiction_id=unsub_state_id,
                            )
                        if out.get("status") == "not_found":
                            st.warning(
                                "No active subscription was found for that email and state."
                            )
                        else:
                            st.toast("Unsubscribed successfully", icon="🔕")
                            log_activity("Unsubscribed from alerts", unsub_email.strip())
                            st.success(
                                "You have been unsubscribed. A confirmation email is sent when "
                                "SMTP is configured (or saved under `emails/`)."
                            )
                    except PermissionError:
                        st.error(
                            "This feature requires additional database setup. "
                            "Please contact your administrator."
                        )
                    except Exception:
                        st.error("Could not process your unsubscribe request. Please try again later.")

    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
    section_heading("Look Up Subscriptions")

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:1.25rem;margin-bottom:0.25rem;">👁️</div>'
            '<div style="font-weight:600;font-size:0.95rem;">View Subscriptions</div>'
            '<div style="font-size:0.78rem;color:var(--rc-text-muted);margin-bottom:0.75rem;">'
            'Look up existing alert subscriptions by email</div>',
            unsafe_allow_html=True,
        )
        col_input, col_action = st.columns([3, 1])
        with col_input:
            view_email = st.text_input(
                "Email to look up",
                placeholder="Enter email to look up",
                key="view_email",
                label_visibility="collapsed",
            )
        with col_action:
            lookup_clicked = st.button("Look up", key="load_subs", use_container_width=True)

        if lookup_clicked:
            if not view_email.strip():
                st.error("Please enter an email.")
            else:
                try:
                    with st.spinner("Looking up subscriptions..."):
                        subs = email_alerts.get_active_subscriptions(email=view_email.strip())
                    if not subs:
                        st.info("No active subscriptions found for this email.")
                    else:
                        for s in subs:
                            st.markdown(
                                f'<span class="rc-badge rc-badge-indigo" style="margin:0.125rem;">'
                                f'{s["jurisdiction_name"]}</span>',
                                unsafe_allow_html=True,
                            )
                except PermissionError:
                    st.error(
                        "This feature requires additional database setup. "
                        "Please contact your administrator."
                    )
                except Exception:
                    st.error("Could not look up subscriptions. Please check your database connection.")


show_page()
