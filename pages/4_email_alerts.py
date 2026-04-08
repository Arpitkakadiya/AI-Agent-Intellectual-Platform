from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from core.regulations.explorer import get_state_jurisdiction_options
from core.notifications.email_alerts import email_alerts
from ui_theme import apply_theme, page_header


def show_page() -> None:
    apply_theme()
    page_header("Email Alerts", "Manage regulatory alert subscriptions and daily digests")

    st.info(
        "**Daily Regulatory Digest** — Subscribers receive a daily email summarizing "
        "all regulatory changes for their selected jurisdictions. Digests are delivered "
        "at 8:00 AM EST on business days when new updates are available."
    )

    state_options = get_state_jurisdiction_options()
    state_names = [s["name"] for s in state_options]
    state_ids = [int(s["id"]) for s in state_options]

    col_sub, col_unsub = st.columns(2)

    with col_sub:
        with st.container(border=True):
            st.markdown("**🔔 Subscribe to Alerts**")
            st.caption("Get notified about regulatory changes")
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

            if st.button("🔔 Subscribe", use_container_width=True, type="primary"):
                if not sub_email.strip():
                    st.error("Please enter an email.")
                elif sub_state_id is None:
                    st.error("Please select a state.")
                else:
                    try:
                        with st.spinner("Subscribing..."):
                            email_alerts.subscribe(email=sub_email.strip(), jurisdiction_id=sub_state_id)
                        st.success("Subscription saved (welcome email sent if SMTP is configured).")
                    except PermissionError as exc:
                        st.error(
                            "⚠️ **Database permission error.** The Supabase `anon` role cannot write to "
                            "`email_subscriptions`. Fix options:\n\n"
                            "1. Run the RLS policy SQL from **LOCAL_DEVELOPMENT.md § Step 6** in the Supabase SQL Editor.\n"
                            "2. Or set `SUPABASE_KEY` to the **service_role** key in `.env`."
                        )
                    except Exception as exc:
                        st.error(f"Subscription failed: {exc}")

    with col_unsub:
        with st.container(border=True):
            st.markdown("**🔕 Unsubscribe**")
            st.caption("Remove an existing subscription")
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

            if st.button("🔕 Unsubscribe", use_container_width=True):
                if not unsub_email.strip():
                    st.error("Please enter an email.")
                elif unsub_state_id is None:
                    st.error("Please select a state.")
                else:
                    try:
                        with st.spinner("Unsubscribing..."):
                            email_alerts.unsubscribe(
                                email=unsub_email.strip(), jurisdiction_id=unsub_state_id
                            )
                        st.success("You have been unsubscribed.")
                    except PermissionError:
                        st.error(
                            "⚠️ **Database permission error.** See LOCAL_DEVELOPMENT.md § Step 6 "
                            "or switch to the service_role key."
                        )
                    except Exception as exc:
                        st.error(f"Unsubscribe failed: {exc}")

    st.write("")
    with st.container(border=True):
        st.markdown("**👁️ View Subscriptions**")
        st.caption("Look up existing alert subscriptions by email")
        view_email = st.text_input(
            "Email to look up",
            placeholder="Enter email to look up",
            key="view_email",
        )
        if st.button("👁️ Look up", key="load_subs"):
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
                            st.write(f"- {s['jurisdiction_name']}")
                except PermissionError:
                    st.error(
                        "⚠️ **Database permission error.** See LOCAL_DEVELOPMENT.md § Step 6 "
                        "or switch to the service_role key."
                    )
                except Exception as exc:
                    st.error(f"Lookup failed: {exc}")


show_page()
