import streamlit as st

st.set_page_config(
    page_title="RegComply",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages/1_agent.py", title="Compliance Agent", icon="💬", default=True),
        st.Page("pages/2_explorer.py", title="Explorer", icon="🔍"),
        st.Page("pages/3_update_log.py", title="Update Log", icon="📄"),
        st.Page("pages/4_email_alerts.py", title="Email Alerts", icon="📧"),
        st.Page("pages/5_settings.py", title="Settings", icon="⚙️"),
        st.Page("pages/6_source_registry.py", title="Source Registry", icon="🗂️"),
    ],
    position="hidden",
)

pg.run()
