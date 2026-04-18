import streamlit as st

st.set_page_config(
    page_title="Multi-Family Housing — Compliance Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Multi-Family Housing — AI-powered compliance platform for property managers, attorneys, and developers.",
    },
)

_CRITICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --rc-secondary: #0f172a;
    --rc-secondary-light: #1e293b;
    --rc-bg: #f8fafc;
}
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--rc-bg);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--rc-secondary) 0%, var(--rc-secondary-light) 100%) !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] [data-baseweb="tab"] { display: none !important; }
</style>
"""
st.markdown(_CRITICAL_CSS, unsafe_allow_html=True)

pg = st.navigation(
    {
        "": [
            st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
        ],
        "Platform": [
            st.Page("pages/1_agent.py", title="Compliance Agent", icon="💬"),
            st.Page("pages/2_explorer.py", title="Explorer", icon="🔍"),
            st.Page("pages/3_update_log.py", title="Update Log", icon="📄"),
            st.Page("pages/4_email_alerts.py", title="Email Alerts", icon="📧"),
        ],
        "Administration": [
            st.Page("pages/5_settings.py", title="Settings", icon="⚙️"),
            st.Page("pages/6_source_registry.py", title="Source Registry", icon="🗂️"),
        ],
    },
)

pg.run()
