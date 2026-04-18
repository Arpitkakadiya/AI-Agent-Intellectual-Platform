from __future__ import annotations

from typing import Any, Optional

import streamlit as st

import config
from core.compliance.checker import checker
from core.rag.qa_system import qa
from core.rag.utils import deduplicate_sources
from db.client import get_db
from db.models import Jurisdiction
from core.regulations.scraper import is_supabase_connected
from ui_theme import apply_theme, callout_box, cross_page_link, log_activity, page_hero, section_heading, setup_banner

COMPLIANCE_KEYWORDS = ("compliant", "compliance", "check", "review my", "analyze my")
FILE_BYTES_KEY = "uploaded_file_bytes"
FILE_NAME_KEY = "uploaded_file_name"
FILE_SIZE_KEY = "uploaded_file_size"
COMPLIANCE_DONE_KEY = "compliance_review_done"
CHAT_KEY = "chat_history"
JURISDICTION_KEY = "jurisdiction_id"
SELECTED_STATE_KEY = "selected_state_id"
SELECTED_CITY_KEY = "selected_city_id"
PENDING_PROMPT_KEY = "pending_prompt"


def _init_state() -> None:
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = []
    if JURISDICTION_KEY not in st.session_state:
        st.session_state[JURISDICTION_KEY] = None
    if SELECTED_STATE_KEY not in st.session_state:
        st.session_state[SELECTED_STATE_KEY] = None
    if SELECTED_CITY_KEY not in st.session_state:
        st.session_state[SELECTED_CITY_KEY] = None


def _clear_chat() -> None:
    st.session_state[CHAT_KEY] = []
    for key in (FILE_BYTES_KEY, FILE_NAME_KEY, FILE_SIZE_KEY, PENDING_PROMPT_KEY, COMPLIANCE_DONE_KEY):
        if key in st.session_state:
            del st.session_state[key]


_DEFAULT_STATES: list[Jurisdiction] = [
    Jurisdiction(id=1, type="state", name="California", state_code="CA"),
    Jurisdiction(id=2, type="state", name="Colorado", state_code="CO"),
    Jurisdiction(id=3, type="state", name="Florida", state_code="FL"),
    Jurisdiction(id=4, type="state", name="New York", state_code="NY"),
    Jurisdiction(id=5, type="state", name="Texas", state_code="TX"),
]


def _load_states() -> list[Jurisdiction]:
    return list(_DEFAULT_STATES)


def _load_cities(state_id: int) -> list[Jurisdiction]:
    db = get_db()
    res = (
        db.table("jurisdictions")
        .select("id,type,name,parent_id,state_code,fips_code")
        .eq("type", "city")
        .eq("parent_id", int(state_id))
        .order("name")
        .execute()
    )
    return [Jurisdiction.model_validate(row) for row in (res.data or [])]


def _show_controls() -> None:
    """Render jurisdiction selector and document upload side-by-side on the main page."""
    col_jurisdiction, col_upload = st.columns(2)

    # ── Jurisdiction ──
    with col_jurisdiction:
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.625rem;margin-bottom:0.5rem;">'
                '<span style="font-size:1.15rem;">📍</span>'
                '<div>'
                '<div style="font-weight:600;font-size:0.9rem;color:var(--rc-text);">Jurisdiction</div>'
                '<div style="font-size:0.75rem;color:var(--rc-text-muted);">State and city for your query</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

            states = _load_states()

            if states:
                state_options = {s.name: int(s.id or 0) for s in states}
                default_state_name = states[0].name
                existing_state_id = st.session_state.get(SELECTED_STATE_KEY)
                for s in states:
                    if int(s.id or 0) == existing_state_id:
                        default_state_name = s.name
                        break

                selected_state_name = st.selectbox(
                    "State",
                    options=list(state_options.keys()),
                    index=list(state_options.keys()).index(default_state_name),
                )
                selected_state_id = state_options[selected_state_name]
                st.session_state[SELECTED_STATE_KEY] = selected_state_id

                try:
                    cities = _load_cities(selected_state_id)
                except Exception:  # noqa: BLE001
                    cities = []

                city_options: dict[str, Optional[int]] = {"Statewide (no city)": None}
                for c in cities:
                    city_options[c.name] = int(c.id or 0)

                existing_city_id = st.session_state.get(SELECTED_CITY_KEY)
                default_city_name = "Statewide (no city)"
                for name, city_id in city_options.items():
                    if city_id == existing_city_id:
                        default_city_name = name
                        break

                selected_city_name = st.selectbox(
                    "City (optional)",
                    options=list(city_options.keys()),
                    index=list(city_options.keys()).index(default_city_name),
                )
                selected_city_id = city_options[selected_city_name]
                st.session_state[SELECTED_CITY_KEY] = selected_city_id
                st.session_state[JURISDICTION_KEY] = selected_city_id or selected_state_id

    # ── Document Upload ──
    with col_upload:
        with st.container(border=True):
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.625rem;margin-bottom:0.5rem;">'
                '<span style="font-size:1.15rem;">📄</span>'
                '<div>'
                '<div style="font-weight:600;font-size:0.9rem;color:var(--rc-text);">Document Review</div>'
                '<div style="font-size:0.75rem;color:var(--rc-text-muted);">Upload a lease or contract for compliance analysis</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Upload lease or document",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=False,
                help="PDF or DOCX for compliance analysis",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                old_name = st.session_state.get(FILE_NAME_KEY)
                if old_name and uploaded.name != old_name:
                    st.session_state[CHAT_KEY].append(
                        {
                            "role": "assistant",
                            "content": f"Document changed from **{old_name}** to **{uploaded.name}**. "
                            "Future questions will reference the new document.",
                            "sources": [],
                        }
                    )
                st.session_state[FILE_BYTES_KEY] = file_bytes
                st.session_state[FILE_NAME_KEY] = uploaded.name
                st.session_state[FILE_SIZE_KEY] = uploaded.size

            if FILE_BYTES_KEY in st.session_state and FILE_NAME_KEY in st.session_state:
                fname = st.session_state[FILE_NAME_KEY]
                fsize = st.session_state.get(FILE_SIZE_KEY, 0)
                st.success(f"**{fname}** ({int(fsize) / 1024:.1f} KB)")


def _get_active_file() -> Optional[dict[str, Any]]:
    if FILE_BYTES_KEY in st.session_state and FILE_NAME_KEY in st.session_state:
        return {
            "bytes": st.session_state[FILE_BYTES_KEY],
            "name": st.session_state[FILE_NAME_KEY],
            "size": st.session_state.get(FILE_SIZE_KEY, 0),
        }
    return None


def _format_compliance_markdown(result: Any) -> str:
    issues = result.issues
    total = max(int(result.total_clauses), 1)
    compliant_count = total - len(issues)
    score = max(0, round((compliant_count / total) * 100))

    lines: list[str] = []
    lines.append("### Compliance Review")
    lines.append(f"- Compliance score: **{score}%**")
    lines.append(f"- Clauses reviewed: **{result.total_clauses}**")
    lines.append(f"- Issues found: **{len(issues)}**")
    lines.append("")
    lines.append(result.summary)

    if issues:
        lines.append("")
        lines.append("### Clause-by-Clause Issues")
        for issue in issues:
            lines.append(
                f"- **Clause {issue.clause_number}: {issue.clause_title}**\n"
                f"  - Regulation: {issue.regulation_applies}\n"
                f"  - What to fix: {issue.what_to_fix}"
            )
            if issue.suggested_revision:
                lines.append(f"  - Suggested revision: {issue.suggested_revision}")

        lines.append("")
        lines.append("### Action Items")
        unique_actions = []
        seen = set()
        for issue in issues:
            action = issue.what_to_fix.strip()
            if action and action not in seen:
                seen.add(action)
                unique_actions.append(action)
        for action in unique_actions:
            lines.append(f"- {action}")
    else:
        lines.append("")
        lines.append("### Action Items")
        lines.append("- No major issues detected in the provided clauses.")

    lines.append("")
    lines.append(f"> {result.disclaimer}")
    return "\n".join(lines)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    deduped = deduplicate_sources(sources)
    with st.expander("Sources", expanded=False):
        if not deduped:
            st.write("No sources available.")
            return
        for src in deduped:
            source_name = str(src.get("source") or src.get("source_name") or "Source")
            url = str(src.get("url") or "").strip()
            if url:
                st.markdown(f"- [{source_name}]({url})")
            else:
                st.markdown(f"- {source_name}")


def _should_run_compliance_review(prompt: str) -> bool:
    """Only run compliance review when the user explicitly asks for it."""
    prompt_lc = prompt.lower()
    return any(k in prompt_lc for k in COMPLIANCE_KEYWORDS)


def _handle_message(prompt: str, active_file: Optional[dict[str, Any]]) -> None:
    jurisdiction_id = st.session_state.get(JURISDICTION_KEY)
    if jurisdiction_id is None:
        callout_box("📍", "Please select a state above before asking a question.", "warning")
        return

    file_present = active_file is not None
    run_compliance = _should_run_compliance_review(prompt)

    st.session_state[CHAT_KEY].append(
        {
            "role": "user",
            "content": prompt,
            "file_uploaded": bool(file_present),
            "filename": active_file["name"] if file_present else None,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    sources: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        if run_compliance:
            if not file_present:
                answer = (
                    "I can run a compliance check, but I need a lease document first. "
                    "Please upload a PDF or DOCX in the **Document Review** section of the sidebar."
                )
            else:
                with st.spinner("Reviewing document for compliance..."):
                    try:
                        result = checker.check_compliance(
                            file_bytes=active_file["bytes"],
                            filename=active_file["name"],
                            jurisdiction_id=int(jurisdiction_id),
                        )
                        answer = _format_compliance_markdown(result)
                        sources = list(result.sources)
                        sources.insert(0, {"source": active_file["name"], "url": ""})
                        st.session_state[COMPLIANCE_DONE_KEY] = True
                        log_activity("Compliance review", active_file["name"])
                        st.toast("Compliance review complete", icon="✅")
                    except Exception:  # noqa: BLE001
                        answer = "Something went wrong during the compliance review. Please check your API key settings and try again."
                        st.toast("Compliance review failed", icon="❌")
        elif file_present:
            with st.spinner("Analyzing document..."):
                try:
                    qa_result = checker.document_qa(
                        question=prompt,
                        file_bytes=active_file["bytes"],
                        filename=active_file["name"],
                        chat_history=st.session_state[CHAT_KEY],
                    )
                    answer = str(qa_result.get("answer") or "")
                    sources = list(qa_result.get("sources") or [])
                    log_activity("Document Q&A", prompt[:60])
                except Exception:  # noqa: BLE001
                    answer = "Unable to analyze the document right now. Please verify your API keys are configured in Settings."
        else:
            with st.spinner("Thinking..."):
                try:
                    qa_result = qa.answer_question(
                        question=prompt,
                        chat_history=st.session_state[CHAT_KEY],
                        jurisdiction_id=int(jurisdiction_id),
                    )
                    answer = str(qa_result.get("answer") or "")
                    sources = list(qa_result.get("sources") or [])
                    log_activity("Asked question", prompt[:60])
                except Exception:  # noqa: BLE001
                    answer = "Unable to process your question right now. Please check that your AI provider is configured in Settings."

        st.markdown(answer)
        deduped_sources = deduplicate_sources(sources)
        _render_sources(deduped_sources)

    if any(k in answer.lower() for k in ("new law", "update", "regulation")):
        cross_page_link("📧", "Want alerts for new regulations? Subscribe to Email Alerts →", "pages/4_email_alerts.py")

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": deduped_sources,
    }
    st.session_state[CHAT_KEY].append(assistant_message)


def _render_history() -> None:
    for msg in st.session_state[CHAT_KEY]:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(str(msg.get("content", "")))
            if role == "assistant":
                _render_sources(list(msg.get("sources") or []))


def _render_empty_state() -> None:
    st.markdown('<div class="rc-cards-heading">Try one of these to get started</div>', unsafe_allow_html=True)

    cards = [
        (
            "🌐", "blue", "Regulation Q&A",
            "Ask about any housing regulation — rent control, eviction rules, tenant rights, and more.",
            "What are the latest rent control regulations in California?",
        ),
        (
            "🔒", "green", "Lease Compliance",
            "Get answers on notice periods, required disclosures, and legal obligations for landlords.",
            "What notice period is required for lease termination in NYC?",
        ),
        (
            "📄", "amber", "Document Review",
            "Upload a lease or contract and I'll check it against applicable regulations.",
            "Does my lease comply with fair housing requirements?",
        ),
    ]

    cols = st.columns(3)
    for i, (icon, color, label, desc, example_text) in enumerate(cards):
        with cols[i]:
            with st.container(border=True):
                st.markdown(
                    f'<div class="rc-prompt-card-v2">'
                    f'<div class="rc-prompt-card-v2-icon-wrap {color}">{icon}</div>'
                    f'<div class="rc-prompt-card-v2-label">{label}</div>'
                    f'<div class="rc-prompt-card-v2-desc">{desc}</div>'
                    f'<div class="rc-prompt-card-v2-example">"{example_text}"</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Ask this →", key=f"ex_{label}", use_container_width=True):
                    st.session_state[PENDING_PROMPT_KEY] = example_text
                    st.rerun()

    quick_questions = [
        "What disclosures must landlords provide?",
        "Explain security deposit limits",
        "Eviction notice requirements",
        "Fair housing protected classes",
    ]
    pill_cols = st.columns(len(quick_questions))
    for i, q in enumerate(quick_questions):
        with pill_cols[i]:
            if st.button(q, key=f"pill_{i}", use_container_width=True):
                st.session_state[PENDING_PROMPT_KEY] = q
                st.rerun()


def _render_onboarding() -> None:
    """Show a getting-started banner when the platform isn't fully configured."""
    db_ok = False
    try:
        db_ok = is_supabase_connected()
    except Exception:  # noqa: BLE001
        pass

    llm_ok = (
        config.settings.has_anthropic_key
        or config.settings.has_openai_key
        or config.settings.has_google_key
    )
    jurisdiction_ok = st.session_state.get(JURISDICTION_KEY) is not None

    setup_banner([
        (db_ok, "Connect database", "Add your Supabase URL and key in Settings"),
        (llm_ok, "Configure an AI provider", "Add an API key for Anthropic, OpenAI, or Google"),
        (jurisdiction_ok, "Select a jurisdiction", "Choose a state (and optional city) at the top of the page"),
    ])


def show_agent_page() -> None:
    _init_state()
    apply_theme()
    with st.sidebar:
        st.markdown(
            f'<div style="font-size:0.7rem;color:var(--rc-text-muted);line-height:1.4;'
            f'padding:0.5rem 0.25rem;margin-top:1rem;">{config.LEGAL_DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )
    page_hero("💬", "Compliance Agent", "AI-powered regulatory Q&A and document analysis — ask questions, review leases, and get citation-backed answers.", "indigo")

    st.markdown(
        '<div class="rc-hero">'
        '<div class="rc-hero-logo">⚖️</div>'
        '<div class="rc-hero-title">How can I help with compliance today?</div>'
        '<div class="rc-hero-subtitle">'
        'I can answer questions about housing regulations, review lease documents '
        'for compliance issues, and keep you informed about regulatory changes.'
        '</div>'
        '<div class="rc-hero-capabilities">'
        '<span class="rc-hero-cap"><span class="rc-hero-cap-dot"></span>Multi-jurisdiction</span>'
        '<span class="rc-hero-cap"><span class="rc-hero-cap-dot"></span>Document analysis</span>'
        '<span class="rc-hero-cap"><span class="rc-hero-cap-dot"></span>Real-time updates</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _show_controls()

    _render_onboarding()

    active_file = _get_active_file()

    if active_file and not st.session_state.get(COMPLIANCE_DONE_KEY):
        with st.container(border=True):
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.75rem;">'
                    f'<span style="font-size:1.5rem;">📄</span>'
                    f'<div>'
                    f'<div style="font-weight:600;font-size:0.95rem;">{active_file["name"]}</div>'
                    f'<div style="font-size:0.78rem;color:var(--rc-text-muted);">'
                    f'Ready for compliance analysis &middot; {int(active_file["size"]) / 1024:.1f} KB</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("Run Compliance Review", type="primary", use_container_width=True, key="btn_compliance_direct"):
                    st.session_state[PENDING_PROMPT_KEY] = f"Review {active_file['name']} for compliance issues"
                    st.rerun()

    if st.session_state.get(JURISDICTION_KEY) is None and not st.session_state[CHAT_KEY]:
        callout_box("📍", "Select a jurisdiction above to start asking questions.", "warning")

    if st.session_state[CHAT_KEY]:
        _render_history()
    else:
        _render_empty_state()

    if PENDING_PROMPT_KEY in st.session_state:
        prompt = st.session_state.pop(PENDING_PROMPT_KEY)
        _handle_message(str(prompt), active_file)
        st.rerun()

    if st.session_state[CHAT_KEY]:
        _, col_clear, _ = st.columns([3, 1, 3])
        with col_clear:
            if st.button("Clear conversation", key="btn_clear_chat", use_container_width=True):
                _clear_chat()
                st.rerun()

    if active_file is not None:
        placeholder = f"Ask anything about {active_file['name']}, or type 'compliance check' to review it..."
    elif st.session_state[CHAT_KEY]:
        placeholder = "Follow up, or ask a new question..."
    else:
        placeholder = "Ask about rent control, lease requirements, tenant rights, or any housing regulation..."

    prompt = st.chat_input(placeholder=placeholder)

    hint = (
        'Tip: Be specific — include the <strong>state or city</strong> and the '
        '<strong>topic</strong> for the best answers. '
        'E.g. "What are NYC rent stabilization rules for renewals?"'
    )
    st.markdown(f'<div class="rc-input-hint">{hint}</div>', unsafe_allow_html=True)

    if prompt:
        _handle_message(prompt, active_file)
        st.rerun()


show_agent_page()
