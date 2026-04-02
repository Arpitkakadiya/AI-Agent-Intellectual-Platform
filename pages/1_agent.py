from __future__ import annotations

from typing import Any, Optional

import streamlit as st

import config
from core.compliance.checker import checker
from core.rag.qa_system import qa
from core.rag.utils import deduplicate_sources
from db.client import get_db
from db.models import Jurisdiction
from ui_theme import apply_theme, page_header

COMPLIANCE_KEYWORDS = ("compliant", "compliance", "check")
FILE_BYTES_KEY = "uploaded_file_bytes"
FILE_NAME_KEY = "uploaded_file_name"
FILE_SIZE_KEY = "uploaded_file_size"
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
    for key in (FILE_BYTES_KEY, FILE_NAME_KEY, FILE_SIZE_KEY, PENDING_PROMPT_KEY):
        if key in st.session_state:
            del st.session_state[key]


def _load_states() -> list[Jurisdiction]:
    db = get_db()
    res = (
        db.table("jurisdictions")
        .select("id,type,name,parent_id,state_code,fips_code")
        .eq("type", "state")
        .order("name")
        .execute()
    )
    return [Jurisdiction.model_validate(row) for row in (res.data or [])]


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


def _show_agent_sidebar_extras() -> None:
    """Extra sidebar widgets specific to the agent page (jurisdiction + document upload)."""
    with st.sidebar:
        st.divider()
        st.caption("JURISDICTION")

        try:
            states = _load_states()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load jurisdictions: {exc}")
            states = []

        if not states:
            st.session_state[JURISDICTION_KEY] = None
            st.selectbox("State", options=["No states available"], index=0, disabled=True)
            st.selectbox("City (optional)", options=["No cities available"], index=0, disabled=True)
        else:
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
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load cities: {exc}")
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

        st.divider()
        st.caption("DOCUMENT REVIEW")

        uploaded = st.file_uploader(
            "Upload lease or document",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=False,
            help="PDF or DOCX for compliance analysis",
        )

        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            st.session_state[FILE_BYTES_KEY] = file_bytes
            st.session_state[FILE_NAME_KEY] = uploaded.name
            st.session_state[FILE_SIZE_KEY] = uploaded.size

        if FILE_BYTES_KEY in st.session_state and FILE_NAME_KEY in st.session_state:
            fname = st.session_state[FILE_NAME_KEY]
            fsize = st.session_state.get(FILE_SIZE_KEY, 0)
            st.success(f"📄 **{fname}** ({int(fsize) / 1024:.1f} KB)")

        st.divider()
        st.caption(config.LEGAL_DISCLAIMER)


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


def _handle_message(prompt: str, active_file: Optional[dict[str, Any]]) -> None:
    jurisdiction_id = st.session_state.get(JURISDICTION_KEY)
    if jurisdiction_id is None:
        st.warning("Please select a state (and optional city) first.")
        return

    file_present = active_file is not None
    prompt_lc = prompt.lower()
    should_run_compliance = file_present or any(k in prompt_lc for k in COMPLIANCE_KEYWORDS)

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
        if should_run_compliance:
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
                    except Exception as exc:  # noqa: BLE001
                        answer = f"Compliance check failed: {exc}"
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
                except Exception as exc:  # noqa: BLE001
                    answer = f"Unable to answer right now: {exc}"

        st.markdown(answer)
        deduped_sources = deduplicate_sources(sources)
        _render_sources(deduped_sources)
        if any(k in answer.lower() for k in ("new law", "update", "regulation")):
            st.info("Want alerts for new regulations? Subscribe on the Email Alerts page.")

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
                text = str(msg.get("content", "")).lower()
                if any(k in text for k in ("new law", "update", "regulation")):
                    st.info("Want alerts for new regulations? Subscribe on the Email Alerts page.")


def _render_empty_state() -> None:
    st.write("")
    st.markdown("#### 💬 Compliance Assistant")
    st.write(
        "Ask questions about housing regulations, review lease compliance, "
        "or explore regulatory requirements."
    )
    st.write("")

    examples = [
        ("🌐 Regulation Q&A", "What are the latest rent control regulations in California?"),
        ("🔒 Lease Compliance", "What notice period is required for lease termination in NYC?"),
        ("📄 Document Review", "Does my lease comply with fair housing requirements?"),
    ]

    for label, text in examples:
        with st.container(border=True):
            if st.button(f"**{label}** — {text}", key=f"ex_{label}", use_container_width=True):
                st.session_state[PENDING_PROMPT_KEY] = text
                st.rerun()


def show_agent_page() -> None:
    _init_state()
    apply_theme()
    _show_agent_sidebar_extras()
    page_header("Compliance Agent", "AI-powered regulatory Q&A and document analysis")

    active_file = _get_active_file()

    if st.session_state[CHAT_KEY]:
        _render_history()
    else:
        _render_empty_state()

    if PENDING_PROMPT_KEY in st.session_state:
        prompt = st.session_state.pop(PENDING_PROMPT_KEY)
        _handle_message(str(prompt), active_file)
        st.rerun()

    if st.session_state[CHAT_KEY]:
        if st.button("🗑️ Clear conversation"):
            _clear_chat()
            st.rerun()

    placeholder = "Ask about housing regulations..."
    if active_file is not None:
        placeholder = f"Ask compliance question for {active_file['name']}..."

    prompt = st.chat_input(placeholder=placeholder)
    if prompt:
        _handle_message(prompt, active_file)
        st.rerun()


show_agent_page()
