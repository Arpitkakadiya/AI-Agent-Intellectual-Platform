from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel

from config import settings
from core.compliance.parser import parse_document
from core.compliance.rules import RuleEngine, rule_engine
from core.llm.client import llm
from core.llm.prompts import COMPLIANCE_SYSTEM_PROMPT, DOCUMENT_QA_SYSTEM_PROMPT
from core.rag.utils import deduplicate_sources
from db.models import InsuranceRequirement, PetPolicy


# NOTE:
# The project spec mandates core/ modules never import streamlit.
# This file follows that rule and uses db/client.py and core/llm/client.py only.


class ComplianceIssue(BaseModel):
    clause_number: int
    clause_title: str
    clause_content: str
    regulation_applies: str
    what_to_fix: str
    suggested_revision: Optional[str] = None


class ComplianceResult(BaseModel):
    is_compliant: bool
    total_clauses: int
    issues: list[ComplianceIssue]
    summary: str
    disclaimer: str
    sources: list[dict[str, Any]]


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_int_from_notes(notes: str, key: str) -> Optional[int]:
    if not notes:
        return None
    # Matches: deposit_return_days: 30 / deposit_return_days = 30 / deposit_return_days 30
    m = re.search(
        rf"{re.escape(key)}\s*(?:[:=]\s*)?(\d+)",
        notes,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return int(m.group(1))


def _extract_float_from_notes(notes: str, key: str) -> Optional[float]:
    if not notes:
        return None
    m = re.search(
        rf"{re.escape(key)}\s*(?:[:=]\s*)?(\d+(?:\.\d+)?)",
        notes,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return float(m.group(1))


def _get_jurisdiction_chain(db: Any, start_id: int) -> list[int]:
    chain: list[int] = []
    current: Optional[int] = int(start_id)

    while current is not None:
        res = (
            db.table("jurisdictions")
            .select("id,parent_id")
            .eq("id", current)
            .limit(1)
            .execute()
        )
        if not res.data:
            break

        row = res.data[0]
        chain.append(int(row["id"]))
        parent_id = row.get("parent_id")
        current = int(parent_id) if parent_id is not None else None

    return chain


def _get_jurisdiction_name(db: Any, jurisdiction_id: int) -> str:
    res = (
        db.table("jurisdictions")
        .select("name")
        .eq("id", int(jurisdiction_id))
        .limit(1)
        .execute()
    )
    if res.data and res.data[0].get("name"):
        return str(res.data[0]["name"])
    return "your selected jurisdiction"


def _load_jurisdiction_rules(db: Any, jurisdiction_id: int) -> dict[str, Any]:
    chain = _get_jurisdiction_chain(db, jurisdiction_id)

    pet_model: Optional[PetPolicy] = None
    for jid in chain:
        try:
            pet_res = (
                db.table("pet_policies")
                .select("*")
                .eq("jurisdiction_id", int(jid))
                .limit(1)
                .execute()
            )
            if pet_res.data:
                pet_model = PetPolicy.model_validate(pet_res.data[0])
                break
        except Exception as exc:
            if "permission denied" in str(exc).lower():
                raise PermissionError(
                    "Permission denied for table pet_policies. "
                    "Run db/migrations/010_pet_insurance_rls.sql in the Supabase SQL Editor, "
                    "or use a service_role key. See LOCAL_DEVELOPMENT.md step 6."
                ) from exc
            raise

    insurance_model: Optional[InsuranceRequirement] = None
    for jid in chain:
        try:
            ins_res = (
                db.table("insurance_requirements")
                .select("*")
                .eq("jurisdiction_id", int(jid))
                .limit(1)
                .execute()
            )
            if ins_res.data:
                insurance_model = InsuranceRequirement.model_validate(ins_res.data[0])
                break
        except Exception as exc:
            if "permission denied" in str(exc).lower():
                raise PermissionError(
                    "Permission denied for table insurance_requirements. "
                    "Run db/migrations/010_pet_insurance_rls.sql in the Supabase SQL Editor, "
                    "or use a service_role key. See LOCAL_DEVELOPMENT.md step 6."
                ) from exc
            raise

    jurisdiction_rules: dict[str, Any] = {}

    if pet_model:
        jurisdiction_rules.update(
            {
                "esa_deposit_allowed": pet_model.esa_deposit_allowed,
                "service_animal_fee": pet_model.service_animal_fee,
                "breed_restrictions": pet_model.breed_restrictions,
                "max_pet_deposit_amount": pet_model.max_pet_deposit_amount,
                "source_regulation_id": pet_model.source_regulation_id,
            }
        )

    if insurance_model:
        jurisdiction_rules.update(
            {
                "landlord_can_require": insurance_model.landlord_can_require,
                "min_liability_coverage": insurance_model.min_liability_coverage,
                "tenant_must_show_proof": insurance_model.tenant_must_show_proof,
                "notes": insurance_model.notes,
                "insurance_source_regulation_id": insurance_model.source_regulation_id,
            }
        )

        notes_text = _safe_str(insurance_model.notes)
        jurisdiction_rules["deposit_return_days"] = _extract_int_from_notes(
            notes_text, "deposit_return_days"
        )
        jurisdiction_rules["rent_increase_cap"] = _extract_float_from_notes(
            notes_text, "rent_increase_cap"
        )

    return jurisdiction_rules


def _build_regulation_context(results: list[Any], max_results: int = 5) -> str:
    blocks: list[str] = []
    for r in results[:max_results]:
        meta = getattr(r, "metadata", None) or {}
        header = meta.get("source_name") or meta.get("url") or "Source"
        doc = getattr(r, "document", "") or ""
        blocks.append(f"[{header}]\n{doc}")
    return "\n---\n".join(blocks)


def generate_summary(result: ComplianceResult, jurisdiction_name: str) -> str:
    jurisdiction_name_norm = jurisdiction_name.strip() or "your selected jurisdiction"
    if result.total_clauses <= 0:
        return f"No clauses were provided for review in {jurisdiction_name_norm}."

    if result.is_compliant:
        return (
            f"Based on the provided lease clauses, no obvious compliance issues were detected "
            f"for {jurisdiction_name_norm}."
        )

    return (
        f"{len(result.issues)} potential compliance issue(s) were identified across "
        f"{result.total_clauses} clause(s) for {jurisdiction_name_norm}. "
        f"Review the suggested fixes to reduce risk."
    )


class ComplianceChecker:
    def __init__(self) -> None:
        self._store = None
        self._vector_store = None
        self._rule_engine: RuleEngine = rule_engine

    def check_compliance(
        self, file_bytes: bytes, filename: str, jurisdiction_id: int
    ) -> ComplianceResult:
        parsed = parse_document(file_bytes, filename)

        # DB access must go through db/client.py.
        from db.client import get_db

        db = get_db()
        jurisdiction_rules = _load_jurisdiction_rules(db, jurisdiction_id)

        jurisdiction_name = _get_jurisdiction_name(db, jurisdiction_id)

        all_sources_raw: list[dict[str, Any]] = []
        issues: list[ComplianceIssue] = []

        # Vector store requires embeddings; only attempt when AI is available.
        vector_store_results_by_clause: list[list[Any]] = [[] for _ in parsed.clauses]
        if llm.is_ai_available():
            from core.rag.vector_store import RegulationVectorStore

            vector_store = RegulationVectorStore()
            for idx, clause in enumerate(parsed.clauses):
                try:
                    vr = vector_store.search(
                        query=clause.content,
                        n_results=10,
                        jurisdiction_id=jurisdiction_id,
                    )
                    vector_store_results_by_clause[idx] = vr

                    for r in vr:
                        meta = getattr(r, "metadata", None) or {}
                        all_sources_raw.append(
                            {
                                "url": meta.get("url"),
                                "source": meta.get("source_name"),
                            }
                        )
                except Exception:
                    # Fall back to rule-only analysis if vector search fails.
                    vector_store_results_by_clause[idx] = []

        for idx, clause in enumerate(parsed.clauses):
            rule_result = self._rule_engine.analyze_clause(
                clause=clause,
                jurisdiction_id=jurisdiction_id,
                jurisdiction_rules=jurisdiction_rules,
            )

            llm_issue: Optional[ComplianceIssue] = None
            if llm.is_ai_available():
                regulation_context = _build_regulation_context(
                    vector_store_results_by_clause[idx], max_results=5
                )
                user_message = (
                    f"Jurisdiction ID: {jurisdiction_id}\n\n"
                    f"Lease clause title: {clause.title}\n"
                    f"Clause number: {clause.number}\n\n"
                    f"Clause text:\n{clause.content}\n\n"
                    f"Regulation context:\n{regulation_context}\n\n"
                    "Return valid JSON only with keys:\n"
                    "- is_compliant: boolean\n"
                    "- regulation_applies: string\n"
                    "- what_to_fix: string\n"
                    "- suggested_revision: string or null\n"
                )

                try:
                    llm_json = llm.ask_json(
                        system=COMPLIANCE_SYSTEM_PROMPT,
                        user=user_message,
                        schema_hint=(
                            '{\"is_compliant\": boolean, \"regulation_applies\": string, '
                            '\"what_to_fix\": string, \"suggested_revision\": string|null}'
                        ),
                    )
                except Exception:
                    llm_json = {"error": "llm_failed"}

                if isinstance(llm_json, dict) and "is_compliant" in llm_json:
                    llm_is_compliant = bool(llm_json.get("is_compliant"))
                    if not llm_is_compliant:
                        llm_issue = ComplianceIssue(
                            clause_number=clause.number,
                            clause_title=clause.title,
                            clause_content=clause.content,
                            regulation_applies=_safe_str(
                                llm_json.get("regulation_applies")
                            ),
                            what_to_fix=_safe_str(llm_json.get("what_to_fix")),
                            suggested_revision=llm_json.get("suggested_revision"),
                        )

            # Merge: prefer LLM if it found an issue; otherwise use rule result as floor.
            if llm_issue is not None:
                if rule_result is not None:
                    # Ensure rule result is not lost when LLM output is incomplete.
                    llm_issue.regulation_applies = (
                        llm_issue.regulation_applies
                        or rule_result.regulation_applies
                    )
                    llm_issue.what_to_fix = llm_issue.what_to_fix or rule_result.what_to_fix
                    if llm_issue.suggested_revision is None:
                        llm_issue.suggested_revision = rule_result.suggested_revision
                issues.append(llm_issue)
                continue

            if rule_result is not None:
                issues.append(
                    ComplianceIssue(
                        clause_number=clause.number,
                        clause_title=clause.title,
                        clause_content=clause.content,
                        regulation_applies=rule_result.regulation_applies,
                        what_to_fix=rule_result.what_to_fix,
                        suggested_revision=rule_result.suggested_revision,
                    )
                )

        is_compliant = len(issues) == 0
        deduped_sources = deduplicate_sources(all_sources_raw)

        result = ComplianceResult(
            is_compliant=is_compliant,
            total_clauses=len(parsed.clauses),
            issues=issues,
            summary="",
            disclaimer=settings.LEGAL_DISCLAIMER,
            sources=deduped_sources,
        )

        # Ensure summary is generated from final result.
        result.summary = generate_summary(result, jurisdiction_name=jurisdiction_name)
        return result


    def document_qa(
        self,
        question: str,
        file_bytes: bytes,
        filename: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Answer a question about an uploaded document using its parsed text."""
        parsed = parse_document(file_bytes, filename)
        doc_text = parsed.text

        max_chars = 30000
        if len(doc_text) > max_chars:
            doc_text = doc_text[:max_chars] + "\n\n[... document truncated for context length ...]"

        history_block = ""
        if chat_history:
            recent = chat_history[-6:]
            lines: list[str] = []
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"{role}: {content}")
            history_block = f"Conversation history:\n" + "\n".join(lines) + "\n\n"

        user_message = (
            f"Document filename: {filename}\n\n"
            f"Document content:\n{doc_text}\n\n"
            f"{history_block}"
            f"Question: {question}"
        )

        doc_source = {"source": filename, "url": ""}

        if not llm.is_ai_available():
            return {
                "answer": (
                    "I cannot answer questions about the document because no LLM API key "
                    "is configured. Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
                    "GOOGLE_API_KEY to enable document Q&A."
                ),
                "sources": [doc_source],
            }

        try:
            raw_answer = llm.ask(system=DOCUMENT_QA_SYSTEM_PROMPT, user=user_message)
            return {"answer": raw_answer.strip(), "sources": [doc_source]}
        except Exception as exc:
            return {"answer": f"Failed to analyze document: {exc}", "sources": [doc_source]}


checker = ComplianceChecker()

