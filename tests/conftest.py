from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest


# Keep IDs stable and explicit for jurisdiction scoping tests.
DALLAS_JURISDICTION_ID = 101
HOUSTON_JURISDICTION_ID = 202
TX_STATE_ID = 10
FEDERAL_ID = 1


@dataclass
class _ExecuteResult:
    data: list[dict[str, Any]]


class _TableQuery:
    def __init__(self, table_name: str, client: "FakeSupabaseClient") -> None:
        self._table_name = table_name
        self._client = client
        self._filters: dict[str, Any] = {}

    def select(self, _columns: str) -> "_TableQuery":
        return self

    def eq(self, column: str, value: Any) -> "_TableQuery":
        self._filters[column] = value
        return self

    def limit(self, _n: int) -> "_TableQuery":
        return self

    def execute(self) -> _ExecuteResult:
        return self._client._execute_table_query(
            table_name=self._table_name, filters=self._filters
        )


class _RPCQuery:
    def __init__(
        self,
        client: "FakeSupabaseClient",
        payload: dict[str, Any],
    ) -> None:
        self._client = client
        self._payload = payload

    def execute(self) -> _ExecuteResult:
        rows = self._client._execute_rpc_match_regulations(self._payload)
        return _ExecuteResult(data=rows)


class FakeSupabaseClient:
    """
    Minimal Supabase client stub supporting:
    - db.table(...).select(...).eq(...).limit(...).execute()
    - db.rpc("match_regulations", payload).execute()
    """

    def __init__(self) -> None:
        # Allow tests to override rpc behavior.
        self._match_regulations_override: Optional[
            Callable[[dict[str, Any]], list[dict[str, Any]]]
        ] = None

    def set_match_regulations_override(
        self, fn: Callable[[dict[str, Any]], list[dict[str, Any]]]
    ) -> None:
        self._match_regulations_override = fn

    def table(self, table_name: str) -> _TableQuery:
        return _TableQuery(table_name=table_name, client=self)

    def rpc(self, _function_name: str, payload: dict[str, Any]) -> _RPCQuery:
        # Function name is ignored; this stub is only for match_regulations.
        return _RPCQuery(client=self, payload=payload)

    # Jurisdiction rows used for ID-based lookups (build_retrieval_plan)
    _JURISDICTION_ROWS: list[dict[str, Any]] = [
        {"id": FEDERAL_ID, "type": "federal", "name": "Federal Government", "parent_id": None, "state_code": None},
        {"id": TX_STATE_ID, "type": "state", "name": "Texas", "parent_id": FEDERAL_ID, "state_code": "TX"},
        {"id": DALLAS_JURISDICTION_ID, "type": "city", "name": "Dallas", "parent_id": TX_STATE_ID, "state_code": "TX"},
        {"id": HOUSTON_JURISDICTION_ID, "type": "city", "name": "Houston", "parent_id": TX_STATE_ID, "state_code": "TX"},
    ]

    def _execute_table_query(
        self, table_name: str, filters: dict[str, Any]
    ) -> _ExecuteResult:
        if table_name != "jurisdictions":
            return _ExecuteResult(data=[])

        # ID-based lookup (used by jurisdiction.py's _lookup_jurisdiction)
        if "id" in filters:
            fid = int(filters["id"])
            for row in self._JURISDICTION_ROWS:
                if row["id"] == fid:
                    return _ExecuteResult(data=[row])
            return _ExecuteResult(data=[])

        row_type = filters.get("type")
        if row_type == "federal":
            return _ExecuteResult(data=[{"id": FEDERAL_ID, "name": "Federal Government"}])

        if row_type == "state":
            state_code = filters.get("state_code")
            state_name = filters.get("name")
            if isinstance(state_code, str) and state_code.upper() == "TX":
                return _ExecuteResult(data=[{"id": TX_STATE_ID}])
            if isinstance(state_name, str) and state_name.strip().lower() == "texas":
                return _ExecuteResult(data=[{"id": TX_STATE_ID}])
            return _ExecuteResult(data=[])

        return _ExecuteResult(data=[])

    def _execute_rpc_match_regulations(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if self._match_regulations_override is not None:
            return self._match_regulations_override(payload)

        jurisdiction_filter = payload.get("filter_jurisdiction")

        dallas_row = {
            "id": 1,
            "chunk_text": (
                "Emotional Support Animals (ESAs) are not pets under applicable "
                "assistance animal guidance. In jurisdictions like Dallas, "
                "landlords must provide reasonable accommodation and may not impose "
                "pet fees, pet rent, deposits, or similar charges solely because a "
                "tenant has an ESA. Additional rules may govern documentation, "
                "but the ESA-specific exemption language should appear in the policy "
                "so that fees are not demanded without lawful basis. "
                "All references should align with the Fair Housing Act and related "
                "HUD assistance animal guidance. "
                "This excerpt is intentionally long to pass QA chunk informativity "
                "thresholds in tests."
            ),
            "metadata": {
                "source_name": "Dallas TX ESA Pet Fee Exemption",
                "url": "https://example.com/dallas-esa",
                "category": "Fair Housing",
                "domain": "tx",
                "jurisdiction_id": DALLAS_JURISDICTION_ID,
            },
            "similarity": 0.91,
        }
        houston_row = {
            "id": 2,
            "chunk_text": (
                "Houston-specific policy language addressing ESAs and reasonable "
                "accommodations explains that Emotional Support Animals are exempt "
                "from pet fee obligations. The regulation excerpt emphasizes that "
                "fees and deposits are only permitted when they are lawful and not "
                "imposed merely due to ESA status, and it mirrors Fair Housing "
                "Act and HUD guidance for assistance animals. "
                "This excerpt is intentionally long to pass QA chunk informativity "
                "thresholds in tests."
            ),
            "metadata": {
                "source_name": "Houston TX ESA Pet Fee Exemption",
                "url": "https://example.com/houston-esa",
                "category": "Fair Housing",
                "domain": "tx",
                "jurisdiction_id": HOUSTON_JURISDICTION_ID,
            },
            "similarity": 0.85,
        }

        if jurisdiction_filter == HOUSTON_JURISDICTION_ID:
            return [houston_row]

        if jurisdiction_filter == DALLAS_JURISDICTION_ID:
            return [dallas_row]

        # Some code paths may query federal or state-level jurisdiction ids.
        if jurisdiction_filter in {TX_STATE_ID, FEDERAL_ID}:
            return [dallas_row]

        # No filter -> return both so tests can assert scoping.
        if jurisdiction_filter is None:
            return [dallas_row, houston_row]

        # Unknown filter -> default to Dallas for safety.
        return [dallas_row]


@pytest.fixture()
def mock_supabase_client(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    """
    Provide a mock Supabase client and patch get_db() across RAG modules.

    This ensures qa_system / vector_store / jurisdiction / hybrid never reach
    the real Supabase.
    """

    client = FakeSupabaseClient()

    # Patch db client factory.
    import db.client as _db_client

    monkeypatch.setattr(_db_client, "get_db", lambda: client)

    # Patch already-imported get_db references in all RAG modules.
    import core.rag.qa_system as _qa_mod
    import core.rag.vector_store as _vs_mod

    monkeypatch.setattr(_qa_mod, "get_db", lambda: client)
    monkeypatch.setattr(_vs_mod, "get_db", lambda: client)

    # Patch new modules that also import get_db
    try:
        import core.rag.jurisdiction as _jur_mod
        monkeypatch.setattr(_jur_mod, "get_db", lambda: client)
    except (ImportError, AttributeError):
        pass

    try:
        import core.rag.hybrid as _hyb_mod
        monkeypatch.setattr(_hyb_mod, "get_db", lambda: client)
    except (ImportError, AttributeError):
        pass

    # Disable hybrid search in tests by default (avoids lexical RPC calls).
    # Patch via all known import paths to ensure consistency.
    try:
        import config as _cfg
        monkeypatch.setattr(_cfg.settings, "RAG_HYBRID_ENABLED", False)
        monkeypatch.setattr(_cfg.settings, "RAG_RETRIEVAL_TOP_N", 5)
    except (ImportError, AttributeError):
        pass

    try:
        monkeypatch.setattr(_qa_mod.settings, "RAG_HYBRID_ENABLED", False)
        monkeypatch.setattr(_qa_mod.settings, "RAG_RETRIEVAL_TOP_N", 5)
    except (ImportError, AttributeError):
        pass

    return client

