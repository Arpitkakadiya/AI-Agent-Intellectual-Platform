from __future__ import annotations

import csv
import hashlib
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.client import get_db


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

FEDERAL_LEGACY_NAMES: list[str] = ["Federal Government", "United States"]


STATE_NAME_TO_CODE: dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

CODE_TO_STATE_NAME: dict[str, str] = {code: name for name, code in STATE_NAME_TO_CODE.items()}

CITY_TO_STATE_CODE: dict[str, str] = {
    "Los Angeles": "CA",
    "San Francisco": "CA",
    "San Diego": "CA",
    "New York City": "NY",
    "Dallas": "TX",
    "Houston": "TX",
    "Austin": "TX",
    "San Antonio": "TX",
    "Fort Worth": "TX",
}


def _get_federal_jurisdiction_id() -> int:
    db = get_db()
    res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "federal")
        .in_("name", FEDERAL_LEGACY_NAMES)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(
            "Federal jurisdiction row is missing. Run scripts/seed_jurisdictions.py first."
        )
    return int(res.data[0]["id"])


def _get_state_jurisdiction_id(state_code: str) -> int:
    db = get_db()
    res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "state")
        .eq("state_code", state_code)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise RuntimeError(
            f"Unknown state_code '{state_code}'. Run scripts/seed_jurisdictions.py first."
        )
    return int(res.data[0]["id"])


def _infer_state_code(city_name: str) -> str | None:
    label = (city_name or "").strip()
    if not label:
        return None

    if label.lower().endswith("-statewide"):
        # Example: "California-Statewide" -> "California"
        state_part = label.split("-", 1)[0].strip()
        state_part = state_part.replace("NewYork", "New York")
        return STATE_NAME_TO_CODE.get(state_part)

    if label in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[label]

    # City name fallback (used by older datasets without state_code).
    return CITY_TO_STATE_CODE.get(label)


def _resolve_jurisdiction_id(category: str, city_name: str, state_code: str) -> int:
    """
    Resolve `jurisdictions.id` for a CSV row.

    Uses the CSV `state_code` column for correct lookups across states.
    """

    kind = (category or "").strip().lower()
    label = (city_name or "").strip()
    code = (state_code or "").strip().upper()

    if kind == "federal":
        return _get_federal_jurisdiction_id()

    if kind == "state":
        if not code:
            raise RuntimeError("Missing state_code for state jurisdiction row.")
        return _get_state_jurisdiction_id(code)

    if kind == "city":
        if not code:
            raise RuntimeError("Missing state_code for city jurisdiction row.")
        # Prefer an actual city jurisdiction if present; otherwise fall back to state.
        db = get_db()
        city_res = (
            db.table("jurisdictions")
            .select("id")
            .eq("type", "city")
            .eq("state_code", code)
            .eq("name", label)
            .limit(1)
            .execute()
        )
        if city_res.data:
            return int(city_res.data[0]["id"])
        # Fallback: if city jurisdictions aren't seeded for every state, keep imports working
        # by using the state jurisdiction as the best available scope.
        return _get_state_jurisdiction_id(code)

    # Regulation-category rows (e.g. "Renters", "Pet Policy", "ESA") use `city_name` + `state_code`
    # to decide whether the target jurisdiction is a city or a state.
    if not code:
        raise RuntimeError(f"Missing state_code for category='{category}' row with city_name='{label}'.")

    expected_state_name = CODE_TO_STATE_NAME.get(code)
    if expected_state_name and label.lower() == expected_state_name.lower():
        return _get_state_jurisdiction_id(code)

    db = get_db()
    city_res = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "city")
        .eq("name", label)
        .eq("state_code", code)
        .limit(1)
        .execute()
    )
    if not city_res.data:
        # Fallback: if city jurisdictions aren't seeded for this state, fall back to the state scope.
        return _get_state_jurisdiction_id(code)
    return int(city_res.data[0]["id"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "seeds" / "sources.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing seed file: {csv_path}")

    db = get_db()

    loaded = 0
    skipped = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("hyperlink") or "").strip()
            if not url:
                continue

            category = (row.get("category") or "").strip()
            city_name = (row.get("city_name") or "").strip()
            law_name = (row.get("law_name") or "").strip()
            state_code = (row.get("state_code") or "").strip().upper()
            if not state_code:
                state_code = _infer_state_code(city_name or "")
            if not state_code:
                raise RuntimeError(
                    f"Missing/unknown state_code for row category='{category}', city_name='{city_name}'."
                )
            # Your requirements: `content` is just law_name, and hash is derived from `content` only.
            content = (law_name or "").strip()
            content_hash = _sha256(content)

            # Idempotency: skip if ANY existing row already has this content_hash.
            existing_hash_res = (
                db.table("regulations")
                .select("id")
                .eq("content_hash", content_hash)
                .limit(1)
                .execute()
            )
            if existing_hash_res.data:
                skipped += 1
                continue

            jurisdiction_id = _resolve_jurisdiction_id(
                category=category, city_name=city_name, state_code=state_code
            )

            payload: dict[str, Any] = {
                "jurisdiction_id": jurisdiction_id,
                "domain": "housing",
                "category": category or "General",
                "source_name": law_name or "Unknown",
                "url": url,
                "content": content,
                "content_hash": content_hash,
                "version": 1,
                "is_current": True,
                "effective_date": None,
            }

            db.table("regulations").insert([payload]).execute()
            loaded += 1

    print(f"Loaded {loaded} regulations, skipped {skipped} duplicates")


if __name__ == "__main__":
    main()

