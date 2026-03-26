from __future__ import annotations

import sys
from typing import Final
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.client import get_db


US_STATES: Final[dict[str, str]] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

TEXAS_CITIES: Final[list[str]] = [
    "Dallas",
    "Houston",
    "Austin",
    "San Antonio",
    "Fort Worth",
]

FEDERAL_NAME: Final[str] = "United States"
FEDERAL_LEGACY_NAMES: Final[list[str]] = ["Federal Government", FEDERAL_NAME]


def _get_federal_id() -> int:
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
        raise RuntimeError("Failed to find federal jurisdiction row after insert.")
    return int(res.data[0]["id"])


def _get_state_id(state_code: str) -> int:
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
        raise RuntimeError(f"Failed to find state row for {state_code} after insert.")
    return int(res.data[0]["id"])


def main() -> None:
    db = get_db()

    # 1) Federal row (avoid ON CONFLICT to be resilient to missing unique indexes)
    existing_federal = (
        db.table("jurisdictions")
        .select("id")
        .eq("type", "federal")
        .in_("name", FEDERAL_LEGACY_NAMES)
        .limit(1)
        .execute()
    )
    if not existing_federal.data:
        db.table("jurisdictions").insert(
            [{"type": "federal", "name": FEDERAL_NAME, "parent_id": None}]
        ).execute()
    federal_id = _get_federal_id()
    # Normalize to the required name so downstream lookups are stable.
    db.table("jurisdictions").update({"name": FEDERAL_NAME}).eq("id", federal_id).execute()

    # 2) 50 states (check-then-insert to avoid needing unique constraints)
    for code, name in sorted(US_STATES.items()):
        existing_state = (
            db.table("jurisdictions")
            .select("id")
            .eq("type", "state")
            .eq("state_code", code)
            .limit(1)
            .execute()
        )
        if existing_state.data:
            if code == "TX":
                # Ensure Texas has the required name + parent relationship.
                db.table("jurisdictions").update({"name": "Texas", "parent_id": federal_id}).eq(
                    "id", int(existing_state.data[0]["id"])
                ).execute()
            continue

        db.table("jurisdictions").insert(
            [
                {
                    "type": "state",
                    "name": name,
                    "parent_id": federal_id,
                    "state_code": code,
                }
            ]
        ).execute()

    # 3) Texas cities
    texas_id = _get_state_id("TX")
    for city in TEXAS_CITIES:
        existing_city = (
            db.table("jurisdictions")
            .select("id")
            .eq("type", "city")
            .eq("name", city)
            .eq("parent_id", texas_id)
            .limit(1)
            .execute()
        )
        if existing_city.data:
            # Ensure state_code is set for the TX-city lookup used by seed_db.py.
            db.table("jurisdictions").update({"state_code": "TX"}).eq(
                "id", int(existing_city.data[0]["id"])
            ).execute()
            continue

        db.table("jurisdictions").insert(
            [{"type": "city", "name": city, "parent_id": texas_id, "state_code": "TX"}]
        ).execute()

    # 6) Print progress
    count_res = db.table("jurisdictions").select("id", count="exact").execute()
    seeded = int(count_res.count or 0)
    print(f"Seeded {seeded} jurisdictions")


if __name__ == "__main__":
    main()

