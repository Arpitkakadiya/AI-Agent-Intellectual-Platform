from __future__ import annotations

import sys
import traceback
from typing import Any

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.regulations.scraper import scraper
from core.regulations.update_checker import update_checker


def main() -> None:
    scrape_result: dict[str, Any] = scraper.scrape_and_index()
    updates = update_checker.check_for_updates(send_alerts=True)

    scraped = scrape_result.get("scraped", 0)
    indexed = scrape_result.get("indexed", 0)
    found = len(updates)
    print(f"Scraped {scraped}, indexed {indexed}, found {found} updates")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)

