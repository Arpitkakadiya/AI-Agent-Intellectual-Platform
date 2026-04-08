"""Index regulations into the vector store.

Uses legal-aware chunking by default (controlled by RAG_USE_LEGAL_CHUNKING).
Falls back to sliding-window when legal structure is not detected.

Usage:
    python scripts/index_regulations.py
    python scripts/index_regulations.py --force    # re-index all
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from core.regulations.scraper import scraper


def main() -> None:
    parser = argparse.ArgumentParser(description="Index regulations into vector store")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index all regulations (not just unindexed)",
    )
    args = parser.parse_args()

    chunking_mode = "legal-aware" if settings.RAG_USE_LEGAL_CHUNKING else "sliding-window"
    print(f"Chunking mode: {chunking_mode}")
    print(f"Embedding provider: {settings.embed_provider}")
    print(f"Hybrid retrieval: {'enabled' if settings.RAG_HYBRID_ENABLED else 'disabled'}")

    result: dict[str, Any] = scraper.initialize_vector_index()
    indexed = int(result.get("indexed_docs") or 0)
    print(f"Indexed {indexed} regulations")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
