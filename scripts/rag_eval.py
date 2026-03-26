#!/usr/bin/env python3
"""Lightweight RAG evaluation harness for the compliance agent.

Usage:
    python scripts/rag_eval.py                    # run all eval cases
    python scripts/rag_eval.py --ids esa_dallas_basic out_of_scope
    python scripts/rag_eval.py --retrieval-only    # skip LLM answer eval

Reads the eval dataset from data/eval/eval_dataset.json and evaluates:
1. Retrieval quality  — did we retrieve chunks from the right jurisdictions/sources?
2. Jurisdiction correctness — does the answer scope match expectations?
3. Source/citation correctness — are expected sources mentioned?
4. Answer completeness — are expected topics covered?
5. Hallucination risk — does the answer contain unsupported claims?

Prints a readable report with per-case pass/fail and aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    sources: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Check whether retrieved sources match expectations."""
    results: dict[str, Any] = {"checks": [], "pass": True}

    should_retrieve = expected.get("should_retrieve_from") or []
    source_texts = " ".join(
        f"{s.get('source', '')} {s.get('url', '')} {s.get('jurisdiction', '')}"
        for s in sources
    ).lower()

    for loc in should_retrieve:
        found = loc.lower() in source_texts
        results["checks"].append({
            "check": f"retrieval_from_{loc}",
            "passed": found,
        })
        if not found:
            results["pass"] = False

    return results


def evaluate_answer(
    answer: str,
    expected: dict[str, Any],
    confidence: str | None = None,
) -> dict[str, Any]:
    """Check answer content against expectations."""
    results: dict[str, Any] = {"checks": [], "pass": True}
    answer_lower = answer.lower()

    if expected.get("should_be_out_of_scope"):
        is_oos = (
            "not related" in answer_lower
            or "out of scope" in answer_lower
            or "can't assist" in answer_lower
            or "specialized in" in answer_lower
            or confidence == "out_of_scope"
        )
        results["checks"].append({
            "check": "out_of_scope_detected",
            "passed": is_oos,
        })
        if not is_oos:
            results["pass"] = False
        return results

    for topic in expected.get("must_mention_topics") or []:
        found = topic.lower() in answer_lower
        results["checks"].append({
            "check": f"topic_{topic}",
            "passed": found,
        })
        if not found:
            results["pass"] = False

    for src in expected.get("must_mention_sources") or []:
        found = src.lower() in answer_lower
        results["checks"].append({
            "check": f"source_cited_{src}",
            "passed": found,
        })
        if not found:
            results["pass"] = False

    for bad in expected.get("should_not_hallucinate") or []:
        found = bad.lower() in answer_lower
        results["checks"].append({
            "check": f"no_hallucination_{bad[:30]}",
            "passed": not found,
        })
        if found:
            results["pass"] = False

    return results


def run_single_eval(
    case: dict[str, Any],
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """Run a single evaluation case through the QA system."""
    from core.rag.qa_system import qa

    question = case["question"]
    jurisdiction_id = case.get("jurisdiction_id")
    expected = case.get("expected", {})

    result = qa.answer_question(
        question=question,
        chat_history=[],
        jurisdiction_id=jurisdiction_id,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    confidence = result.get("confidence")

    retrieval_eval = evaluate_retrieval(sources, expected)

    answer_eval: dict[str, Any] = {"checks": [], "pass": True}
    if not retrieval_only:
        answer_eval = evaluate_answer(answer, expected, confidence)

    overall_pass = retrieval_eval["pass"] and answer_eval["pass"]

    return {
        "id": case["id"],
        "question": question,
        "overall_pass": overall_pass,
        "retrieval": retrieval_eval,
        "answer": answer_eval,
        "confidence": confidence,
        "num_sources": len(sources),
        "answer_length": len(answer),
    }


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a human-readable evaluation report."""
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])

    print("\n" + "=" * 70)
    print(f"  RAG EVALUATION REPORT  —  {passed}/{total} passed")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["overall_pass"] else "FAIL"
        print(f"\n  [{status}] {r['id']}")
        print(f"    Q: {r['question']}")
        print(f"    Confidence: {r['confidence']}  |  Sources: {r['num_sources']}  |  Answer length: {r['answer_length']}")

        all_checks = r["retrieval"]["checks"] + r["answer"]["checks"]
        failed = [c for c in all_checks if not c["passed"]]
        if failed:
            for c in failed:
                print(f"    FAILED: {c['check']}")

    print("\n" + "-" * 70)
    print(f"  TOTAL: {passed}/{total} passed ({100 * passed / max(total, 1):.0f}%)")
    print("-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--ids", nargs="*", help="Run only these eval case IDs")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer evaluation (faster, no API calls for answer gen)",
    )
    parser.add_argument("--dataset", type=str, help="Path to eval dataset JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    cases = load_eval_dataset(dataset_path)

    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]

    if not cases:
        print("No eval cases to run.")
        return

    print(f"Running {len(cases)} evaluation case(s)...")
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            r = run_single_eval(case, retrieval_only=args.retrieval_only)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR on {case['id']}: {exc}")
            results.append({
                "id": case["id"],
                "question": case["question"],
                "overall_pass": False,
                "retrieval": {"checks": [], "pass": False},
                "answer": {"checks": [], "pass": False},
                "confidence": "error",
                "num_sources": 0,
                "answer_length": 0,
            })

    print_report(results)

    output_path = PROJECT_ROOT / "data" / "eval" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
