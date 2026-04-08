#!/usr/bin/env python3
"""RAG evaluation harness for the compliance agent.

Usage:
    python scripts/rag_eval.py                    # run all eval cases
    python scripts/rag_eval.py --ids esa_dallas_basic out_of_scope
    python scripts/rag_eval.py --retrieval-only    # skip LLM answer eval

Evaluates:
1. Retrieval hit@k — did we retrieve chunks from the right jurisdictions/sources?
2. Source grounding / citation presence — are citations present in the answer?
3. Confidence label coverage — does every result carry a valid confidence label?
4. Answer support heuristics — are expected topics covered?
5. Regression checks for jurisdiction-sensitive questions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or PROJECT_ROOT / "data" / "eval" / "eval_dataset.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    sources: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Check whether retrieved sources match expectations (hit@k)."""
    results: dict[str, Any] = {"checks": [], "pass": True}

    should_retrieve = expected.get("should_retrieve_from") or []
    source_texts = " ".join(
        f"{s.get('source', '')} {s.get('url', '')} {s.get('jurisdiction', '')}"
        for s in sources
    ).lower()

    hits = 0
    for loc in should_retrieve:
        found = loc.lower() in source_texts
        results["checks"].append({
            "check": f"retrieval_hit_{loc}",
            "passed": found,
        })
        if found:
            hits += 1
        else:
            results["pass"] = False

    total = len(should_retrieve) or 1
    results["hit_at_k"] = hits / total

    return results


def evaluate_confidence(
    confidence: str | None,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Check confidence label is present and valid."""
    results: dict[str, Any] = {"checks": [], "pass": True}
    valid_labels = {"grounded", "weak_evidence", "conflicting", "out_of_scope"}

    has_confidence = confidence is not None and confidence in valid_labels
    results["checks"].append({
        "check": "confidence_label_present",
        "passed": has_confidence,
    })
    if not has_confidence:
        results["pass"] = False

    if expected.get("should_be_out_of_scope"):
        is_oos = confidence == "out_of_scope"
        results["checks"].append({
            "check": "confidence_is_out_of_scope",
            "passed": is_oos,
        })
        if not is_oos:
            results["pass"] = False

    return results


def evaluate_grounding(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check if the answer references sources and contains citation markers."""
    results: dict[str, Any] = {"checks": [], "pass": True}

    has_sources = len(sources) > 0
    results["checks"].append({
        "check": "sources_present",
        "passed": has_sources,
    })

    import re
    citation_re = re.compile(
        r"§|Section\s+\d|Fair Housing|HUD|statute|regulation|Act\b",
        re.IGNORECASE,
    )
    has_citation_language = bool(citation_re.search(answer))
    results["checks"].append({
        "check": "citation_language_present",
        "passed": has_citation_language,
    })

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


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------


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
    confidence_eval = evaluate_confidence(confidence, expected)
    grounding_eval = evaluate_grounding(answer, sources)

    answer_eval: dict[str, Any] = {"checks": [], "pass": True}
    if not retrieval_only:
        answer_eval = evaluate_answer(answer, expected, confidence)

    overall_pass = (
        retrieval_eval["pass"]
        and confidence_eval["pass"]
        and answer_eval["pass"]
    )

    return {
        "id": case["id"],
        "question": question,
        "overall_pass": overall_pass,
        "retrieval": retrieval_eval,
        "confidence_eval": confidence_eval,
        "grounding": grounding_eval,
        "answer": answer_eval,
        "confidence": confidence,
        "num_sources": len(sources),
        "answer_length": len(answer),
        "hit_at_k": retrieval_eval.get("hit_at_k", 0),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[dict[str, Any]]) -> None:
    """Print a human-readable evaluation report."""
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    avg_hit_k = sum(r.get("hit_at_k", 0) for r in results) / max(total, 1)
    confidence_labels = [r.get("confidence") for r in results]
    confidence_coverage = sum(1 for c in confidence_labels if c is not None) / max(total, 1)

    print("\n" + "=" * 70)
    print(f"  RAG EVALUATION REPORT  —  {passed}/{total} passed")
    print(f"  Avg hit@k: {avg_hit_k:.2f}  |  Confidence coverage: {confidence_coverage:.0%}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["overall_pass"] else "FAIL"
        print(f"\n  [{status}] {r['id']}")
        print(f"    Q: {r['question']}")
        print(f"    Confidence: {r['confidence']}  |  Sources: {r['num_sources']}  |  hit@k: {r.get('hit_at_k', 0):.2f}  |  Len: {r['answer_length']}")

        all_checks = (
            r["retrieval"]["checks"]
            + r["confidence_eval"]["checks"]
            + r.get("grounding", {}).get("checks", [])
            + r["answer"]["checks"]
        )
        failed = [c for c in all_checks if not c["passed"]]
        if failed:
            for c in failed:
                print(f"    FAILED: {c['check']}")

    print("\n" + "-" * 70)
    print(f"  TOTAL: {passed}/{total} passed ({100 * passed / max(total, 1):.0f}%)")
    print(f"  Avg retrieval hit@k: {avg_hit_k:.2f}")
    print(f"  Confidence label coverage: {confidence_coverage:.0%}")
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
                "retrieval": {"checks": [], "pass": False, "hit_at_k": 0},
                "confidence_eval": {"checks": [], "pass": False},
                "grounding": {"checks": [], "pass": True},
                "answer": {"checks": [], "pass": False},
                "confidence": "error",
                "num_sources": 0,
                "answer_length": 0,
                "hit_at_k": 0,
            })

    print_report(results)

    output_path = PROJECT_ROOT / "data" / "eval" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
