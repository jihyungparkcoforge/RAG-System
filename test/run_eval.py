"""Run the evaluation without pytest and print the results as tables.

Usage (from the project root):
    python test/run_eval.py                              # retrieval only, no API calls
    python test/run_eval.py --generation                 # + answer generation (LLM API)
    python test/run_eval.py --generation --judge         # + Microsoft-Decision-1.1 judge
    python test/run_eval.py --case hotel_conflict --k 3  # one case, different k
    python test/run_eval.py --report test/results-all.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from evaluation import (
    EVAL_COLLECTION,
    JUDGE_ENV,
    ROOT,
    attach_llm_client,
    build_eval_pipeline,
    build_report,
    evaluate_generation,
    evaluate_retrieval,
    load_cases,
    missing_env,
)


def parse_args():
    doc = __doc__
    assert doc is not None
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k cutoff for recall_at_k (default 5)."
    )
    parser.add_argument(
        "--case", action="append", help="Run only this case id (repeatable)."
    )
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Also evaluate generated answers (API calls).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also run the decision judge (needs --generation).",
    )
    parser.add_argument("--report", help="Write the full JSON report to this path.")
    args = parser.parse_args()
    if args.judge and not args.generation:
        parser.error("--judge requires --generation")
    return args


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in column) for column in zip(headers, *rows)]
    line = "  ".join(f"{{:<{width}}}" for width in widths)
    print(line.format(*headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line.format(*row))


def fmt_ranks(row):
    """Gold chunk ranks as 'pre→post' per gold chunk, e.g. '3→3, 1→1'."""
    return ", ".join(
        f"{ranks['pre_rerank'] or '-'}→{ranks['post_rerank'] or '-'}"
        for ranks in row["gold_ranks"].values()
    )


def print_retrieval(rows, k):
    print(f"\n=== Retrieval (recall_at_k uses k={k}; R@n = post-rerank recall@n) ===")
    print_table(
        [
            "case",
            f"recall@{k}",
            "cand_recall",
            "R@1",
            "R@3",
            "R@5",
            "MRR pre→post",
            "gold rank pre→post",
            "via",
            "gold keywords",
        ],
        [
            [
                row["case_id"],
                f"{row['recall_at_k']:.2f}",
                f"{row['candidate_recall']:.2f}",
                f"{row['post_rerank']['recall_at_1']:.2f}",
                f"{row['post_rerank']['recall_at_3']:.2f}",
                f"{row['post_rerank']['recall_at_5']:.2f}",
                f"{row['pre_rerank']['mrr']:.2f}→{row['post_rerank']['mrr']:.2f}",
                fmt_ranks(row),
                ",".join(
                    sorted(
                        {
                            way
                            for ways in row["gold_retrieved_by"].values()
                            for way in ways
                        }
                    )
                )
                or "-",
                "; ".join(
                    ",".join(words) for words in row["gold_matched_keywords"].values()
                )
                or "-",
            ]
            for row in rows
        ],
    )


def print_generation(rows):
    print("\n=== Generation (gold context) ===")
    print_table(
        ["case", "missing_facts", "invalid_cit", "missing_cit", "p_correct", "verdict"],
        [
            [
                row["case_id"],
                ",".join(row["missing_facts"]) or "-",
                ",".join(row["invalid_citations"]) or "-",
                ",".join(row["missing_evidence_citations"]) or "-",
                f"{row['decision_judge']['probability_correct']:.3f}"
                if row["decision_judge"]
                else "-",
                row["decision_judge"]["verdict"] if row["decision_judge"] else "-",
            ]
            for row in rows
        ],
    )


def print_summary(report, k):
    print("\n=== Summary ===")
    print(
        f"mean recall@{k}: {report['mean_recall_at_k']:.3f} ({report['retrieval_cases_evaluated']} cases)"
    )
    for stage, metrics in report["rank_metrics"].items():
        print(
            f"{stage:12} "
            + "  ".join(f"{name}={value:.3f}" for name, value in metrics.items())
        )
    generation = [row for row in report["results"] if row["stage"] == "generation"]
    if generation:
        passed = sum(
            not (
                row["missing_facts"]
                or row["invalid_citations"]
                or row["missing_evidence_citations"]
            )
            for row in generation
        )
        print(f"generation regex+citation checks passed: {passed}/{len(generation)}")
    if report["judge_verdicts"]:
        print(f"judge verdicts: {report['judge_verdicts']}")


def main():
    args = parse_args()
    load_dotenv(ROOT / ".env")

    cases = load_cases()
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            sys.exit(f"Unknown case id: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in args.case]
    if args.judge and missing_env(JUDGE_ENV):
        sys.exit("Missing judge configuration: " + ", ".join(missing_env(JUDGE_ENV)))

    with tempfile.TemporaryDirectory(prefix="rag_eval_") as db_dir:
        print("Loading models and ingesting docs into a temporary DB...")
        pipeline, client = build_eval_pipeline(db_dir)
        try:
            retrieval = [evaluate_retrieval(pipeline, case, k=args.k) for case in cases]
            generation = []
            if args.generation:
                attach_llm_client(pipeline)
                try:
                    for case in cases:
                        print(f"Generating: {case['id']}")
                        generation.append(
                            evaluate_generation(pipeline, case, run_judge=args.judge)
                        )
                finally:
                    pipeline.llm_client.close()
        finally:
            client.delete_collection(EVAL_COLLECTION)

    report = build_report(retrieval + generation)
    print_retrieval(retrieval, args.k)
    if generation:
        print_generation(generation)
    print_summary(report, args.k)

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
