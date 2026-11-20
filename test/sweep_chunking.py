"""Compare retrieval quality across chunk sizes and overlaps.

Gold chunk IDs in evaluation_cases.json are tied to the default chunking
(800 chars / 50 overlap), so they cannot be reused when chunking changes.
Here a retrieved chunk counts as relevant if it covers at least half of a
case's evidence quote (character span in the source document). A quote split
across a boundary is therefore still found by either half-covering chunk.

With --generation, each configuration is also evaluated end to end like
ask(): every reranked candidate is sent to generate(), and the answer is
checked with the case's regex facts, citations, and the decision judge.
This makes 2 API calls per case per configuration.

Usage (from the project root):
    python test/sweep_chunking.py                                  # retrieval only, no API calls
    python test/sweep_chunking.py --sizes 300 500 800 --overlaps 0 50 100
    python test/sweep_chunking.py --sizes 300 500 800 1200 --overlaps 50 --generation
"""

import argparse
import contextlib
import io
import tempfile

from dotenv import load_dotenv
from evaluation import (
    EVAL_COLLECTION,
    JUDGE_ENV,
    ROOT,
    attach_llm_client,
    build_eval_pipeline,
    cited_labels,
    judge_answer_with_decision,
    load_cases,
    missing_answer_facts,
    missing_env,
)

MIN_QUOTE_COVERAGE = 0.5


def parse_args():
    doc = __doc__
    assert doc is not None
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[200, 300, 500, 800, 1200]
    )
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument(
        "--generation",
        action="store_true",
        help="Also generate answers end to end and judge them (2 API calls per case per config).",
    )
    return parser.parse_args()


def read_docs():
    # Same newline normalization as chunk_text_recursive.
    return {
        path.name: path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        for path in sorted((ROOT / "docs").glob("*.txt"))
    }


def chunk_spans(text, chunk_size, overlap):
    """(start, end) of each chunk in the source text, in chunk-index order."""
    from src.chunking import chunk_text_recursive

    spans, cursor = [], 0
    for chunk in chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap):
        start = text.find(chunk, cursor)
        if start == -1:
            raise ValueError(f"Chunk not found in source text: {chunk[:60]!r}")
        spans.append((start, start + len(chunk)))
        cursor = start + 1
    return spans


def relevant_chunks(case, docs, spans_by_doc):
    """For each evidence quote, the set of chunk IDs covering >= half of it."""
    groups = []
    for gold_id, quote in case["evidence_quotes"].items():
        doc_name = gold_id.split(":")[0]
        q_start = docs[doc_name].find(quote)
        if q_start == -1:
            raise ValueError(f"Evidence quote not found in {doc_name}: {quote!r}")
        q_end = q_start + len(quote)
        groups.append(
            {
                f"{doc_name}:{index:04d}"
                for index, (start, end) in enumerate(spans_by_doc[doc_name])
                if min(end, q_end) - max(start, q_start)
                >= MIN_QUOTE_COVERAGE * len(quote)
            }
        )
    return groups


def score(groups, ranked_ids):
    """recall@n = share of quotes with a relevant chunk in the top n; MRR of the first hit."""

    def recall(ids):
        return sum(bool(group & set(ids)) for group in groups) / len(groups)

    relevant = set().union(*groups)
    first = next(
        (rank for rank, chunk_id in enumerate(ranked_ids, 1) if chunk_id in relevant),
        None,
    )
    return {
        "recall@1": recall(ranked_ids[:1]),
        "recall@3": recall(ranked_ids[:3]),
        "recall@5": recall(ranked_ids[:5]),
        "mrr": 1.0 / first if first else 0.0,
    }


def evaluate_answer(pipeline, case, groups, ranked):
    """End-to-end answer metrics for one case, using all reranked candidates as ask() does."""
    with contextlib.redirect_stdout(io.StringIO()):
        answer = pipeline.generate(case["question"], ranked)
    labels = {chunk["id"]: str(index) for index, chunk in enumerate(ranked, 1)}
    citations = cited_labels(answer)
    missing = missing_answer_facts(case, answer)
    judge = judge_answer_with_decision(case, answer)
    return {
        "facts_all": float(not missing),
        "fact_rate": 1 - len(missing) / len(case["answer_checks"]),
        # Every evidence quote is supported by at least one cited relevant chunk.
        "cites_evidence": float(
            all(
                any(labels.get(chunk_id) in citations for chunk_id in group)
                for group in groups
            )
        ),
        "invalid_cit": float(bool(citations - set(labels.values()))),
        "p_correct": judge["probability_correct"],
        "judge_ok": float(judge["verdict"] == "correct"),
        "context_chars": sum(len(chunk["document"]) for chunk in ranked),
    }


def evaluate_config(
    pipeline, client, cases, docs, chunk_size, overlap, generation=False
):
    # Fresh collection per configuration; the loaded models are reused.
    client.delete_collection(EVAL_COLLECTION)
    pipeline.collection = client.create_collection(
        name=EVAL_COLLECTION,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    pipeline.ingest(ROOT / "docs", chunk_size=chunk_size, overlap=overlap)

    spans_by_doc = {
        name: chunk_spans(text, chunk_size, overlap) for name, text in docs.items()
    }
    lengths = [end - start for spans in spans_by_doc.values() for start, end in spans]

    rows = []
    for case in cases:
        groups = relevant_chunks(case, docs, spans_by_doc)
        with contextlib.redirect_stdout(
            io.StringIO()
        ):  # retrieve() prints debug output
            candidates = pipeline.retrieve(case["question"])
        ranked = pipeline.rerank(case["question"], candidates)
        ranked_ids = [chunk["id"] for chunk in ranked]
        candidate_ids = [chunk["id"] for chunk in candidates]
        row = {
            **score(groups, ranked_ids),
            "cand_recall": sum(bool(g & set(candidate_ids)) for g in groups)
            / len(groups),
        }
        if generation:
            print(f"  generating: {case['id']}")
            row.update(evaluate_answer(pipeline, case, groups, ranked))
        rows.append(row)

    mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    return {
        "size": chunk_size,
        "overlap": overlap,
        "chunks": len(lengths),
        "avg_len": sum(lengths) / len(lengths),
        **mean,
    }


def print_table(headers, rows, title):
    print(f"\n=== {title} ===")
    width = max(11, max(len(h) for h in headers))
    print("  ".join(f"{h:>{width}}" for h in headers))
    for row in rows:
        print(
            "  ".join(
                f"{row[h]:>{width}.3f}"
                if isinstance(row[h], float)
                else f"{row[h]:>{width}}"
                for h in headers
            )
        )


def main():
    args = parse_args()
    load_dotenv(ROOT / ".env")
    if args.generation and missing_env(JUDGE_ENV):
        raise SystemExit(
            "Missing judge configuration: " + ", ".join(missing_env(JUDGE_ENV))
        )
    cases, docs = load_cases(), read_docs()
    configs = [(s, o) for s in args.sizes for o in args.overlaps if 0 <= o < s]

    with tempfile.TemporaryDirectory(prefix="rag_sweep_") as db_dir:
        print("Loading models...")
        pipeline, client = build_eval_pipeline(db_dir)
        if args.generation:
            attach_llm_client(pipeline)
        results = []
        try:
            for size, overlap in configs:
                print(f"Evaluating chunk_size={size}, overlap={overlap}...")
                results.append(
                    evaluate_config(
                        pipeline, client, cases, docs, size, overlap, args.generation
                    )
                )
        finally:
            client.delete_collection(EVAL_COLLECTION)

    print_table(
        [
            "size",
            "overlap",
            "chunks",
            "avg_len",
            "cand_recall",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr",
        ],
        results,
        "Retrieval",
    )
    if args.generation:
        print_table(
            [
                "size",
                "overlap",
                "context_chars",
                "facts_all",
                "fact_rate",
                "cites_evidence",
                "invalid_cit",
                "p_correct",
                "judge_ok",
            ],
            results,
            "Generation (end to end, all reranked candidates; rates are means over cases)",
        )
    print(
        f"\nRelevance: chunk covers >= {MIN_QUOTE_COVERAGE:.0%} of an evidence quote; "
        f"recall averaged over quotes per case, then over {len(cases)} cases (post-rerank)."
    )


if __name__ == "__main__":
    main()
