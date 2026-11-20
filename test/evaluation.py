"""Evaluate RAG retrieval and generation; optionally judge generation using Foundry."""

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_COLLECTION = "policy_evaluation"
GENERATION_ENV = ("FOUNDRY_BASE_URL", "FOUNDRY_API_KEY", "FOUNDRY_DEPLOYMENT_NAME")
JUDGE_ENV = ("FOUNDRY_ENDPOINT", "FOUNDRY_DECISION_DEPLOYMENT", "FOUNDRY_API_KEY")


def load_cases():
    return json.loads(Path(__file__).with_name("evaluation_cases.json").read_text())


def missing_env(names):
    return [name for name in names if not os.environ.get(name)]


def build_eval_pipeline(db_path, chunk_size=800, overlap=50):
    """Create a pipeline on an isolated DB with real models, then ingest docs.

    Bypasses RAGPipeline.__init__, which eagerly creates an API client, so
    retrieval can be evaluated without credentials. Returns (pipeline, client).
    """
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from src.rag_pipeline import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)
    client = chromadb.PersistentClient(path=str(db_path))
    pipeline.collection = client.create_collection(
        name=EVAL_COLLECTION,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    pipeline.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    pipeline.reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cpu")
    pipeline.ingest(ROOT / "docs", chunk_size=chunk_size, overlap=overlap)
    return pipeline, client


def attach_llm_client(pipeline):
    """Give the pipeline a Foundry client for generate(); fails if config is missing."""
    from openai import OpenAI

    missing = missing_env(GENERATION_ENV)
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))
    pipeline.llm_client = OpenAI(
        base_url=os.environ["FOUNDRY_BASE_URL"],
        api_key=os.environ["FOUNDRY_API_KEY"],
        timeout=60,
        max_retries=0,
    )
    pipeline.llm_model = os.environ["FOUNDRY_DEPLOYMENT_NAME"]
    return pipeline


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def rank_summary(rows):
    """Mean rank metrics before and after reranking for retrieval rows."""
    if not rows:
        return None
    return {
        stage: {
            metric: _mean(row[stage][metric] for row in rows)
            for metric in rows[0][stage]
        }
        for stage in ("pre_rerank", "post_rerank")
    }


def build_report(rows, **extra):
    """Summarize evaluation rows into the JSON report structure."""
    retrieval = [row for row in rows if row["stage"] == "retrieval"]
    generation = [row for row in rows if row["stage"] == "generation"]
    verdicts = [
        row["decision_judge"]["verdict"] for row in generation if row["decision_judge"]
    ]
    return {
        **extra,
        "retrieval_cases_evaluated": len(retrieval),
        "retrieval_k_values": sorted({row["k"] for row in retrieval}),
        "mean_recall_at_k": _mean(row["recall_at_k"] for row in retrieval),
        "rank_metrics": rank_summary(retrieval),
        "generation_cases_evaluated": len(generation),
        "judge_verdicts": (
            {verdict: verdicts.count(verdict) for verdict in sorted(set(verdicts))}
            if verdicts
            else None
        ),
        "results": rows,
    }


def gold_chunks(pipeline, case):
    """Fetch fixed evidence in a stable order; never send expected answers to the LLM."""
    ids = list(dict.fromkeys(case["context_ids"] + case["expected_chunk_ids"]))
    result = pipeline.collection.get(ids=ids, include=["documents", "metadatas"])
    records = {
        chunk_id: {"id": chunk_id, "document": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        )
    }
    missing = set(ids) - records.keys()
    if missing:
        raise AssertionError(f"Gold evidence is missing: {sorted(missing)}")
    for chunk_id, quote in case["evidence_quotes"].items():
        assert quote in records[chunk_id]["document"], (
            f"Gold evidence changed for {chunk_id}; review the dataset and chunk boundaries."
        )
    return [records[chunk_id] for chunk_id in ids]


RANK_CUTOFFS = (1, 3, 5)


def _recall(expected, ids):
    return len(expected.intersection(ids)) / len(expected)


def _reciprocal_rank(expected, ids):
    """1/rank of the first gold chunk; 0.0 if none is present."""
    return next(
        (1.0 / rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in expected),
        0.0,
    )


def _rank_metrics(expected, ids):
    metrics = {f"recall_at_{n}": _recall(expected, ids[:n]) for n in RANK_CUTOFFS}
    metrics["mrr"] = _reciprocal_rank(expected, ids)
    return metrics


def evaluate_retrieval(pipeline, case, k=5):
    """Measure actual production retrieval and reranking against fixed gold IDs.

    `retrieve()` returns vector hits in distance order followed by keyword-only
    hits, so the candidate order is the pre-rerank baseline. Comparing it with
    the reranked order shows whether reranking changes gold positions.
    """
    gold_chunks(pipeline, case)
    expected = set(case["expected_chunk_ids"])
    candidates = pipeline.retrieve(case["question"])
    ranked = pipeline.rerank(case["question"], candidates)
    candidate_ids = [chunk["id"] for chunk in candidates]
    ranked_ids = [chunk["id"] for chunk in ranked]
    top_ids = ranked_ids[:k]
    return {
        "case_id": case["id"],
        "stage": "retrieval",
        "question": case["question"],
        "expected_ids": sorted(expected),
        "candidate_ids": candidate_ids,
        "top_ids": top_ids,
        "k": k,
        "candidate_recall": _recall(expected, candidate_ids),
        "recall_at_k": _recall(expected, top_ids),
        "missing_ids": sorted(expected - set(top_ids)),
        "pre_rerank": _rank_metrics(expected, candidate_ids),
        "post_rerank": _rank_metrics(expected, ranked_ids),
        "gold_ranks": {
            chunk_id: {
                "pre_rerank": candidate_ids.index(chunk_id) + 1
                if chunk_id in candidate_ids
                else None,
                "post_rerank": ranked_ids.index(chunk_id) + 1
                if chunk_id in ranked_ids
                else None,
            }
            for chunk_id in sorted(expected)
        },
        "gold_retrieved_by": {
            chunk["id"]: chunk["retrieved_by"]
            for chunk in candidates
            if chunk["id"] in expected
        },
        "gold_matched_keywords": {
            chunk["id"]: chunk["matched_keywords"]
            for chunk in candidates
            if chunk["id"] in expected and "matched_keywords" in chunk
        },
    }


def judge_answer_with_decision(case, answer, *, threshold=0.5):
    """Judge reference-answer correctness using the Foundry SystemOne decision API.

    Uses the returned noul probability as P(correct), not a generated confidence.
    For a binary decision, confidence in the most likely verdict is max(p, 1-p).
    This judge does NOT validate grounding against retrieved documents.
    """
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0.5, 1.0]")

    endpoint = os.environ.get("FOUNDRY_ENDPOINT", "").rstrip("/")
    deployment = os.environ.get("FOUNDRY_DECISION_DEPLOYMENT", "")
    api_key = os.environ.get("FOUNDRY_API_KEY", "")
    if not (endpoint and deployment and api_key):
        raise RuntimeError(
            "Set FOUNDRY_ENDPOINT, FOUNDRY_DECISION_DEPLOYMENT, and FOUNDRY_API_KEY."
        )

    payload = {
        "model": deployment,  # Foundry deployment name, not necessarily model catalog name
        "state": {
            "question": case["question"],
            "reference_answer": case["expected_answer"],
            "generated_answer": answer,
        },
        "questions": {
            "answer_correctness": {
                "type": "noul",  # Yes/no question; returns probability of True
                "instructions": (
                    "Does the generated answer fully and correctly answer the question "
                    "according to the reference answer? Judge meaning, not word overlap. "
                    "Require all material facts and relationships (including dates, "
                    "amounts, units, negations, qualifications, and source conflicts). "
                    "Accept correct paraphrases, but reject missing details, factual "
                    "contradictions, and unsupported additions that change the answer."
                ),
                "criteria": {
                    "true": (
                        "All material facts requested by the question are correctly "
                        "covered and consistent with the reference answer."
                    ),
                    "false": (
                        "At least one requested material fact is missing, wrong, "
                        "contradicted, or materially misrepresented."
                    ),
                },
            }
        },
    }
    request = urllib.request.Request(
        f"{endpoint}/providers/microsoft/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Foundry decision API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach Foundry decision API") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Foundry decision API returned invalid JSON") from exc

    answer_data = result.get("answers", {}).get("answer_correctness", {})
    if answer_data.get("type") != "noul":
        raise RuntimeError("Foundry decision API did not return a noul answer")
    p = answer_data.get("noul")
    if (
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 <= p <= 1
    ):
        raise RuntimeError("Foundry decision API returned an invalid probability")

    verdict = (
        "correct"
        if p >= threshold
        else "incorrect"
        if p <= 1.0 - threshold
        else "review"
    )
    return {
        "model": result.get("model", deployment),
        "probability_correct": p,
        "confidence": max(p, 1.0 - p),
        "verdict": verdict,
        "threshold": threshold,
    }


def missing_answer_facts(case, answer):
    """Names of the case's regex answer checks that the answer does not satisfy."""
    return [
        check["name"]
        for check in case["answer_checks"]
        if not re.search(check["pattern"], answer, re.IGNORECASE | re.DOTALL)
    ]


def cited_labels(answer):
    """Source label numbers cited as [S<n>] in the answer, as strings."""
    return set(re.findall(r"\[S(\d+)\]", answer))


def evaluate_generation(pipeline, case, *, run_judge=False, decision_threshold=0.5):
    """Generate from gold evidence and evaluate it with lexical checks.

    The Microsoft-Decision-1.1 judge is opt-in (`run_judge=True`) because it needs
    a separate decision-model deployment and endpoint configuration.
    """
    chunks = gold_chunks(pipeline, case)
    answer = pipeline.generate(case["question"], chunks)
    missing_facts = missing_answer_facts(case, answer)
    citations = cited_labels(answer)
    allowed = {str(i) for i in range(1, len(chunks) + 1)}
    required = {
        str(i)
        for i, chunk in enumerate(chunks, 1)
        if chunk["id"] in case["expected_chunk_ids"]
    }
    decision_judge = (
        judge_answer_with_decision(case, answer, threshold=decision_threshold)
        if run_judge
        else None
    )
    return {
        "case_id": case["id"],
        "stage": "generation",
        "context_mode": "gold",
        "question": case["question"],
        "context_ids": [chunk["id"] for chunk in chunks],
        "answer": answer,
        "missing_facts": missing_facts,
        "invalid_citations": sorted(citations - allowed),
        "missing_evidence_citations": sorted(required - citations),
        "decision_judge": decision_judge,
    }
