import io
import json
import urllib.error
from email.message import Message

import evaluation
import openai
import pytest
import sentence_transformers
from fakes import FakeCollection, FakeEmbeddingModel, FakeReranker, make_pipeline

JUDGE_VARS = {
    "FOUNDRY_ENDPOINT": "https://res.services.ai.azure.com/",
    "FOUNDRY_DECISION_DEPLOYMENT": "Decision-1.1",
    "FOUNDRY_API_KEY": "key",
}

CASE = {
    "id": "case",
    "question": "What is the allowance?",
    "expected_answer": "$500 per year.",
    "expected_chunk_ids": ["doc:0002"],
    "context_ids": ["doc:0000"],
    "evidence_quotes": {"doc:0002": "up to $500"},
    "answer_checks": [
        {"name": "amount", "pattern": r"\$\s*500"},
        {"name": "annual", "pattern": "per year|annual"},
    ],
}

RECORDS = {
    "doc:0000": {"document": "Header", "metadata": {"document_name": "doc"}},
    "doc:0002": {
        "document": "Employees may claim up to $500 per year.",
        "metadata": {},
    },
}


@pytest.fixture
def judge_env(monkeypatch):
    for name, value in JUDGE_VARS.items():
        monkeypatch.setenv(name, value)


def respond_with(monkeypatch, body, captured=None):
    """Make urlopen return `body` (dict -> JSON, bytes as-is) or raise it if it is an exception."""

    def fake_urlopen(request, timeout):
        if captured is not None:
            captured["request"] = request
            captured["timeout"] = timeout
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(
            body if isinstance(body, bytes) else json.dumps(body).encode()
        )

    monkeypatch.setattr(evaluation.urllib.request, "urlopen", fake_urlopen)


def noul(p, **extra):
    return {
        "model": "microsoft-decision-1.1",
        "answers": {"answer_correctness": {"type": "noul", "noul": p, **extra}},
    }


# --- small helpers -----------------------------------------------------------


def test_load_cases_reads_the_dataset():
    cases = evaluation.load_cases()
    assert len(cases) == 10
    assert {
        "id",
        "question",
        "expected_chunk_ids",
        "evidence_quotes",
        "answer_checks",
    } <= cases[0].keys()


def test_missing_env_lists_unset_or_empty_names(monkeypatch):
    monkeypatch.setenv("SET_VAR", "x")
    monkeypatch.setenv("EMPTY_VAR", "")
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert evaluation.missing_env(["SET_VAR", "EMPTY_VAR", "UNSET_VAR"]) == [
        "EMPTY_VAR",
        "UNSET_VAR",
    ]


def test_rank_metrics_recall_and_reciprocal_rank():
    metrics = evaluation._rank_metrics({"b", "d"}, ["a", "b", "c", "d"])
    assert metrics == {
        "recall_at_1": 0.0,
        "recall_at_3": 0.5,
        "recall_at_5": 1.0,
        "mrr": 0.5,
    }
    assert evaluation._reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_missing_answer_facts_and_cited_labels():
    assert evaluation.missing_answer_facts(CASE, "It is $500 [S2].") == ["annual"]
    assert evaluation.missing_answer_facts(CASE, "$ 500 PER YEAR") == []
    assert evaluation.cited_labels(
        "A [S1], B [S12] and [S1][S3]; not S4 or [S 5]."
    ) == {"1", "12", "3"}


# --- pipeline setup ----------------------------------------------------------


def test_attach_llm_client_requires_configuration(monkeypatch):
    for name in evaluation.GENERATION_ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="FOUNDRY_BASE_URL"):
        evaluation.attach_llm_client(make_pipeline())


def test_attach_llm_client_creates_client(monkeypatch):
    monkeypatch.setenv("FOUNDRY_BASE_URL", "https://x/openai/v1")
    monkeypatch.setenv("FOUNDRY_API_KEY", "key")
    monkeypatch.setenv("FOUNDRY_DEPLOYMENT_NAME", "model")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: kwargs)

    pipeline = evaluation.attach_llm_client(make_pipeline())

    assert pipeline.llm_client == {
        "base_url": "https://x/openai/v1",
        "api_key": "key",
        "timeout": 60,
        "max_retries": 0,
    }
    assert pipeline.llm_model == "model"


def test_build_eval_pipeline_ingests_docs_into_isolated_db(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeEmbeddingModel
    )
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeReranker)

    pipeline, client = evaluation.build_eval_pipeline(
        tmp_path, chunk_size=800, overlap=50
    )

    assert pipeline.collection.name == evaluation.EVAL_COLLECTION
    ids = pipeline.collection.get()["ids"]
    for case in evaluation.load_cases():
        assert set(case["expected_chunk_ids"] + case["context_ids"]) <= set(ids)
    client.delete_collection(evaluation.EVAL_COLLECTION)


# --- gold evidence and retrieval ---------------------------------------------


def test_gold_chunks_returns_context_then_evidence():
    pipeline = make_pipeline(collection=FakeCollection(records=RECORDS))
    assert [chunk["id"] for chunk in evaluation.gold_chunks(pipeline, CASE)] == [
        "doc:0000",
        "doc:0002",
    ]


def test_gold_chunks_fails_when_a_chunk_is_missing():
    pipeline = make_pipeline(
        collection=FakeCollection(records={"doc:0000": RECORDS["doc:0000"]})
    )
    with pytest.raises(AssertionError, match="Gold evidence is missing"):
        evaluation.gold_chunks(pipeline, CASE)


def test_gold_chunks_fails_when_evidence_text_changed():
    records = {**RECORDS, "doc:0002": {"document": "Different text.", "metadata": {}}}
    with pytest.raises(AssertionError, match="Gold evidence changed"):
        evaluation.gold_chunks(
            make_pipeline(collection=FakeCollection(records=records)), CASE
        )


def test_evaluate_retrieval_compares_candidate_and_reranked_order(monkeypatch):
    pipeline = make_pipeline(collection=FakeCollection(records=RECORDS))
    candidates = [
        {"id": "doc:0001", "document": "x", "retrieved_by": ["vector"]},
        {
            "id": "doc:0002",
            "document": "y",
            "retrieved_by": ["vector", "keyword"],
            "matched_keywords": ["allowance"],
        },
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda q: candidates)
    monkeypatch.setattr(pipeline, "rerank", lambda q, c: [c[1], c[0]])

    row = evaluation.evaluate_retrieval(pipeline, CASE, k=1)

    assert row["candidate_ids"] == ["doc:0001", "doc:0002"]
    assert row["top_ids"] == ["doc:0002"]
    assert row["recall_at_k"] == 1.0
    assert row["missing_ids"] == []
    assert row["pre_rerank"]["mrr"] == 0.5
    assert row["post_rerank"]["mrr"] == 1.0
    assert row["gold_ranks"] == {"doc:0002": {"pre_rerank": 2, "post_rerank": 1}}
    assert row["gold_retrieved_by"] == {"doc:0002": ["vector", "keyword"]}
    assert row["gold_matched_keywords"] == {"doc:0002": ["allowance"]}


def test_evaluate_retrieval_reports_missing_gold(monkeypatch):
    pipeline = make_pipeline(collection=FakeCollection(records=RECORDS))
    monkeypatch.setattr(
        pipeline, "retrieve", lambda q: [{"id": "doc:0001", "retrieved_by": ["vector"]}]
    )
    monkeypatch.setattr(pipeline, "rerank", lambda q, c: c)

    row = evaluation.evaluate_retrieval(pipeline, CASE, k=5)

    assert row["recall_at_k"] == 0.0
    assert row["missing_ids"] == ["doc:0002"]
    assert row["gold_ranks"] == {"doc:0002": {"pre_rerank": None, "post_rerank": None}}


# --- report ------------------------------------------------------------------


def retrieval_row(recall, mrr):
    metrics = {
        "recall_at_1": recall,
        "recall_at_3": recall,
        "recall_at_5": recall,
        "mrr": mrr,
    }
    return {
        "stage": "retrieval",
        "k": 5,
        "recall_at_k": recall,
        "pre_rerank": metrics,
        "post_rerank": metrics,
    }


def test_build_report_summarizes_rows():
    rows = [
        retrieval_row(1.0, 1.0),
        retrieval_row(0.0, 0.5),
        {"stage": "generation", "decision_judge": {"verdict": "correct"}},
        {"stage": "generation", "decision_judge": {"verdict": "correct"}},
        {"stage": "generation", "decision_judge": None},
    ]
    report = evaluation.build_report(rows, pytest_exit_status=0)
    assert report["pytest_exit_status"] == 0
    assert report["retrieval_cases_evaluated"] == 2
    assert report["mean_recall_at_k"] == 0.5
    assert report["rank_metrics"]["post_rerank"]["mrr"] == 0.75
    assert report["generation_cases_evaluated"] == 3
    assert report["judge_verdicts"] == {"correct": 2}


def test_build_report_without_rows():
    report = evaluation.build_report([])
    assert report["mean_recall_at_k"] is None
    assert report["rank_metrics"] is None
    assert report["judge_verdicts"] is None


# --- decision judge ----------------------------------------------------------


@pytest.mark.parametrize(
    "p, threshold, verdict",
    [
        (0.9, 0.5, "correct"),
        (0.5, 0.5, "correct"),
        (0.1, 0.5, "incorrect"),
        (0.5, 0.8, "review"),
        (0.15, 0.8, "incorrect"),
    ],
)
def test_judge_verdict_from_probability(monkeypatch, judge_env, p, threshold, verdict):
    respond_with(monkeypatch, noul(p))
    result = evaluation.judge_answer_with_decision(CASE, "answer", threshold=threshold)
    assert result["verdict"] == verdict
    assert result["probability_correct"] == p
    assert result["confidence"] == max(p, 1 - p)
    assert result["model"] == "microsoft-decision-1.1"


def test_judge_sends_expected_request(monkeypatch, judge_env):
    captured = {}
    respond_with(monkeypatch, noul(0.7), captured)

    evaluation.judge_answer_with_decision(CASE, "generated")

    request = captured["request"]
    assert (
        request.full_url
        == "https://res.services.ai.azure.com/providers/microsoft/v1/systemone"
    )
    assert request.get_method() == "POST"
    assert request.get_header("Api-key") == "key"
    body = json.loads(request.data)
    assert body["model"] == "Decision-1.1"
    assert body["state"] == {
        "question": CASE["question"],
        "reference_answer": CASE["expected_answer"],
        "generated_answer": "generated",
    }
    assert body["questions"]["answer_correctness"]["type"] == "noul"
    assert captured["timeout"] == 60


@pytest.mark.parametrize("threshold", [0.49, 1.01])
def test_judge_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        evaluation.judge_answer_with_decision(CASE, "a", threshold=threshold)


def test_judge_requires_configuration(monkeypatch):
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="Set FOUNDRY_ENDPOINT"):
        evaluation.judge_answer_with_decision(CASE, "a")


@pytest.mark.parametrize(
    "body, message",
    [
        (urllib.error.HTTPError("url", 422, "bad", Message(), None), "HTTP 422"),
        (urllib.error.URLError("down"), "Could not reach"),
        (b"not json", "invalid JSON"),
        (
            {"answers": {"answer_correctness": {"type": "choice"}}},
            "did not return a noul",
        ),
        ({}, "did not return a noul"),
        (noul(True), "invalid probability"),
        (noul("0.7"), "invalid probability"),
        (noul(1.5), "invalid probability"),
        (noul(float("nan")), "invalid probability"),
    ],
)
def test_judge_reports_api_failures(monkeypatch, judge_env, body, message):
    respond_with(monkeypatch, body)
    with pytest.raises(RuntimeError, match=message):
        evaluation.judge_answer_with_decision(CASE, "a")


def test_judge_falls_back_to_deployment_name_for_model(monkeypatch, judge_env):
    respond_with(
        monkeypatch, {"answers": {"answer_correctness": {"type": "noul", "noul": 0.6}}}
    )
    assert evaluation.judge_answer_with_decision(CASE, "a")["model"] == "Decision-1.1"


# --- generation --------------------------------------------------------------


def generation_pipeline(answer):
    pipeline = make_pipeline(collection=FakeCollection(records=RECORDS))
    pipeline.generate = lambda question, chunks: answer
    return pipeline


def test_evaluate_generation_checks_facts_and_citations_without_judge():
    row = evaluation.evaluate_generation(generation_pipeline("$500 [S1] [S7]"), CASE)
    assert row["context_ids"] == ["doc:0000", "doc:0002"]
    assert row["missing_facts"] == ["annual"]
    assert row["invalid_citations"] == ["7"]
    assert row["missing_evidence_citations"] == ["2"]
    assert row["decision_judge"] is None


def test_evaluate_generation_runs_judge_when_requested(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "judge_answer_with_decision",
        lambda case, answer, threshold: {"verdict": "correct", "threshold": threshold},
    )
    row = evaluation.evaluate_generation(
        generation_pipeline("$500 per year [S2]"),
        CASE,
        run_judge=True,
        decision_threshold=0.6,
    )
    assert row["missing_facts"] == []
    assert row["missing_evidence_citations"] == []
    assert row["decision_judge"] == {"verdict": "correct", "threshold": 0.6}
