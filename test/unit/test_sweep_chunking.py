from unittest.mock import MagicMock

import pytest

import src.chunking
import sweep_chunking
from fakes import make_pipeline

DOCS = {"doc.txt": "Intro paragraph here.\n\nThe hotel limit is $180 per night.\n\nMeals are $65."}
CASE = {
    "id": "hotel",
    "question": "What is the hotel limit?",
    "expected_answer": "$180",
    "evidence_quotes": {"doc.txt:0001": "The hotel limit is $180 per night."},
    "answer_checks": [{"name": "amount", "pattern": r"\$180"}, {"name": "night", "pattern": "night"}],
}


def spans(chunk_size=40, overlap=0):
    return {"doc.txt": sweep_chunking.chunk_spans(DOCS["doc.txt"], chunk_size, overlap)}


# --- relevance and scoring ---------------------------------------------------

def test_read_docs_loads_every_document():
    docs = sweep_chunking.read_docs()
    assert len(docs) == 4
    assert all("\r" not in text for text in docs.values())


def test_chunk_spans_point_at_the_chunk_text():
    text = DOCS["doc.txt"]
    for start, end in spans()["doc.txt"]:
        assert text[start:end].strip() == text[start:end]
    assert text[slice(*spans()["doc.txt"][1])] == "The hotel limit is $180 per night."


def test_chunk_spans_fail_if_a_chunk_is_not_in_the_text(monkeypatch):
    monkeypatch.setattr(src.chunking, "chunk_text_recursive", lambda text, **kwargs: ["missing"])
    with pytest.raises(ValueError, match="Chunk not found"):
        sweep_chunking.chunk_spans("some text", 10, 0)


def test_relevant_chunks_include_chunks_covering_half_the_quote():
    # 20-char chunks split the 34-char quote; only pieces covering >= 17 chars count.
    groups = sweep_chunking.relevant_chunks(CASE, DOCS, spans(chunk_size=20))
    assert len(groups) == 1
    text, quote = DOCS["doc.txt"], CASE["evidence_quotes"]["doc.txt:0001"]
    q_start = text.find(quote)
    for chunk_id in groups[0]:
        start, end = spans(chunk_size=20)["doc.txt"][int(chunk_id.split(":")[1])]
        assert min(end, q_start + len(quote)) - max(start, q_start) >= len(quote) / 2


def test_relevant_chunks_fail_for_unknown_quote():
    case = {**CASE, "evidence_quotes": {"doc.txt:0001": "not in the document"}}
    with pytest.raises(ValueError, match="Evidence quote not found"):
        sweep_chunking.relevant_chunks(case, DOCS, spans())


def test_score_recall_and_mrr():
    groups = [{"a"}, {"c", "d"}]
    assert sweep_chunking.score(groups, ["x", "a", "y", "d"]) == {
        "recall@1": 0.0, "recall@3": 0.5, "recall@5": 1.0, "mrr": 0.5,
    }
    assert sweep_chunking.score(groups, ["x"])["mrr"] == 0.0


# --- end-to-end answer and configuration ------------------------------------

def test_evaluate_answer_scores_facts_citations_and_judge(monkeypatch):
    monkeypatch.setattr(sweep_chunking, "judge_answer_with_decision",
                        lambda case, answer: {"probability_correct": 0.7, "verdict": "correct"})
    pipeline = make_pipeline()
    pipeline.generate = lambda question, ranked: "The limit is $180 [S2] [S9]."
    ranked = [{"id": "doc.txt:0000", "document": "abc"}, {"id": "doc.txt:0001", "document": "defg"}]

    result = sweep_chunking.evaluate_answer(pipeline, CASE, [{"doc.txt:0001"}], ranked)

    assert result == {
        "facts_all": 0.0, "fact_rate": 0.5, "cites_evidence": 1.0, "invalid_cit": 1.0,
        "p_correct": 0.7, "judge_ok": 1.0, "context_chars": 7,
    }


def fake_sweep_pipeline(ranked_ids):
    pipeline = make_pipeline()
    pipeline.ingest = MagicMock()
    pipeline.retrieve = lambda question: [{"id": chunk_id, "document": "x"} for chunk_id in ranked_ids]
    pipeline.rerank = lambda question, candidates: list(reversed(candidates))
    pipeline.generate = lambda question, ranked: "$180 per night [S1]"
    return pipeline


def test_evaluate_config_rebuilds_collection_and_averages(monkeypatch):
    monkeypatch.setattr(sweep_chunking, "judge_answer_with_decision",
                        lambda case, answer: {"probability_correct": 0.9, "verdict": "correct"})
    client = MagicMock()
    pipeline = fake_sweep_pipeline(["doc.txt:0001", "doc.txt:0000"])

    row = sweep_chunking.evaluate_config(pipeline, client, [CASE], DOCS, 40, 0, generation=True)

    client.delete_collection.assert_called_once_with(sweep_chunking.EVAL_COLLECTION)
    pipeline.ingest.assert_called_once_with(sweep_chunking.ROOT / "docs", chunk_size=40, overlap=0)
    assert row["size"] == 40 and row["overlap"] == 0 and row["chunks"] == 3
    assert row["cand_recall"] == 1.0
    assert row["recall@1"] == 0.0  # reranked order puts doc.txt:0000 first
    assert row["mrr"] == 0.5
    assert row["p_correct"] == 0.9
    assert row["cites_evidence"] == 0.0  # [S1] is doc.txt:0000 after reranking


def test_main_runs_each_valid_configuration(monkeypatch, capsys):
    monkeypatch.setattr(sweep_chunking, "load_dotenv", lambda path: None)
    for name in sweep_chunking.JUDGE_ENV:
        monkeypatch.setenv(name, "x")
    client = MagicMock()
    monkeypatch.setattr(sweep_chunking, "build_eval_pipeline",
                        lambda db_dir: (fake_sweep_pipeline([]), client))
    monkeypatch.setattr(sweep_chunking, "attach_llm_client", lambda pipeline: pipeline)
    monkeypatch.setattr(sweep_chunking, "judge_answer_with_decision",
                        lambda case, answer: {"probability_correct": 0.4, "verdict": "incorrect"})
    monkeypatch.setattr("sys.argv", ["sweep_chunking.py", "--sizes", "100", "300",
                                     "--overlaps", "0", "100", "--generation"])

    sweep_chunking.main()

    out = capsys.readouterr().out
    # (100, 100) is skipped because overlap must be smaller than the size.
    assert "chunk_size=100, overlap=0" in out
    assert "chunk_size=100, overlap=100" not in out
    assert "chunk_size=300, overlap=100" in out
    assert "=== Retrieval ===" in out and "=== Generation" in out
    assert client.delete_collection.call_count == 4  # once per configuration + final cleanup


def test_main_requires_judge_configuration_for_generation(monkeypatch):
    monkeypatch.setattr(sweep_chunking, "load_dotenv", lambda path: None)
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.setattr("sys.argv", ["sweep_chunking.py", "--generation"])
    with pytest.raises(SystemExit, match="Missing judge configuration"):
        sweep_chunking.main()
