import os
from datetime import datetime
from types import SimpleNamespace

import chromadb
import pytest

import src.RAGPipeline as rag_module
from src.RAGPipeline import COLLECTION_NAME, RAGPipeline, file_metadata_for
from fakes import (
    FakeCollection, FakeEmbeddingModel, FakeReranker, make_llm_client, make_pipeline,
)


# --- file_metadata_for -------------------------------------------------------

def test_file_metadata_has_title_and_modified_time(tmp_path):
    path = tmp_path / "policy_v1.txt"
    path.write_text("text")
    timestamp = datetime(2025, 11, 18, 11, 51).timestamp()
    os.utime(path, (timestamp, timestamp))

    metadata = file_metadata_for(path)

    assert metadata["title"] == "policy_v1"
    assert metadata["modified_at"] == "2025-11-18T11:51:00"


def test_file_metadata_includes_created_at_when_birthtime_exists():
    path = SimpleNamespace(stem="doc", stat=lambda: SimpleNamespace(
        st_mtime=datetime(2026, 1, 2).timestamp(), st_birthtime=datetime(2025, 1, 2).timestamp(),
    ))
    assert file_metadata_for(path)["created_at"] == "2025-01-02T00:00:00"


def test_file_metadata_omits_created_at_without_birthtime():
    # Most Linux filesystems (e.g. CI runners) expose no st_birthtime.
    path = SimpleNamespace(stem="doc", stat=lambda: SimpleNamespace(st_mtime=0.0))
    assert "created_at" not in file_metadata_for(path)


# --- __init__ ----------------------------------------------------------------

def test_init_wires_models_collection_and_llm_client(monkeypatch, tmp_path):
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: ("collection", kwargs))
    monkeypatch.setattr(rag_module.chromadb, "PersistentClient", lambda path: client)
    monkeypatch.setattr(rag_module, "SentenceTransformer", lambda name: ("embedder", name))
    monkeypatch.setattr(rag_module, "CrossEncoder", lambda name: ("reranker", name))
    monkeypatch.setattr(rag_module, "OpenAI", lambda **kwargs: ("llm", kwargs))
    monkeypatch.setenv("FOUNDRY_BASE_URL", "https://example.test/openai/v1")
    monkeypatch.setenv("FOUNDRY_API_KEY", "key")
    monkeypatch.setenv("FOUNDRY_DEPLOYMENT_NAME", "deployment")

    rag = RAGPipeline(tmp_path)

    assert rag.embedding_model == ("embedder", rag_module.EMBEDDING_MODEL_NAME)
    assert rag.reranker == ("reranker", rag_module.RERANKER_MODEL_NAME)
    assert rag.collection[1]["name"] == COLLECTION_NAME
    assert rag.llm_client == ("llm", {"base_url": "https://example.test/openai/v1", "api_key": "key"})
    assert rag.llm_model == "deployment"


# --- ingest ------------------------------------------------------------------

@pytest.fixture
def ingest_pipeline(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.create_collection(name="unit_ingest", embedding_function=None)
    return make_pipeline(collection=collection, embedding_model=FakeEmbeddingModel())


def test_ingest_stores_chunks_with_ids_and_file_metadata(ingest_pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("First paragraph.\n\nSecond paragraph.")
    (docs / "b.md").write_text("Markdown doc.")
    (docs / "ignored.pdf").write_text("not ingested")

    ingest_pipeline.ingest(docs, chunk_size=20, overlap=0)

    stored = ingest_pipeline.collection.get(include=["documents", "metadatas"])
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    assert sorted(by_id) == ["a.txt:0000", "a.txt:0001", "b.md:0000"]
    assert by_id["a.txt:0001"][0] == "Second paragraph."
    metadata = by_id["a.txt:0001"][1]
    assert metadata["document_name"] == "a.txt"
    assert metadata["chunk_index"] == 1
    assert metadata["title"] == "a"
    assert "modified_at" in metadata


def test_reingest_removes_chunks_that_no_longer_exist(ingest_pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
    ingest_pipeline.ingest(docs, chunk_size=20, overlap=0)
    assert ingest_pipeline.collection.count() == 3

    (docs / "a.txt").write_text("First paragraph.")
    ingest_pipeline.ingest(docs, chunk_size=20, overlap=0)

    assert ingest_pipeline.collection.get()["ids"] == ["a.txt:0000"]


def test_ingest_without_documents_fails(ingest_pipeline, tmp_path):
    with pytest.raises(ValueError, match="no file name"):
        ingest_pipeline.ingest(tmp_path)


def test_ingest_of_empty_file_fails(ingest_pipeline, tmp_path):
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(ValueError, match="file is empty"):
        ingest_pipeline.ingest(tmp_path)


# --- retrieve ----------------------------------------------------------------

def vector_result(*rows):
    """Chroma query() result for (id, text, distance) rows."""
    return {
        "ids": [[row[0] for row in rows]],
        "documents": [[row[1] for row in rows]],
        "metadatas": [[{"document_name": row[0].split(":")[0]} for row in rows]],
        "distances": [[row[2] for row in rows]],
    }


def keyword_result(*rows):
    """Chroma get() result for (id, text) rows."""
    return {
        "ids": [row[0] for row in rows],
        "documents": [row[1] for row in rows],
        "metadatas": [{"document_name": row[0].split(":")[0]} for row in rows],
    }


def test_retrieve_merges_vector_and_keyword_hits(monkeypatch):
    monkeypatch.setattr(rag_module, "extract_keywords", lambda q: ["Hotel", "hotel", "EXP-310"])
    collection = FakeCollection(
        query_result=vector_result(("a:0", "vector only", 0.1), ("b:0", "Hotel limit", 0.2)),
        get_result=keyword_result(("b:0", "Hotel limit"), ("c:0", "Under exp-310 the HOTEL rule")),
    )
    rag = make_pipeline(collection=collection, embedding_model=FakeEmbeddingModel())

    candidates = rag.retrieve("hotel limit under EXP-310?")

    # Keywords differing only in case are queried once, case-insensitively and escaped.
    assert collection.get_calls[0]["where_document"] == {
        "$or": [{"$regex": "(?i)hotel"}, {"$regex": r"(?i)EXP\-310"}]
    }
    assert collection.get_calls[0]["limit"] == 5
    assert collection.query_calls[0]["n_results"] == 5

    by_id = {chunk["id"]: chunk for chunk in candidates}
    assert [chunk["id"] for chunk in candidates] == ["a:0", "b:0", "c:0"]
    assert by_id["a:0"]["retrieved_by"] == ["vector"]
    assert "matched_keywords" not in by_id["a:0"]
    assert by_id["b:0"]["retrieved_by"] == ["vector", "keyword"]
    assert by_id["b:0"]["distance"] == 0.2
    assert by_id["b:0"]["matched_keywords"] == ["hotel"]
    assert by_id["c:0"]["retrieved_by"] == ["keyword"]
    assert by_id["c:0"]["distance"] is None
    assert by_id["c:0"]["matched_keywords"] == ["hotel", "EXP-310"]


def test_retrieve_with_single_keyword_sends_plain_filter(monkeypatch):
    monkeypatch.setattr(rag_module, "extract_keywords", lambda q: ["meal"])
    collection = FakeCollection()
    make_pipeline(collection=collection, embedding_model=FakeEmbeddingModel()).retrieve("meal?")
    assert collection.get_calls[0]["where_document"] == {"$regex": "(?i)meal"}


def test_retrieve_without_keywords_uses_vector_search_only(monkeypatch):
    monkeypatch.setattr(rag_module, "extract_keywords", lambda q: [])
    collection = FakeCollection(query_result=vector_result(("a:0", "text", 0.3)))
    candidates = make_pipeline(
        collection=collection, embedding_model=FakeEmbeddingModel(),
    ).retrieve("is it?")
    assert collection.get_calls == []
    assert [chunk["id"] for chunk in candidates] == ["a:0"]


# --- rerank ------------------------------------------------------------------

def test_rerank_sorts_by_score_and_records_it():
    reranker = FakeReranker([0.1, 0.9, 0.5])
    chunks = [{"id": "a", "document": "x"}, {"id": "b", "document": "y"}, {"id": "c", "document": "z"}]

    ranked = make_pipeline(reranker=reranker).rerank("question", chunks)

    assert [chunk["id"] for chunk in ranked] == ["b", "c", "a"]
    assert [chunk["rerank_score"] for chunk in ranked] == [0.9, 0.5, 0.1]
    assert isinstance(ranked[0]["rerank_score"], float)
    assert reranker.pairs == [["question", "x"], ["question", "y"], ["question", "z"]]


def test_rerank_of_no_chunks_returns_empty_list():
    assert make_pipeline(reranker=FakeReranker()).rerank("q", []) == []


# --- generate ----------------------------------------------------------------

def test_generate_without_chunks_does_not_call_llm():
    client = make_llm_client()
    answer = make_pipeline(llm_client=client, llm_model="m").generate("q", [])
    assert answer == "No supporting documents were found for your question."
    client.responses.create.assert_not_called()


def test_generate_sends_labeled_sources_with_created_date():
    client = make_llm_client("  The limit is $500 [S1].  ")
    chunks = [
        {"id": "v2.txt:0002", "document": "Up to $500 per year.",
         "metadata": {"document_name": "v2.txt", "created_at": "2026-11-16T16:47:04"}},
        {"id": "x.txt:0000", "document": "No metadata.", "metadata": None},
    ]

    answer = make_pipeline(llm_client=client, llm_model="deployment").generate("Allowance?", chunks)

    assert answer == "The limit is $500 [S1]."
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "deployment"
    assert kwargs["input"].startswith("Question:\nAllowance?\n\nSource documents:\n[S1]\n")
    assert "Chunk ID: v2.txt:0002\nDocument: v2.txt\nCreated: 2026-11-16T16:47:04" in kwargs["input"]
    assert "[S2]\nChunk ID: x.txt:0000\nDocument: Not specified\nCreated: Not specified" in kwargs["input"]
    assert "Section:" not in kwargs["input"]
    assert "created date" in kwargs["instructions"]


@pytest.mark.parametrize("output_text", ["", "   ", None])
def test_generate_rejects_empty_model_output(output_text):
    rag = make_pipeline(llm_client=make_llm_client(output_text), llm_model="m")
    with pytest.raises(RuntimeError, match="empty response"):
        rag.generate("q", [{"id": "a", "document": "text"}])


# --- ask ---------------------------------------------------------------------

def test_ask_runs_retrieve_rerank_generate_in_order(monkeypatch):
    rag = make_pipeline()
    calls = []
    monkeypatch.setattr(rag, "retrieve", lambda q: calls.append(("retrieve", q)) or ["c"])
    monkeypatch.setattr(rag, "rerank", lambda q, c: calls.append(("rerank", c)) or ["r"])
    monkeypatch.setattr(rag, "generate", lambda q, c: calls.append(("generate", c)) or "answer")

    assert rag.ask("question") == "answer"
    assert calls == [("retrieve", "question"), ("rerank", ["c"]), ("generate", ["r"])]
